"""Atomic, manifest-committed Parquet shards."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True, slots=True)
class TableStats:
    path: str
    row_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ShardManifest:
    shard_id: int
    prompt_ids: tuple[str, ...]
    tables: Mapping[str, TableStats]


def _stem(shard_id: int) -> str:
    if type(shard_id) is not int or shard_id < 0:
        raise ValueError("shard_id must be a non-negative integer")
    return f"shard-{shard_id:05d}"


def _table_path(root: Path, table: str, shard_id: int) -> Path:
    return root / table / f"{_stem(shard_id)}.parquet"


def _temporary_table_path(root: Path, table: str, shard_id: int) -> Path:
    return root / table / f"{_stem(shard_id)}.parquet.tmp"


def _manifest_path(root: Path, shard_id: int) -> Path:
    return root / "manifests" / f"{_stem(shard_id)}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_payload(manifest: ShardManifest) -> dict[str, Any]:
    return {
        "shard_id": manifest.shard_id,
        "prompt_ids": list(manifest.prompt_ids),
        "tables": {
            table: {
                "path": stats.path,
                "row_count": stats.row_count,
                "sha256": stats.sha256,
            }
            for table, stats in sorted(manifest.tables.items())
        },
    }


def read_shard_manifest(root: Path, *, shard_id: int) -> ShardManifest:
    """Read one completion manifest."""
    payload = json.loads(_manifest_path(root, shard_id).read_text(encoding="utf-8"))
    tables = {
        table: TableStats(
            path=str(stats["path"]),
            row_count=int(stats["row_count"]),
            sha256=str(stats["sha256"]),
        )
        for table, stats in payload["tables"].items()
    }
    return ShardManifest(
        shard_id=int(payload["shard_id"]),
        prompt_ids=tuple(str(prompt_id) for prompt_id in payload["prompt_ids"]),
        tables=tables,
    )


def validate_shard_manifest(
    root: Path,
    manifest: ShardManifest,
    *,
    schemas: Mapping[str, pa.Schema],
    required_tables: Sequence[str],
) -> ShardManifest:
    """Validate table presence, schema, row counts, and checksums."""
    required = tuple(required_tables)
    if set(manifest.tables) != set(required):
        raise ValueError("Shard manifest does not contain every required table")
    for table in required:
        if table not in schemas:
            raise ValueError(f"No schema declared for required table {table!r}")
        stats = manifest.tables[table]
        path = root / stats.path
        expected = _table_path(root, table, manifest.shard_id)
        if path.resolve() != expected.resolve() or not path.is_file():
            raise ValueError(f"Shard table {table!r} is missing or misaddressed")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != schemas[table]:
            raise ValueError(f"Shard table {table!r} has the wrong Arrow schema")
        if parquet.metadata.num_rows != stats.row_count:
            raise ValueError(f"Shard table {table!r} row count does not match manifest")
        if _sha256(path) != stats.sha256:
            raise ValueError(f"Shard table {table!r} checksum does not match manifest")
    return manifest


def is_shard_complete(
    root: Path,
    *,
    shard_id: int,
    schemas: Mapping[str, pa.Schema],
    required_tables: Sequence[str],
) -> bool:
    """Return true only for a manifest-backed, fully validated shard."""
    try:
        manifest = read_shard_manifest(root, shard_id=shard_id)
        if manifest.shard_id != shard_id:
            return False
        validate_shard_manifest(
            root,
            manifest,
            schemas=schemas,
            required_tables=required_tables,
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False
    return True


class ShardWriter:
    """Append prompt-sized batches and expose the shard only at manifest commit."""

    def __init__(
        self,
        root: Path,
        *,
        shard_id: int,
        schemas: Mapping[str, pa.Schema],
        required_tables: Sequence[str],
        prompt_ids: Sequence[str],
    ) -> None:
        self.root = Path(root)
        self.shard_id = shard_id
        _stem(shard_id)
        self.schemas = dict(schemas)
        self.required_tables = tuple(required_tables)
        if set(self.schemas) != set(self.required_tables):
            raise ValueError("schemas must match required_tables exactly")
        self.prompt_ids = tuple(prompt_ids)
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._row_counts = {table: 0 for table in self.required_tables}
        self._committed = False
        self._prepare_paths()

    def _prepare_paths(self) -> None:
        manifest_path = _manifest_path(self.root, self.shard_id)
        if manifest_path.exists():
            raise FileExistsError(
                f"Shard {self.shard_id} already has a completion manifest"
            )
        for table in self.required_tables:
            directory = self.root / table
            directory.mkdir(parents=True, exist_ok=True)
            for path in (
                _table_path(self.root, table, self.shard_id),
                _temporary_table_path(self.root, table, self.shard_id),
            ):
                if path.exists():
                    path.unlink()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        if temporary_manifest.exists():
            temporary_manifest.unlink()

    def append(self, table: str, batch: pa.RecordBatch) -> None:
        """Write one typed batch immediately; retain no row dictionaries."""
        if self._committed:
            raise RuntimeError("Cannot append to a committed shard")
        if table not in self.schemas:
            raise ValueError(f"Unknown shard table: {table!r}")
        if not isinstance(batch, pa.RecordBatch):
            raise TypeError("ShardWriter.append requires an Arrow RecordBatch")
        if batch.schema != self.schemas[table]:
            raise ValueError(f"Batch schema does not match table {table!r}")
        writer = self._writers.get(table)
        if writer is None:
            writer = pq.ParquetWriter(
                _temporary_table_path(self.root, table, self.shard_id),
                self.schemas[table],
                compression="zstd",
            )
            self._writers[table] = writer
        writer.write_batch(batch)
        self._row_counts[table] += batch.num_rows

    def commit(self) -> ShardManifest:
        """Close and rename all tables, then atomically write the manifest last."""
        missing = [
            table for table in self.required_tables if table not in self._writers
        ]
        if missing:
            raise RuntimeError(f"Cannot commit shard; missing tables: {missing}")
        for writer in self._writers.values():
            writer.close()
        self._writers.clear()

        stats: dict[str, TableStats] = {}
        for table in self.required_tables:
            temporary = _temporary_table_path(self.root, table, self.shard_id)
            final = _table_path(self.root, table, self.shard_id)
            os.replace(temporary, final)
            relative = final.relative_to(self.root).as_posix()
            stats[table] = TableStats(
                path=relative,
                row_count=self._row_counts[table],
                sha256=_sha256(final),
            )
        manifest = ShardManifest(
            shard_id=self.shard_id,
            prompt_ids=self.prompt_ids,
            tables=stats,
        )
        validate_shard_manifest(
            self.root,
            manifest,
            schemas=self.schemas,
            required_tables=self.required_tables,
        )
        path = _manifest_path(self.root, self.shard_id)
        temporary_manifest = path.with_suffix(".json.tmp")
        temporary_manifest.write_text(
            json.dumps(_manifest_payload(manifest), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_manifest, path)
        self._committed = True
        return manifest
