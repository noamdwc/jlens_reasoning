from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pytest

from jlens_reasoning.experiments_utils.storage import (
    ShardWriter,
    is_shard_complete,
    read_shard_manifest,
    validate_shard_manifest,
)

SCHEMAS = {
    "left": pa.schema(
        [pa.field("prompt_id", pa.string()), pa.field("value", pa.int32())]
    ),
    "right": pa.schema(
        [pa.field("prompt_id", pa.string()), pa.field("label", pa.string())]
    ),
}


def _batch(table: str, prompt_id: str, value: int = 1) -> pa.RecordBatch:
    if table == "left":
        return pa.RecordBatch.from_arrays(
            [pa.array([prompt_id]), pa.array([value], type=pa.int32())],
            schema=SCHEMAS[table],
        )
    return pa.RecordBatch.from_arrays(
        [pa.array([prompt_id]), pa.array([f"label-{value}"])],
        schema=SCHEMAS[table],
    )


def test_shard_is_incomplete_until_every_table_and_manifest_are_committed(
    tmp_path: Path,
) -> None:
    writer = ShardWriter(
        tmp_path,
        shard_id=3,
        schemas=SCHEMAS,
        required_tables=("left", "right"),
        prompt_ids=("p1",),
    )
    writer.append("left", _batch("left", "p1"))

    with pytest.raises(RuntimeError, match="right"):
        writer.commit()

    assert (
        is_shard_complete(
            tmp_path,
            shard_id=3,
            schemas=SCHEMAS,
            required_tables=("left", "right"),
        )
        is False
    )


def test_complete_shard_round_trips_manifest_counts_and_checksums(
    tmp_path: Path,
) -> None:
    writer = ShardWriter(
        tmp_path,
        shard_id=0,
        schemas=SCHEMAS,
        required_tables=("left", "right"),
        prompt_ids=("p1", "p2"),
    )
    writer.append("left", _batch("left", "p1", 1))
    writer.append("left", _batch("left", "p2", 2))
    writer.append("right", _batch("right", "p1", 1))
    writer.append("right", _batch("right", "p2", 2))

    manifest = writer.commit()

    assert manifest.shard_id == 0
    assert manifest.prompt_ids == ("p1", "p2")
    assert manifest.tables["left"].row_count == 2
    assert manifest.tables["right"].row_count == 2
    assert is_shard_complete(
        tmp_path,
        shard_id=0,
        schemas=SCHEMAS,
        required_tables=("left", "right"),
    )
    assert (
        validate_shard_manifest(
            tmp_path,
            read_shard_manifest(tmp_path, shard_id=0),
            schemas=SCHEMAS,
            required_tables=("left", "right"),
        )
        == manifest
    )
    manifest_path = tmp_path / "manifests" / "shard-00000.json"
    assert manifest_path.stat().st_mtime_ns >= max(
        (tmp_path / table / "shard-00000.parquet").stat().st_mtime_ns
        for table in SCHEMAS
    )


def test_corrupt_or_missing_table_never_counts_as_complete(tmp_path: Path) -> None:
    writer = ShardWriter(
        tmp_path,
        shard_id=1,
        schemas=SCHEMAS,
        required_tables=("left", "right"),
        prompt_ids=("p1",),
    )
    writer.append("left", _batch("left", "p1"))
    writer.append("right", _batch("right", "p1"))
    writer.commit()
    (tmp_path / "left" / "shard-00001.parquet").write_bytes(b"corrupt")

    assert not is_shard_complete(
        tmp_path,
        shard_id=1,
        schemas=SCHEMAS,
        required_tables=("left", "right"),
    )


def test_new_writer_discards_only_its_own_incomplete_shard_files(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "left" / "shard-00002.parquet.tmp"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"partial")
    unrelated = tmp_path / "left" / "shard-00099.parquet"
    unrelated.write_bytes(b"keep")

    ShardWriter(
        tmp_path,
        shard_id=2,
        schemas=SCHEMAS,
        required_tables=("left", "right"),
        prompt_ids=("p1",),
    )

    assert not partial.exists()
    assert unrelated.read_bytes() == b"keep"


def test_manifest_json_is_saved_last_and_is_parseable(tmp_path: Path) -> None:
    writer = ShardWriter(
        tmp_path,
        shard_id=4,
        schemas=SCHEMAS,
        required_tables=("left", "right"),
        prompt_ids=("p1",),
    )
    writer.append("left", _batch("left", "p1"))
    writer.append("right", _batch("right", "p1"))
    writer.commit()

    payload = json.loads(
        (tmp_path / "manifests" / "shard-00004.json").read_text(encoding="utf-8")
    )
    assert payload["shard_id"] == 4
    assert set(payload["tables"]) == {"left", "right"}
