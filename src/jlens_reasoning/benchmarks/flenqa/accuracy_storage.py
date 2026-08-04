"""Typed storage for paper-compatible FLenQA accuracy results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa

from jlens_reasoning.benchmarks.flenqa.storage import PROVENANCE_TYPE
from jlens_reasoning.experiments_utils.storage import (
    ShardManifest,
    ShardWriter,
    is_shard_complete,
    read_shard_manifest,
    validate_shard_manifest,
)

REQUIRED_TABLES = ("results",)
TABLE_SCHEMAS = {
    "results": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("canonical_index", pa.int32(), nullable=False),
            pa.field("problem_id", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("label", pa.bool_(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("ctx_size", pa.int32(), nullable=False),
            pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("n_input_tokens", pa.int32(), nullable=False),
            pa.field("provenance", PROVENANCE_TYPE, nullable=False),
            pa.field("generated_token_ids", pa.list_(pa.int32()), nullable=False),
            pa.field(
                "generated_token_pieces", pa.list_(pa.string()), nullable=False
            ),
            pa.field("generated_text", pa.string(), nullable=False),
            pa.field("generation_status", pa.string(), nullable=False),
            pa.field("finish_reason", pa.string()),
            pa.field("verdict", pa.bool_()),
            pa.field("correct", pa.bool_(), nullable=False),
        ]
    )
}


def record_batch(columns: Mapping[str, Sequence[Any]]) -> pa.RecordBatch:
    """Build one schema-exact accuracy result batch."""
    if not isinstance(columns, Mapping):
        raise TypeError("result batches require column-oriented input")
    schema = TABLE_SCHEMAS["results"]
    if set(columns) != set(schema.names):
        raise ValueError(f"results columns must be exactly {schema.names}")
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    if len({len(array) for array in arrays}) > 1:
        raise ValueError("results columns must have equal lengths")
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def reset_incomplete_shard(root: Path, *, shard_id: int) -> None:
    """Remove only the known files for one incomplete accuracy shard."""
    if type(shard_id) is not int or shard_id < 0:
        raise ValueError("shard_id must be a non-negative integer")
    root = Path(root)
    stem = f"shard-{shard_id:05d}"
    manifest = root / "manifests" / f"{stem}.json"
    for path in (manifest, manifest.with_suffix(".json.tmp")):
        if path.exists():
            path.unlink()
    final = root / "results" / f"{stem}.parquet"
    for path in (final, final.with_suffix(".parquet.tmp")):
        if path.exists():
            path.unlink()


__all__ = [
    "REQUIRED_TABLES",
    "TABLE_SCHEMAS",
    "ShardManifest",
    "ShardWriter",
    "is_shard_complete",
    "read_shard_manifest",
    "record_batch",
    "reset_incomplete_shard",
    "validate_shard_manifest",
]
