"""Arrow schemas and record-batch builders for FLenQA shard tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

PROVENANCE_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("source_row_id", pa.int32(), nullable=False),
            pa.field("ctx_size", pa.int32(), nullable=False),
            pa.field("padding_type", pa.string(), nullable=False),
            pa.field("dispersion", pa.string(), nullable=False),
        ]
    )
)

REQUIRED_TABLES = ("prompts", "positions", "topk")

TABLE_SCHEMAS = {
    "prompts": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("canonical_index", pa.int32(), nullable=False),
            pa.field("problem_id", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("label", pa.bool_(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
            pa.field("input_ids", pa.list_(pa.int32()), nullable=False),
            pa.field("max_abs_logit_diff", pa.float32(), nullable=False),
            pa.field("provenance", PROVENANCE_TYPE, nullable=False),
        ]
    ),
    "positions": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
        ]
    ),
    "topk": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("lens_kind", pa.string(), nullable=False),
            pa.field("layer", pa.int16(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("rank", pa.int16(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
            pa.field("logit", pa.float32(), nullable=False),
        ]
    ),
}


def _schema(table: str) -> pa.Schema:
    try:
        return TABLE_SCHEMAS[table]
    except KeyError as exc:
        raise ValueError(f"Unknown FLenQA table: {table!r}") from exc


def record_batch(
    table: str,
    columns: Mapping[str, Sequence[Any]],
) -> pa.RecordBatch:
    """Build one schema-exact batch from column-oriented values."""
    if not isinstance(columns, Mapping):
        raise TypeError("Record batches require column-oriented input")
    schema = _schema(table)
    if set(columns) != set(schema.names):
        raise ValueError(
            f"{table} columns must be exactly {schema.names}; "
            f"received {sorted(columns)}"
        )
    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    lengths = {len(array) for array in arrays}
    if len(lengths) > 1:
        raise ValueError(f"{table} columns must have equal lengths")
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def empty_batch(table: str) -> pa.RecordBatch:
    """Return a zero-row batch retaining the declared table schema."""
    schema = _schema(table)
    return pa.RecordBatch.from_arrays(
        [pa.array([], type=field.type) for field in schema],
        schema=schema,
    )


__all__ = [
    "PROVENANCE_TYPE",
    "REQUIRED_TABLES",
    "TABLE_SCHEMAS",
    "empty_batch",
    "record_batch",
]
