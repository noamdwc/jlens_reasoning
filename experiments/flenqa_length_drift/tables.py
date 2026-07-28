"""Typed Arrow schemas for every FLenQA readout artifact table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

REQUIRED_SHARD_TABLES = (
    "prompts",
    "source_rows",
    "spans",
    "anchors",
    "topk",
    "bridge",
    "summary",
    "scoring",
)

TABLE_SCHEMAS = {
    "prompts": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("canonical_index", pa.int32(), nullable=False),
            pa.field("problem_id", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("label", pa.bool_(), nullable=False),
            pa.field("final_text_hash", pa.string(), nullable=False),
            pa.field("token_hash", pa.string(), nullable=False),
            pa.field("n_tokens", pa.int32(), nullable=False),
            pa.field("ctx_size_declared", pa.int32(), nullable=False),
            pa.field(
                "padding_type_declared",
                pa.list_(pa.string()),
                nullable=False,
            ),
            pa.field(
                "dispersion_declared",
                pa.list_(pa.string()),
                nullable=False,
            ),
            pa.field("padding_type_effective", pa.string()),
            pa.field("dispersion_effective", pa.string(), nullable=False),
            pa.field("frac_padding_before", pa.float32(), nullable=False),
            pa.field("frac_padding_between", pa.float32(), nullable=False),
            pa.field("frac_padding_after", pa.float32(), nullable=False),
            pa.field("n_padding_tokens", pa.int32(), nullable=False),
            pa.field("bridge", pa.string()),
        ]
    ),
    "source_rows": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("source_row_id", pa.int32(), nullable=False),
            pa.field("problem_id", pa.int32(), nullable=False),
            pa.field("sample_id", pa.int32(), nullable=False),
            pa.field("task", pa.string(), nullable=False),
            pa.field("label", pa.bool_(), nullable=False),
            pa.field("ctx_size_declared", pa.int32(), nullable=False),
            pa.field("padding_type_declared", pa.string(), nullable=False),
            pa.field("dispersion_declared", pa.string(), nullable=False),
        ]
    ),
    "spans": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("span_kind", pa.string(), nullable=False),
            pa.field("ordinal", pa.int16(), nullable=False),
            pa.field("fact_ordinal", pa.int16()),
            pa.field("surface", pa.string(), nullable=False),
            pa.field("span_status", pa.string(), nullable=False),
            pa.field("span_match_count", pa.int16(), nullable=False),
            pa.field("char_start", pa.int32()),
            pa.field("char_end", pa.int32()),
            pa.field("token_start", pa.int32()),
            pa.field("token_end", pa.int32()),
        ]
    ),
    "anchors": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("anchor_label", pa.string(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
        ]
    ),
    "topk": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("layer", pa.int16(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("anchor_label", pa.string(), nullable=False),
            pa.field("lens_kind", pa.string(), nullable=False),
            pa.field("rank", pa.int16(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
            pa.field("logit", pa.float32(), nullable=False),
        ]
    ),
    "bridge": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("layer", pa.int16(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("anchor_label", pa.string(), nullable=False),
            pa.field("lens_kind", pa.string(), nullable=False),
            pa.field("surface", pa.string(), nullable=False),
            pa.field("token_id", pa.int32(), nullable=False),
            pa.field("rank", pa.int32(), nullable=False),
            pa.field("logit", pa.float32(), nullable=False),
        ]
    ),
    "summary": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("layer", pa.int16(), nullable=False),
            pa.field("position", pa.int32(), nullable=False),
            pa.field("provenance", pa.string(), nullable=False),
            pa.field("lens_kind", pa.string(), nullable=False),
            pa.field("entropy", pa.float32(), nullable=False),
            pa.field("max_logit", pa.float32(), nullable=False),
            pa.field("top1_token_id", pa.int32(), nullable=False),
        ]
    ),
    "scoring": pa.schema(
        [
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("logit_true", pa.float32(), nullable=False),
            pa.field("logit_false", pa.float32(), nullable=False),
            pa.field("rank_true", pa.int32(), nullable=False),
            pa.field("rank_false", pa.int32(), nullable=False),
            pa.field("predicted", pa.bool_(), nullable=False),
            pa.field("correct", pa.bool_(), nullable=False),
            pa.field("generated_text", pa.string()),
            pa.field("extracted", pa.bool_()),
            pa.field("generated_correct", pa.bool_()),
            pa.field("agrees", pa.bool_()),
        ]
    ),
}

GLOBAL_SCHEMAS = {
    "vocab": pa.schema(
        [
            pa.field("token_id", pa.int32(), nullable=False),
            pa.field("token_text", pa.string(), nullable=False),
        ]
    ),
    "run_meta": pa.schema(
        [
            pa.field("config_hash", pa.string(), nullable=False),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("lens_revision", pa.string(), nullable=False),
            pa.field("tokenizer_name", pa.string(), nullable=False),
            pa.field("template_hash", pa.string(), nullable=False),
            pa.field("top_k", pa.int16(), nullable=False),
            pa.field("anchor_budget", pa.int16(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("code_revision", pa.string(), nullable=False),
        ]
    ),
}


def _schema(table: str) -> pa.Schema:
    try:
        return TABLE_SCHEMAS[table]
    except KeyError:
        try:
            return GLOBAL_SCHEMAS[table]
        except KeyError as exc:
            raise ValueError(f"Unknown FLenQA table: {table!r}") from exc


def record_batch(
    table: str,
    columns: Mapping[str, Sequence[Any]],
) -> pa.RecordBatch:
    """Build one schema-exact batch from columns, never row dictionaries."""
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
