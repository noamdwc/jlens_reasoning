from __future__ import annotations

import pyarrow as pa
import pytest

from experiments.flenqa_length_drift.tables import (
    GLOBAL_SCHEMAS,
    REQUIRED_SHARD_TABLES,
    TABLE_SCHEMAS,
    empty_batch,
    record_batch,
)


def test_every_required_table_has_a_typed_schema() -> None:
    assert REQUIRED_SHARD_TABLES == (
        "prompts",
        "source_rows",
        "spans",
        "anchors",
        "topk",
        "bridge",
        "summary",
        "scoring",
    )
    assert set(TABLE_SCHEMAS) == set(REQUIRED_SHARD_TABLES)
    assert set(GLOBAL_SCHEMAS) == {"vocab", "run_meta"}
    assert all(isinstance(schema, pa.Schema) for schema in TABLE_SCHEMAS.values())


def test_prompts_schema_carries_declared_and_effective_conditions() -> None:
    names = set(TABLE_SCHEMAS["prompts"].names)

    assert {
        "ctx_size_declared",
        "padding_type_declared",
        "dispersion_declared",
        "padding_type_effective",
        "dispersion_effective",
        "frac_padding_before",
        "frac_padding_between",
        "frac_padding_after",
        "n_padding_tokens",
        "token_hash",
    } <= names


def test_provenance_anchor_vocab_and_span_diagnostics_are_explicit() -> None:
    assert {
        "source_row_id",
        "sample_id",
        "padding_type_declared",
        "dispersion_declared",
    } <= set(TABLE_SCHEMAS["source_rows"].names)
    assert {"anchor_label", "position"} <= set(TABLE_SCHEMAS["anchors"].names)
    assert {
        "span_kind",
        "ordinal",
        "span_status",
        "span_match_count",
        "char_start",
        "token_start",
    } <= set(TABLE_SCHEMAS["spans"].names)
    assert GLOBAL_SCHEMAS["vocab"].names == ["token_id", "token_text"]


def test_record_batch_requires_columnar_input_and_exact_schema() -> None:
    batch = record_batch(
        "anchors",
        {
            "prompt_id": ["p"],
            "anchor_label": ["final_prompt"],
            "position": [9],
        },
    )

    assert isinstance(batch, pa.RecordBatch)
    assert batch.schema == TABLE_SCHEMAS["anchors"]
    assert batch.num_rows == 1

    with pytest.raises(TypeError, match="column"):
        record_batch(
            "anchors",
            [{"prompt_id": "p", "anchor_label": "final_prompt", "position": 9}],
        )
    with pytest.raises(ValueError, match="columns"):
        record_batch("anchors", {"prompt_id": ["p"]})


def test_empty_batch_retains_exact_schema() -> None:
    batch = empty_batch("bridge")

    assert batch.num_rows == 0
    assert batch.schema == TABLE_SCHEMAS["bridge"]
