from __future__ import annotations

import pyarrow as pa
import pytest

from jlens_reasoning.benchmarks.flenqa.storage import (
    PROVENANCE_TYPE,
    REQUIRED_TABLES,
    TABLE_SCHEMAS,
    empty_batch,
    record_batch,
)


def test_only_three_shard_tables_are_required() -> None:
    assert REQUIRED_TABLES == ("prompts", "positions", "topk")
    assert set(TABLE_SCHEMAS) == set(REQUIRED_TABLES)


def test_prompt_provenance_is_a_typed_list_of_structs() -> None:
    provenance = TABLE_SCHEMAS["prompts"].field("provenance")

    assert provenance.type == PROVENANCE_TYPE
    assert pa.types.is_list(provenance.type)
    assert pa.types.is_struct(provenance.type.value_type)
    assert provenance.type.value_type.names == [
        "source_row_id",
        "ctx_size",
        "padding_type",
        "dispersion",
    ]
    assert provenance.type.value_type.field("source_row_id").type == pa.int32()
    assert provenance.type.value_type.field("ctx_size").type == pa.int32()
    assert provenance.type.value_type.field("padding_type").type == pa.string()
    assert provenance.type.value_type.field("dispersion").type == pa.string()


def test_record_batch_accepts_typed_provenance_records() -> None:
    batch = record_batch(
        "prompts",
        {
            "prompt_id": ["p"],
            "canonical_index": [0],
            "problem_id": [1],
            "task": ["PIR"],
            "label": [True],
            "text": ["prompt"],
            "input_ids": [[1, 2]],
            "max_abs_logit_diff": [1e-7],
            "provenance": [
                [
                    {
                        "source_row_id": 3,
                        "ctx_size": 500,
                        "padding_type": "same",
                        "dispersion": "last",
                    }
                ]
            ],
        },
    )

    assert batch.schema == TABLE_SCHEMAS["prompts"]
    assert batch.to_pydict()["provenance"][0][0] == {
        "source_row_id": 3,
        "ctx_size": 500,
        "padding_type": "same",
        "dispersion": "last",
    }


def test_record_batch_requires_columnar_input_and_exact_schema() -> None:
    with pytest.raises(TypeError, match="column"):
        record_batch(
            "positions",
            [{"prompt_id": "p", "position": 9, "label": "final_prompt"}],
        )
    with pytest.raises(ValueError, match="columns"):
        record_batch("positions", {"prompt_id": ["p"]})


def test_empty_batch_retains_exact_schema() -> None:
    batch = empty_batch("topk")

    assert batch.num_rows == 0
    assert batch.schema == TABLE_SCHEMAS["topk"]
