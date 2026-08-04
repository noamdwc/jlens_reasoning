from __future__ import annotations

from pathlib import Path

import pyarrow as pa

from jlens_reasoning.benchmarks.flenqa.accuracy_storage import (
    REQUIRED_TABLES,
    TABLE_SCHEMAS,
    record_batch,
    reset_incomplete_shard,
)


def _columns() -> dict[str, list[object]]:
    return {
        "prompt_id": ["p"],
        "canonical_index": [0],
        "problem_id": [3],
        "task": ["PIR"],
        "label": [True],
        "text": ["prompt"],
        "ctx_size": [250],
        "input_ids": [[1, 2]],
        "n_input_tokens": [2],
        "provenance": [
            [
                {
                    "source_row_id": 7,
                    "ctx_size": 250,
                    "padding_type": "books",
                    "dispersion": "random",
                }
            ]
        ],
        "generated_token_ids": [[9]],
        "generated_token_pieces": [["maybe"]],
        "generated_text": ["maybe"],
        "generation_status": ["complete"],
        "finish_reason": ["eos"],
        "verdict": [None],
        "correct": [False],
    }


def test_accuracy_result_schema_is_exact_and_typed() -> None:
    schema = TABLE_SCHEMAS["results"]

    assert REQUIRED_TABLES == ("results",)
    assert schema.field("prompt_id").type == pa.string()
    assert schema.field("ctx_size").type == pa.int32()
    assert schema.field("n_input_tokens").type == pa.int32()
    assert schema.field("verdict").nullable
    assert (
        schema.field("provenance").type.value_type.field("dispersion").type
        == pa.string()
    )


def test_accuracy_record_batch_round_trips_nullable_verdict() -> None:
    batch = record_batch(_columns())

    assert batch.schema == TABLE_SCHEMAS["results"]
    assert batch.to_pydict()["verdict"] == [None]


def test_accuracy_record_batch_requires_exact_columns() -> None:
    columns = _columns()
    del columns["correct"]

    try:
        record_batch(columns)
    except ValueError as exc:
        assert "columns must be exactly" in str(exc)
    else:
        raise AssertionError("missing result columns must be rejected")


def test_reset_removes_only_one_accuracy_shard(tmp_path: Path) -> None:
    target = tmp_path / "results" / "shard-00002.parquet"
    other = tmp_path / "results" / "shard-00003.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"target")
    other.write_bytes(b"other")

    reset_incomplete_shard(tmp_path, shard_id=2)

    assert not target.exists()
    assert other.read_bytes() == b"other"
