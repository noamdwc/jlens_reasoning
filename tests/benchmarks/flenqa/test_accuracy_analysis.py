from __future__ import annotations

import pyarrow as pa
import pytest

from jlens_reasoning.benchmarks.flenqa.accuracy_analysis import (
    FULL_UNIQUE_PROMPT_COUNTS,
    AccuracyPoint,
    TokenLengthPoint,
    VerdictCountPoint,
    summarize_paper_random,
    summarize_token_lengths,
    summarize_unique_prompts,
    summarize_verdicts,
)
from jlens_reasoning.benchmarks.flenqa.accuracy_storage import record_batch


def _table() -> pa.Table:
    batch = record_batch(
        {
            "prompt_id": ["p1", "p2"],
            "canonical_index": [0, 1],
            "problem_id": [1, 2],
            "task": ["PIR", "MonoRel"],
            "label": [True, False],
            "text": ["first", "second"],
            "ctx_size": [250, 500],
            "input_ids": [[1, 2], [1, 2, 3]],
            "n_input_tokens": [240, 520],
            "provenance": [
                [
                    {
                        "source_row_id": 1,
                        "ctx_size": 250,
                        "padding_type": "books",
                        "dispersion": "random",
                    },
                    {
                        "source_row_id": 2,
                        "ctx_size": 250,
                        "padding_type": "same",
                        "dispersion": "random",
                    },
                ],
                [
                    {
                        "source_row_id": 3,
                        "ctx_size": 500,
                        "padding_type": "books",
                        "dispersion": "random",
                    },
                    {
                        "source_row_id": 4,
                        "ctx_size": 500,
                        "padding_type": "books",
                        "dispersion": "first",
                    },
                ],
            ],
            "generated_token_ids": [[8], [9]],
            "generated_token_pieces": [["True"], ["maybe"]],
            "generated_text": ["True", "maybe"],
            "generation_status": ["complete", "complete"],
            "finish_reason": ["eos", "eos"],
            "verdict": [True, None],
            "correct": [True, False],
        }
    )
    return pa.Table.from_batches([batch])


def test_unique_summary_weights_each_prompt_once() -> None:
    assert summarize_unique_prompts(_table()) == (
        AccuracyPoint(250, correct=1, total=1, no_verdict=0),
        AccuracyPoint(500, correct=0, total=1, no_verdict=1),
    )


def test_paper_summary_expands_only_random_source_rows() -> None:
    assert summarize_paper_random(_table()) == (
        AccuracyPoint(250, correct=2, total=2, no_verdict=0),
        AccuracyPoint(500, correct=0, total=1, no_verdict=1),
    )


def test_task_filter_applies_before_aggregation() -> None:
    assert summarize_unique_prompts(_table(), task="PIR") == (
        AccuracyPoint(250, correct=1, total=1, no_verdict=0),
    )
    with pytest.raises(ValueError, match="task"):
        summarize_unique_prompts(_table(), task="unknown")


def test_verdict_and_token_length_diagnostics() -> None:
    assert summarize_verdicts(_table()) == (
        VerdictCountPoint(250, true=1, false=0, no_verdict=0),
        VerdictCountPoint(500, true=0, false=0, no_verdict=1),
    )
    assert summarize_token_lengths(_table()) == (
        TokenLengthPoint(250, minimum=240, median=240.0, maximum=240),
        TokenLengthPoint(500, minimum=520, median=520.0, maximum=520),
    )


def test_duplicate_prompt_ids_are_rejected() -> None:
    table = pa.concat_tables([_table(), _table()])

    with pytest.raises(ValueError, match="duplicate prompt"):
        summarize_unique_prompts(table)


def test_mixed_provenance_context_sizes_are_rejected() -> None:
    table = _table().to_pydict()
    table["provenance"][0][0]["ctx_size"] = 500

    with pytest.raises(ValueError, match="context size"):
        summarize_paper_random(pa.table(table, schema=_table().schema))


def test_full_unique_counts_are_pinned() -> None:
    assert FULL_UNIQUE_PROMPT_COUNTS == {
        250: 300,
        500: 2_368,
        1000: 2_394,
        2000: 2_400,
        3000: 2_400,
    }
