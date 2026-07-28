from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

import jlens_reasoning.benchmarks.flenqa as flenqa_module
from jlens_reasoning.benchmarks.flenqa import (
    FlenqaPrompt,
    FlenqaRow,
    deduplicate,
    normalize_rows,
    verify_schema,
)
from jlens_reasoning.benchmarks.flenqa_prompts import compute_prompt_id


def _raw_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "global_sample_id": 0,
        "sample_id": 0,
        "dataset": "PIR",
        "label": "True",
        "facts": ["The key is in the study."],
        "statement": None,
        "rule": None,
        "assertion/question": "Is the key in the study?",
        "mixin": "The key is in the study.",
        "ctx_size": 250,
        "padding_type": "books",
        "dispersion": "first",
    }
    row.update(overrides)
    return row


def _row(**overrides: object) -> FlenqaRow:
    values: dict[str, object] = {
        "source_row_id": 0,
        "problem_id": 0,
        "sample_id": 0,
        "task": "PIR",
        "label": True,
        "key_texts": ("The key is in the study.",),
        "rule": None,
        "question": "Is the key in the study?",
        "mixin": "The key is in the study.",
        "ctx_size_declared": 250,
        "padding_type_declared": "books",
        "dispersion_declared": "first",
    }
    values.update(overrides)
    return FlenqaRow(**values)


@pytest.mark.parametrize(
    ("column", "invalid"),
    [
        ("task", "Other"),
        ("ctx_size", 251),
        ("padding_type", "different"),
        ("dispersion", "scattered"),
        ("label", 1),
        ("label", []),
    ],
)
def test_verify_schema_rejects_unknown_categorical_values(
    column: str, invalid: object
) -> None:
    source_column = "dataset" if column == "task" else column
    with pytest.raises(ValueError, match=column):
        verify_schema([_raw_row(**{source_column: invalid})])


def test_verify_schema_requires_every_source_column() -> None:
    row = _raw_row()
    del row["statement"]

    with pytest.raises(ValueError, match="statement"):
        verify_schema([row])


def test_verify_schema_accepts_small_valid_fixtures_by_default() -> None:
    verify_schema(
        [
            _raw_row(),
            _raw_row(
                global_sample_id=200,
                sample_id=0,
                dataset="Simplified RuleTaker",
                facts=None,
                statement=["The cow is young."],
                rule=["If someone is young then they are blue."],
                **{"assertion/question": "The cow is blue."},
                mixin="The cow is young.",
                ctx_size=3000,
                padding_type="same",
                dispersion="random",
                label="False",
            ),
        ]
    )


def test_verify_schema_can_require_the_full_published_row_count() -> None:
    with pytest.raises(ValueError, match="12,000"):
        verify_schema([_raw_row()], full=True)


def _compact_count_rows() -> list[dict[str, object]]:
    return [
        _raw_row(
            global_sample_id=0,
            dataset="PIR",
            ctx_size=250,
            padding_type="books",
            dispersion="first",
            label="True",
        ),
        _raw_row(
            global_sample_id=0,
            dataset="PIR",
            ctx_size=500,
            padding_type="same",
            dispersion="middle",
            label="False",
        ),
        _raw_row(
            global_sample_id=1,
            dataset="MonoRel",
            ctx_size=250,
            padding_type="same",
            dispersion="first",
            label="False",
        ),
        _raw_row(
            global_sample_id=1,
            dataset="MonoRel",
            ctx_size=500,
            padding_type="books",
            dispersion="middle",
            label="True",
        ),
    ]


COMPACT_MARGINAL_COUNTS = {
    "task": {"PIR": 2, "MonoRel": 2},
    "ctx_size": {250: 2, 500: 2},
    "padding_type": {"books": 2, "same": 2},
    "dispersion": {"first": 2, "middle": 2},
    "label": {True: 2, False: 2},
}


@pytest.mark.parametrize(
    ("logical_name", "source_column", "replacement"),
    [
        ("task", "dataset", "MonoRel"),
        ("problem_id", "global_sample_id", 2),
        ("ctx_size", "ctx_size", 500),
        ("padding_type", "padding_type", "same"),
        ("dispersion", "dispersion", "middle"),
        ("label", "label", "False"),
    ],
)
def test_count_invariants_reject_each_marginal_mismatch(
    logical_name: str,
    source_column: str,
    replacement: object,
) -> None:
    rows = _compact_count_rows()
    flenqa_module.verify_count_invariants(
        rows,
        expected_row_count=4,
        expected_marginals=COMPACT_MARGINAL_COUNTS,
        expected_problem_count=2,
        expected_rows_per_problem=2,
    )
    rows[0][source_column] = replacement

    with pytest.raises(ValueError, match=logical_name):
        flenqa_module.verify_count_invariants(
            rows,
            expected_row_count=4,
            expected_marginals=COMPACT_MARGINAL_COUNTS,
            expected_problem_count=2,
            expected_rows_per_problem=2,
        )


def test_normalize_rows_assigns_source_ids_and_task_specific_key_texts() -> None:
    raw_rows = [
        _raw_row(),
        _raw_row(
            global_sample_id=100,
            sample_id=0,
            dataset="MonoRel",
            facts=["Ada is older than Cy.", "Cy is older than Bea."],
            mixin="Ada is older than Cy.\nCy is older than Bea.",
            **{"assertion/question": "Is Ada older than Bea?"},
        ),
        _raw_row(
            global_sample_id=200,
            sample_id=0,
            dataset="Simplified RuleTaker",
            facts=None,
            statement=["The cow is young.", "The cow is kind."],
            rule=["If someone is young then they are blue."],
            mixin="The cow is young.\nThe cow is kind.",
            label="False",
            **{"assertion/question": "The cow is blue."},
        ),
    ]

    rows = normalize_rows(raw_rows)

    assert [row.source_row_id for row in rows] == [0, 1, 2]
    assert rows[0].problem_id == 0
    assert rows[0].sample_id == 0
    assert rows[0].label is True
    assert rows[0].key_texts == ("The key is in the study.",)
    assert rows[1].key_texts == (
        "Ada is older than Cy.",
        "Cy is older than Bea.",
    )
    assert rows[2].key_texts == ("The cow is young.", "The cow is kind.")
    assert rows[2].rule == ["If someone is young then they are blue."]
    assert rows[2].label is False


def test_row_and_prompt_models_are_frozen_and_slotted() -> None:
    row = _row()
    prompt = deduplicate([row])[0]

    with pytest.raises(FrozenInstanceError):
        row.label = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        prompt.canonical_index = 99  # type: ignore[misc]
    assert not hasattr(row, "__dict__")
    assert not hasattr(prompt, "__dict__")


def test_deduplicate_collapses_identical_prompts_and_aggregates_provenance() -> None:
    rows = [
        _row(source_row_id=10, padding_type_declared="books"),
        _row(
            source_row_id=11,
            padding_type_declared="same",
            dispersion_declared="middle",
        ),
        _row(
            source_row_id=12,
            padding_type_declared="books",
            dispersion_declared="middle",
        ),
    ]

    prompts = deduplicate(rows)

    assert len(prompts) == 1
    assert isinstance(prompts[0], FlenqaPrompt)
    assert prompts[0].canonical_index == 0
    assert prompts[0].source_row_ids == (10, 11, 12)
    assert prompts[0].padding_type_declared == ("books", "same")
    assert prompts[0].dispersion_declared == ("first", "middle")
    assert prompts[0].prompt_id == compute_prompt_id(prompts[0].text)


def test_deduplicate_splits_distinct_final_prompt_text() -> None:
    prompts = deduplicate(
        [
            _row(source_row_id=4),
            _row(
                source_row_id=5,
                mixin="The key is in the kitchen.",
                key_texts=("The key is in the kitchen.",),
            ),
        ]
    )

    assert len(prompts) == 2
    assert prompts[0].prompt_id != prompts[1].prompt_id


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("problem_id", 999),
        ("label", False),
        ("ctx_size_declared", 500),
    ],
)
def test_deduplicate_rejects_identical_text_with_mixed_invariants(
    field: str, different: object
) -> None:
    original = _row(source_row_id=7)
    changed = _row(source_row_id=8, **{field: different})

    with pytest.raises(ValueError, match=field):
        deduplicate([original, changed])


def test_deduplicate_rejects_identical_text_with_mixed_tasks() -> None:
    monorel_question = "True/False Question: Q"
    monorel_mixin = "M"
    pir_mixin = (
        "Here are some facts. Answer the exact following question based on the "
        f"text: {monorel_question} Answer the question as it appears exactly.\n"
        f"{monorel_mixin}"
    )
    pir = _row(
        source_row_id=7,
        question="Q",
        mixin=pir_mixin,
        key_texts=(pir_mixin,),
    )
    monorel = _row(
        source_row_id=8,
        task="MonoRel",
        question=monorel_question,
        mixin=monorel_mixin,
        key_texts=(monorel_mixin,),
    )

    with pytest.raises(ValueError, match="task"):
        deduplicate([pir, monorel])


def test_deduplicate_preserves_first_occurrence_order() -> None:
    first = _row(
        source_row_id=20,
        problem_id=2,
        mixin="B fact.",
        key_texts=("B fact.",),
    )
    second = _row(
        source_row_id=21,
        problem_id=1,
        mixin="A fact.",
        key_texts=("A fact.",),
    )
    duplicate_first = _row(
        source_row_id=22,
        problem_id=2,
        mixin="B fact.",
        key_texts=("B fact.",),
        padding_type_declared="same",
    )

    prompts = deduplicate([first, second, duplicate_first])

    assert [prompt.canonical_index for prompt in prompts] == [0, 1]
    assert [prompt.problem_id for prompt in prompts] == [2, 1]
    assert prompts[0].source_row_ids == (20, 22)
