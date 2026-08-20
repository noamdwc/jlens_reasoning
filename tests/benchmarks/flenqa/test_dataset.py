from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import get_type_hints

import pytest

import jlens_reasoning.benchmarks.flenqa.dataset as flenqa_module
from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    FlenqaRow,
    SourceProvenance,
    build_prompt_text,
    compute_prompt_id,
    normalize_rows,
    prepare_prompts,
    verify_schema,
)


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


@pytest.mark.parametrize(
    ("column", "invalid", "error_name"),
    [
        ("global_sample_id", 0.0, "global_sample_id"),
        ("global_sample_id", True, "global_sample_id"),
        ("sample_id", 0.0, "sample_id"),
        ("sample_id", False, "sample_id"),
        ("ctx_size", 250.0, "ctx_size"),
        ("ctx_size", True, "ctx_size"),
        ("dataset", [], "task"),
        ("padding_type", [], "padding_type"),
        ("dispersion", [], "dispersion"),
        ("assertion/question", b"Is the key in the study?", "assertion/question"),
        ("mixin", ["The key is in the study."], "mixin"),
    ],
)
def test_verify_schema_rejects_wrong_scalar_types(
    column: str,
    invalid: object,
    error_name: str,
) -> None:
    with pytest.raises(ValueError, match=error_name):
        verify_schema([_raw_row(**{column: invalid})])


@pytest.mark.parametrize(
    ("task", "column", "invalid"),
    [
        ("PIR", "facts", "The key is in the study."),
        ("MonoRel", "facts", [1]),
        ("Simplified RuleTaker", "statement", "The cow is young."),
        ("Simplified RuleTaker", "statement", [1]),
    ],
)
def test_verify_schema_rejects_invalid_task_key_collections(
    task: str,
    column: str,
    invalid: object,
) -> None:
    overrides: dict[str, object] = {"dataset": task, column: invalid}
    if task == "Simplified RuleTaker":
        overrides["facts"] = None
        overrides["rule"] = "If someone is young then they are blue."

    with pytest.raises(ValueError, match=column):
        verify_schema([_raw_row(**overrides)])


@pytest.mark.parametrize(
    "invalid_rule",
    [
        None,
        "",
        " ",
        [],
        (),
        [""],
        [" "],
        ["If someone is young then they are blue.", ""],
        [1],
        7,
        {"rule": "If someone is young then they are blue."},
    ],
)
def test_verify_schema_rejects_invalid_ruletaker_rules(
    invalid_rule: object,
) -> None:
    with pytest.raises(ValueError, match="rule"):
        verify_schema(
            [
                _raw_row(
                    dataset="Simplified RuleTaker",
                    facts=None,
                    statement=["The cow is young."],
                    rule=invalid_rule,
                )
            ]
        )


@pytest.mark.parametrize(
    ("raw_rule", "expected_snapshot"),
    [
        (
            "If someone is young then they are blue.",
            "If someone is young then they are blue.",
        ),
        (
            ["If someone is young then they are blue."],
            "['If someone is young then they are blue.']",
        ),
        (
            ("If someone is young then they are blue.",),
            "('If someone is young then they are blue.',)",
        ),
    ],
)
def test_normalize_rows_accepts_and_snapshots_supported_ruletaker_rules(
    raw_rule: object,
    expected_snapshot: str,
) -> None:
    row = normalize_rows(
        [
            _raw_row(
                dataset="Simplified RuleTaker",
                facts=None,
                statement=["The cow is young."],
                rule=raw_rule,
            )
        ]
    )[0]

    assert row.rule == expected_snapshot


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


def test_normalize_rows_can_require_the_full_published_row_count() -> None:
    with pytest.raises(ValueError, match="12,000"):
        normalize_rows([_raw_row()], full=True)


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
    assert rows[2].rule == "['If someone is young then they are blue.']"
    assert rows[2].label is False


def test_normalize_rows_detaches_ruletaker_rule_from_mutable_source() -> None:
    raw_rule = ["If someone is young then they are blue."]
    normalized = normalize_rows(
        [
            _raw_row(
                global_sample_id=200,
                dataset="Simplified RuleTaker",
                facts=None,
                statement=["The cow is young."],
                rule=raw_rule,
                mixin="The cow is young.",
                **{"assertion/question": "The cow is blue."},
            )
        ]
    )[0]
    expected_rule = "['If someone is young then they are blue.']"
    expected_prompt = (
        "Answer whether the statement The cow is blue. can be derived from the "
        'rule and the facts. Answer with either "True" or "False".\n'
        "Rule: ['If someone is young then they are blue.']\n"
        "Facts: The cow is young.\n"
        'Answer with either "True or "False".\n'
    )

    raw_rule.append("If someone is kind then they are green.")
    prompt = prepare_prompts([normalized])[0]

    assert normalized.rule == expected_rule
    assert prompt.rule == expected_rule
    assert prompt.text == expected_prompt


def test_rule_fields_are_annotated_as_immutable_text() -> None:
    assert get_type_hints(FlenqaRow)["rule"] == str | None
    assert get_type_hints(FlenqaPrompt)["rule"] == str | None


def test_row_and_prompt_models_are_frozen_and_slotted() -> None:
    row = _row()
    prompt = prepare_prompts([row])[0]

    with pytest.raises(FrozenInstanceError):
        row.label = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        prompt.canonical_index = 99  # type: ignore[misc]
    assert not hasattr(row, "__dict__")
    assert not hasattr(prompt, "__dict__")


def test_prepare_prompts_deduplicates_and_aggregates_provenance() -> None:
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

    prompts = prepare_prompts(rows)

    assert len(prompts) == 1
    assert isinstance(prompts[0], FlenqaPrompt)
    assert prompts[0].canonical_index == 0
    assert prompts[0].provenance == (
        SourceProvenance(10, 250, "books", "first"),
        SourceProvenance(11, 250, "same", "middle"),
        SourceProvenance(12, 250, "books", "middle"),
    )
    assert prompts[0].prompt_id == compute_prompt_id(prompts[0].text)


def test_prepare_prompts_keeps_distinct_final_prompt_text() -> None:
    first = _row(source_row_id=4)
    second = _row(
        source_row_id=5,
        mixin="The key is in the kitchen.",
        key_texts=("The key is in the kitchen.",),
    )

    prompts = prepare_prompts([first, second])

    assert len(prompts) == 2


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("problem_id", 999),
        ("label", False),
        ("ctx_size_declared", 500),
    ],
)
def test_prepare_prompts_rejects_identical_text_with_mixed_invariants(
    field: str, different: object
) -> None:
    original = _row(source_row_id=7)
    changed = _row(source_row_id=8, **{field: different})

    with pytest.raises(ValueError, match=field):
        prepare_prompts([original, changed])


def test_prepare_prompts_rejects_identical_text_with_mixed_tasks() -> None:
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
        prepare_prompts([pir, monorel])


def test_prepare_prompts_preserves_first_occurrence_order() -> None:
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

    prompts = prepare_prompts([first, second, duplicate_first])

    assert [prompt.canonical_index for prompt in prompts] == [0, 1]
    assert [prompt.problem_id for prompt in prompts] == [2, 1]
    assert prompts[0].provenance == (
        SourceProvenance(20, 250, "books", "first"),
        SourceProvenance(22, 250, "same", "first"),
    )


def test_prepare_prompts_sorts_complete_source_provenance_records() -> None:
    prompts = prepare_prompts(
        [
            _row(
                source_row_id=7,
                ctx_size_declared=500,
                padding_type_declared="books",
                dispersion_declared="first",
            ),
            _row(
                source_row_id=3,
                ctx_size_declared=500,
                padding_type_declared="same",
                dispersion_declared="last",
            ),
        ]
    )

    assert prompts[0].provenance == (
        SourceProvenance(3, 500, "same", "last"),
        SourceProvenance(7, 500, "books", "first"),
    )


def test_pir_prompt_matches_authors_template_byte_for_byte() -> None:
    prompt = build_prompt_text(
        task="PIR",
        question="Is the key in the study?",
        mixin="The key is in the study.\nThe lamp is in the hall.",
        rule=None,
    )

    assert prompt == (
        "The key is in the study.\nThe lamp is in the hall.\n"
        "True/False Question: Is the key in the study?\n"
        "Answer only True or False.\n"
    )


def test_legacy_flenqa_experiment_package_is_absent() -> None:
    assert not Path("experiments/flenqa_length_drift/__init__.py").exists()


def test_monorel_prompt_matches_authors_template_byte_for_byte() -> None:
    prompt = build_prompt_text(
        task="MonoRel",
        question="Is Ada older than Bea?",
        mixin="Ada is older than Cy.\nCy is older than Bea.",
        rule=None,
    )

    assert prompt == (
        "Here are some facts. Answer the exact following question based on the "
        "text: Is Ada older than Bea? Answer the question as it appears exactly.\n"
        "Ada is older than Cy.\nCy is older than Bea.\n"
        "Is Ada older than Bea?\n"
        "Answer only True or False.\n"
    )


def test_ruletaker_prompt_preserves_raw_rule_typo_and_trailing_newline() -> None:
    prompt = build_prompt_text(
        task="Simplified RuleTaker",
        question="The cow is blue.",
        mixin="The cow is young.\nThe cow is kind.",
        rule=["If someone is young then they are blue."],
    )

    assert prompt == (
        "Answer whether the statement The cow is blue. can be derived from the "
        'rule and the facts. Answer with either "True" or "False".\n'
        "Rule: ['If someone is young then they are blue.']\n"
        "Facts: The cow is young.\nThe cow is kind.\n"
        'Answer with either "True or "False".\n'
    )
