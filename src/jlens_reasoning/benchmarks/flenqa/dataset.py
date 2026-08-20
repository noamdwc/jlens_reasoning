"""Typed normalization and ordered prompt deduplication for FLenQA."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

TASKS = frozenset({"PIR", "MonoRel", "Simplified RuleTaker"})
CONTEXT_SIZES = frozenset({250, 500, 1000, 2000, 3000})
PADDING_TYPES = frozenset({"books", "same"})
DISPERSIONS = frozenset({"first", "middle", "last", "random"})
FULL_DATASET_ROW_COUNT = 12_000

REQUIRED_COLUMNS = frozenset(
    {
        "sample_id",
        "label",
        "facts",
        "padding_type",
        "dispersion",
        "ctx_size",
        "mixin",
        "dataset",
        "global_sample_id",
        "assertion/question",
        "rule",
        "statement",
    }
)


def build_prompt_text(
    *,
    task: str,
    question: str,
    mixin: str,
    rule: object,
) -> str:
    """Render the authors' task-specific FLenQA prompt byte for byte."""
    if task == "PIR":
        return f"{mixin}\nTrue/False Question: {question}\nAnswer only True or False.\n"
    if task == "MonoRel":
        return (
            "Here are some facts. Answer the exact following question based on the "
            f"text: {question} Answer the question as it appears exactly.\n"
            f"{mixin}\n"
            f"{question}\n"
            "Answer only True or False.\n"
        )
    if task == "Simplified RuleTaker":
        return (
            f"Answer whether the statement {question} can be derived from the "
            'rule and the facts. Answer with either "True" or "False".\n'
            f"Rule: {rule}\n"
            f"Facts: {mixin}\n"
            'Answer with either "True or "False".\n'
        )
    raise ValueError(f"Unknown FLenQA task: {task!r}")


def compute_prompt_id(text: str) -> str:
    """Return the full SHA-256 digest of the exact final prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FlenqaRow:
    """One normalized source row with its original declared conditions."""

    source_row_id: int
    problem_id: int
    sample_id: int
    task: str
    label: bool
    key_texts: tuple[str, ...]
    rule: str | None
    question: str
    mixin: str
    ctx_size_declared: int
    padding_type_declared: str
    dispersion_declared: str


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Declared conditions retained together for one source row."""

    source_row_id: int
    ctx_size: int
    padding_type: str
    dispersion: str


@dataclass(frozen=True, slots=True)
class FlenqaPrompt:
    """One unique final prompt plus ordered source-row provenance."""

    canonical_index: int
    prompt_id: str
    problem_id: int
    task: str
    text: str
    question: str
    key_texts: tuple[str, ...]
    rule: str | None
    label: bool
    mixin: str
    provenance: tuple[SourceProvenance, ...]


def _valid_label(value: object) -> bool:
    return type(value) is bool or (
        isinstance(value, str) and value in {"True", "False"}
    )


def _validate_ruletaker_rule(value: object, *, source_row_id: int) -> None:
    valid = False
    if isinstance(value, str):
        valid = bool(value.strip())
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        valid = len(value) > 0 and all(
            isinstance(rule, str) and bool(rule.strip()) for rule in value
        )
    if not valid:
        raise ValueError(
            "rule in Simplified RuleTaker source row "
            f"{source_row_id} must be a non-empty string or a non-empty "
            "sequence of non-empty strings"
        )


def verify_count_invariants(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_row_count: int,
    expected_marginals: Mapping[str, Mapping[object, int]],
    expected_problem_count: int,
    expected_rows_per_problem: int,
) -> None:
    """Validate dataset count invariants against caller-supplied expectations."""
    if len(rows) != expected_row_count:
        raise ValueError(
            f"FLenQA dataset must contain exactly {expected_row_count:,} rows; "
            f"received {len(rows):,}"
        )

    for logical_name, expected in expected_marginals.items():
        source_name = "dataset" if logical_name == "task" else logical_name
        if logical_name == "label":
            actual = Counter(
                value if type(value) is bool else value == "True"
                for value in (row[source_name] for row in rows)
            )
        else:
            actual = Counter(row[source_name] for row in rows)
        if actual != dict(expected):
            raise ValueError(
                f"Full FLenQA {logical_name} counts do not match "
                f"the published dataset: {dict(actual)!r}"
            )

    problem_counts = Counter(row["global_sample_id"] for row in rows)
    if len(problem_counts) != expected_problem_count or set(
        problem_counts.values()
    ) != {expected_rows_per_problem}:
        raise ValueError(
            "Full FLenQA problem_id counts do not match the published dataset: "
            f"expected {expected_problem_count} problems with "
            f"{expected_rows_per_problem} rows each; received "
            f"{len(problem_counts)} problems with row counts "
            f"{sorted(set(problem_counts.values()))!r}"
        )


def _validate_full_counts(rows: Sequence[Mapping[str, Any]]) -> None:
    expected_marginals: dict[str, dict[object, int]] = {
        "task": {task: 4_000 for task in TASKS},
        "ctx_size": {ctx_size: 2_400 for ctx_size in CONTEXT_SIZES},
        "padding_type": {padding_type: 6_000 for padding_type in PADDING_TYPES},
        "dispersion": {dispersion: 3_000 for dispersion in DISPERSIONS},
        "label": {True: 6_000, False: 6_000},
    }
    verify_count_invariants(
        rows,
        expected_row_count=FULL_DATASET_ROW_COUNT,
        expected_marginals=expected_marginals,
        expected_problem_count=300,
        expected_rows_per_problem=40,
    )


def verify_schema(
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    full: bool = False,
) -> None:
    """Validate required source columns and published categorical values."""
    rows = tuple(raw_rows)
    if not rows:
        if full:
            _validate_full_counts(rows)
        raise ValueError("FLenQA rows must not be empty")

    for source_row_id, row in enumerate(rows):
        missing = REQUIRED_COLUMNS.difference(row)
        if missing:
            missing_columns = ", ".join(sorted(missing))
            raise ValueError(
                f"FLenQA source row {source_row_id} is missing required columns: "
                f"{missing_columns}"
            )
        for integer_column in ("global_sample_id", "sample_id"):
            integer_value = row[integer_column]
            if type(integer_value) is not int:
                raise ValueError(
                    f"Invalid {integer_column} in source row "
                    f"{source_row_id}: {integer_value!r}"
                )
        task = row["dataset"]
        if not isinstance(task, str) or task not in TASKS:
            raise ValueError(f"Invalid task in source row {source_row_id}: {task!r}")
        ctx_size = row["ctx_size"]
        if type(ctx_size) is not int or ctx_size not in CONTEXT_SIZES:
            raise ValueError(
                f"Invalid ctx_size in source row {source_row_id}: {ctx_size!r}"
            )
        padding_type = row["padding_type"]
        if not isinstance(padding_type, str) or padding_type not in PADDING_TYPES:
            raise ValueError(
                f"Invalid padding_type in source row {source_row_id}: {padding_type!r}"
            )
        dispersion = row["dispersion"]
        if not isinstance(dispersion, str) or dispersion not in DISPERSIONS:
            raise ValueError(
                f"Invalid dispersion in source row {source_row_id}: {dispersion!r}"
            )
        label = row["label"]
        if not _valid_label(label):
            raise ValueError(f"Invalid label in source row {source_row_id}: {label!r}")
        _text(
            row["assertion/question"],
            column="assertion/question",
            source_row_id=source_row_id,
        )
        _text(row["mixin"], column="mixin", source_row_id=source_row_id)
        key_column = "statement" if task == "Simplified RuleTaker" else "facts"
        _text_tuple(
            row[key_column],
            column=key_column,
            source_row_id=source_row_id,
        )
        if task == "Simplified RuleTaker":
            _validate_ruletaker_rule(
                row["rule"],
                source_row_id=source_row_id,
            )

    if full:
        _validate_full_counts(rows)


def _text_tuple(value: object, *, column: str, source_row_id: int) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(
            f"{column} in source row {source_row_id} must be a sequence of strings"
        )
    texts = tuple(value)
    if not all(isinstance(text, str) for text in texts):
        raise ValueError(
            f"{column} in source row {source_row_id} must contain only strings"
        )
    return texts


def _text(value: object, *, column: str, source_row_id: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{column} in source row {source_row_id} must be a string")
    return value


def normalize_rows(
    raw_rows: Iterable[Mapping[str, Any]],
    *,
    full: bool = False,
) -> tuple[FlenqaRow, ...]:
    """Normalize published FLenQA rows and assign sequential provenance IDs."""
    rows = tuple(raw_rows)
    verify_schema(rows, full=full)
    normalized: list[FlenqaRow] = []
    for source_row_id, raw in enumerate(rows):
        task = raw["dataset"]
        key_column = "statement" if task == "Simplified RuleTaker" else "facts"
        key_texts = _text_tuple(
            raw[key_column],
            column=key_column,
            source_row_id=source_row_id,
        )
        label = raw["label"]
        normalized.append(
            FlenqaRow(
                source_row_id=source_row_id,
                problem_id=raw["global_sample_id"],
                sample_id=raw["sample_id"],
                task=task,
                label=label if type(label) is bool else label == "True",
                key_texts=key_texts,
                rule=str(raw["rule"]) if task == "Simplified RuleTaker" else None,
                question=_text(
                    raw["assertion/question"],
                    column="assertion/question",
                    source_row_id=source_row_id,
                ),
                mixin=_text(
                    raw["mixin"],
                    column="mixin",
                    source_row_id=source_row_id,
                ),
                ctx_size_declared=raw["ctx_size"],
                padding_type_declared=raw["padding_type"],
                dispersion_declared=raw["dispersion"],
            )
        )
    return tuple(normalized)


def _validate_duplicate_rows(rows: Sequence[FlenqaRow]) -> None:
    first = rows[0]
    for row in rows[1:]:
        for field in ("problem_id", "label", "ctx_size_declared", "task"):
            expected = getattr(first, field)
            actual = getattr(row, field)
            if actual != expected:
                raise ValueError(
                    "Identical FLenQA prompt text mixes invariant "
                    f"{field}: {expected!r} != {actual!r}"
                )


def prepare_prompts(rows: Iterable[FlenqaRow]) -> tuple[FlenqaPrompt, ...]:
    """Create unique prompts in first-occurrence order."""
    rows_by_text: dict[str, list[FlenqaRow]] = {}
    for row in rows:
        text = build_prompt_text(
            task=row.task,
            question=row.question,
            mixin=row.mixin,
            rule=row.rule,
        )
        rows_by_text.setdefault(text, []).append(row)

    prompts = []
    for canonical_index, (text, source_rows) in enumerate(rows_by_text.items()):
        _validate_duplicate_rows(source_rows)
        first = source_rows[0]
        provenance = tuple(
            SourceProvenance(
                source_row_id=row.source_row_id,
                ctx_size=row.ctx_size_declared,
                padding_type=row.padding_type_declared,
                dispersion=row.dispersion_declared,
            )
            for row in sorted(source_rows, key=lambda row: row.source_row_id)
        )
        prompts.append(
            FlenqaPrompt(
                canonical_index=canonical_index,
                prompt_id=compute_prompt_id(text),
                problem_id=first.problem_id,
                task=first.task,
                text=text,
                question=first.question,
                key_texts=first.key_texts,
                rule=first.rule,
                label=first.label,
                mixin=first.mixin,
                provenance=provenance,
            )
        )
    return tuple(prompts)
