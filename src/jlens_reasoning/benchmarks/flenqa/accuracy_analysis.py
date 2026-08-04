"""Pure summaries for FLenQA accuracy artifacts."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

import pyarrow as pa

from jlens_reasoning.benchmarks.flenqa.accuracy_storage import TABLE_SCHEMAS
from jlens_reasoning.benchmarks.flenqa.dataset import TASKS

FULL_UNIQUE_PROMPT_COUNTS = {
    250: 300,
    500: 2_368,
    1000: 2_394,
    2000: 2_400,
    3000: 2_400,
}


@dataclass(frozen=True, slots=True)
class AccuracyPoint:
    ctx_size: int
    correct: int
    total: int
    no_verdict: int

    def __post_init__(self) -> None:
        if self.total <= 0:
            raise ValueError("accuracy total must be positive")
        if not 0 <= self.correct <= self.total:
            raise ValueError("accuracy correct count must be within total")
        if not 0 <= self.no_verdict <= self.total:
            raise ValueError("no-verdict count must be within total")

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


@dataclass(frozen=True, slots=True)
class VerdictCountPoint:
    ctx_size: int
    true: int
    false: int
    no_verdict: int

    @property
    def total(self) -> int:
        return self.true + self.false + self.no_verdict


@dataclass(frozen=True, slots=True)
class TokenLengthPoint:
    ctx_size: int
    minimum: int
    median: float
    maximum: int


def _validated_rows(table: pa.Table, *, task: str | None) -> tuple[dict[str, Any], ...]:
    if not isinstance(table, pa.Table):
        raise TypeError("accuracy summaries require a PyArrow table")
    if table.schema != TABLE_SCHEMAS["results"]:
        raise ValueError("accuracy table uses the wrong schema")
    if task is not None and task not in TASKS:
        raise ValueError(f"Unknown FLenQA task: {task!r}")
    rows = tuple(table.to_pylist())
    prompt_ids = tuple(str(row["prompt_id"]) for row in rows)
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("accuracy table contains duplicate prompt IDs")
    for row in rows:
        ctx_size = int(row["ctx_size"])
        provenance = row["provenance"]
        if not provenance:
            raise ValueError("accuracy row must contain source provenance")
        if any(int(item["ctx_size"]) != ctx_size for item in provenance):
            raise ValueError("accuracy provenance mixes nominal context sizes")
    return tuple(row for row in rows if task is None or row["task"] == task)


def _accuracy_points(
    groups: dict[int, list[tuple[bool, bool | None]]],
) -> tuple[AccuracyPoint, ...]:
    return tuple(
        AccuracyPoint(
            ctx_size=ctx_size,
            correct=sum(correct for correct, _ in observations),
            total=len(observations),
            no_verdict=sum(verdict is None for _, verdict in observations),
        )
        for ctx_size, observations in sorted(groups.items())
    )


def summarize_unique_prompts(
    table: pa.Table,
    *,
    task: str | None = None,
) -> tuple[AccuracyPoint, ...]:
    """Summarize accuracy with each unique final prompt weighted once."""
    groups: dict[int, list[tuple[bool, bool | None]]] = {}
    for row in _validated_rows(table, task=task):
        groups.setdefault(int(row["ctx_size"]), []).append(
            (bool(row["correct"]), row["verdict"])
        )
    return _accuracy_points(groups)


def summarize_paper_random(
    table: pa.Table,
    *,
    task: str | None = None,
) -> tuple[AccuracyPoint, ...]:
    """Reproduce the paper's random-placement source-row weighting."""
    groups: dict[int, list[tuple[bool, bool | None]]] = {}
    for row in _validated_rows(table, task=task):
        ctx_size = int(row["ctx_size"])
        observation = (bool(row["correct"]), row["verdict"])
        for provenance in row["provenance"]:
            if provenance["dispersion"] == "random":
                groups.setdefault(ctx_size, []).append(observation)
    return _accuracy_points(groups)


def summarize_verdicts(table: pa.Table) -> tuple[VerdictCountPoint, ...]:
    """Count unique-prompt True, False, and missing verdicts by length."""
    groups: dict[int, list[bool | None]] = {}
    for row in _validated_rows(table, task=None):
        groups.setdefault(int(row["ctx_size"]), []).append(row["verdict"])
    return tuple(
        VerdictCountPoint(
            ctx_size=ctx_size,
            true=sum(verdict is True for verdict in verdicts),
            false=sum(verdict is False for verdict in verdicts),
            no_verdict=sum(verdict is None for verdict in verdicts),
        )
        for ctx_size, verdicts in sorted(groups.items())
    )


def summarize_token_lengths(table: pa.Table) -> tuple[TokenLengthPoint, ...]:
    """Summarize exact model-token lengths inside each nominal bucket."""
    groups: dict[int, list[int]] = {}
    for row in _validated_rows(table, task=None):
        groups.setdefault(int(row["ctx_size"]), []).append(int(row["n_input_tokens"]))
    return tuple(
        TokenLengthPoint(
            ctx_size=ctx_size,
            minimum=min(lengths),
            median=float(statistics.median(lengths)),
            maximum=max(lengths),
        )
        for ctx_size, lengths in sorted(groups.items())
    )


__all__ = [
    "FULL_UNIQUE_PROMPT_COUNTS",
    "AccuracyPoint",
    "TokenLengthPoint",
    "VerdictCountPoint",
    "summarize_paper_random",
    "summarize_token_lengths",
    "summarize_unique_prompts",
    "summarize_verdicts",
]
