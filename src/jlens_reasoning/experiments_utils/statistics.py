"""Uncertainty for paired experiments with repeated observations per problem."""

from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairedMeanInterval:
    mean: float
    low: float
    high: float
    n_problems: int


def paired_problem_bootstrap(
    problem_ids: Sequence[Hashable],
    differences: Sequence[float],
    *,
    seed: int,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> PairedMeanInterval:
    """Percentile bootstrap of the equally weighted mean problem difference.

    Supply already-paired differences (e.g. treated minus clean correctness).
    First average all observations within each problem, then resample problems.
    Variants and random seeds are not independent sampling units. The caller
    owns pairing and must supply the intended within-problem weighting/coverage.
    """
    values = np.asarray(differences, dtype=float)
    if (
        values.ndim != 1
        or len(problem_ids) != len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Supply one finite paired difference per problem ID")
    if type(n_resamples) is not int or n_resamples < 1 or not 0 < confidence < 1:
        raise ValueError(
            "Require positive resample count and confidence between 0 and 1"
        )
    grouped: dict[Hashable, list[float]] = {}
    for problem_id, difference in zip(problem_ids, values, strict=True):
        grouped.setdefault(problem_id, []).append(float(difference))
    if len(grouped) < 2:
        raise ValueError("Uncertainty requires at least two distinct problems")
    means = np.array([np.mean(group) for group in grouped.values()])
    rng = np.random.default_rng(seed)
    # One replicate at a time keeps memory independent of the resample count.
    samples = np.array(
        [
            rng.choice(means, size=len(means), replace=True).mean()
            for _ in range(n_resamples)
        ]
    )
    tail = (1 - confidence) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return PairedMeanInterval(float(means.mean()), float(low), float(high), len(means))
