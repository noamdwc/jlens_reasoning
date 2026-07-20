"""Deterministic negative-control calculations for J-Lens sanity runs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

CONTROL_SEEDS = (
    11,
    29,
    47,
    71,
    101,
    131,
    167,
    199,
    239,
    281,
    331,
    379,
    431,
    487,
    547,
    607,
)
IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 1e-5
NORM_ATOL = 1e-6
NORM_RTOL = 1e-5
PERCENTILE_QUANTILE = 0.95
PERCENTILE_INTERPRETATION = (
    "deterministic sanity check; not statistical significance"
)


def log_rank_gain(clean_rank: int, intervened_rank: int) -> float:
    if clean_rank < 1 or intervened_rank < 1:
        raise ValueError("Ranks must be positive one-based integers")
    return math.log(clean_rank) - math.log(intervened_rank)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(values) / len(values)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def strict_percentile_gate(
    real_score: float,
    control_scores: Sequence[float],
    *,
    quantile: float = PERCENTILE_QUANTILE,
) -> dict[str, Any]:
    threshold = percentile(control_scores, quantile)
    return {
        "real_score": float(real_score),
        "percentile": quantile,
        "threshold": threshold,
        "comparison": "real_score > threshold",
        "interpretation": PERCENTILE_INTERPRETATION,
        "passed": float(real_score) > threshold,
    }


def derive_subseed(base_seed: int, layer_index: int, role: str) -> int:
    if role not in {"source", "target"}:
        raise ValueError("Role must be 'source' or 'target'")
    payload = f"jlens-control-v1:{base_seed}:{layer_index}:{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
