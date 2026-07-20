"""Deterministic negative-control calculations for J-Lens sanity runs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

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


def _matched_random_vector(
    real_vector: torch.Tensor,
    *,
    base_seed: int,
    layer_index: int,
    role: str,
) -> torch.Tensor:
    real_cpu = real_vector.detach().to(device="cpu", dtype=torch.float32)
    if real_cpu.numel() == 0:
        raise ValueError("Cannot match the norm of an empty vector")
    real_norm = torch.linalg.vector_norm(real_cpu)
    if real_norm.item() == 0.0:
        matched_cpu = torch.zeros_like(real_cpu)
    else:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(derive_subseed(base_seed, layer_index, role))
        random_vector = torch.randn(
            real_cpu.shape,
            generator=generator,
            dtype=torch.float32,
        )
        random_norm = torch.linalg.vector_norm(random_vector)
        if random_norm.item() == 0.0:
            random_vector.zero_()
            random_vector.reshape(-1)[0] = 1.0
            random_norm = torch.linalg.vector_norm(random_vector)
        matched_cpu = random_vector * (real_norm / random_norm)
    return matched_cpu.to(device=real_vector.device, dtype=real_vector.dtype)


def _norm_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in {torch.float16, torch.bfloat16}:
        return 1e-2, 1e-2
    return NORM_ATOL, NORM_RTOL


def matched_random_vectors(
    real_vectors: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    base_seed: int,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, dict[str, Any]]],
]:
    generated: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for layer in sorted(real_vectors):
        real_pair = real_vectors[layer]
        random_pair = tuple(
            _matched_random_vector(
                real_vector,
                base_seed=base_seed,
                layer_index=layer,
                role=role,
            )
            for real_vector, role in zip(
                real_pair,
                ("source", "target"),
                strict=True,
            )
        )
        generated[layer] = random_pair  # type: ignore[assignment]
        report[str(layer)] = {}
        for real_vector, random_vector, role in zip(
            real_pair,
            random_pair,
            ("source", "target"),
            strict=True,
        ):
            real_norm = torch.linalg.vector_norm(real_vector.detach().float()).item()
            random_norm = torch.linalg.vector_norm(
                random_vector.detach().float()
            ).item()
            atol, rtol = _norm_tolerances(real_vector.dtype)
            report[str(layer)][role] = {
                "real_norm": real_norm,
                "random_norm": random_norm,
                "atol": atol,
                "rtol": rtol,
                "matched": math.isclose(
                    real_norm,
                    random_norm,
                    abs_tol=atol,
                    rel_tol=rtol,
                ),
                "device": str(random_vector.device),
                "dtype": str(random_vector.dtype),
                "device_matches": random_vector.device == real_vector.device,
                "dtype_matches": random_vector.dtype == real_vector.dtype,
            }
    return generated, report
