"""Policy-free deterministic helpers for experiment controls."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


def mean(values: Sequence[float]) -> float:
    """Return an accurate arithmetic mean for a non-empty sequence."""
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(values) / len(values)


def percentile(values: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for a non-empty sequence."""
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


def percentile_label(quantile: float) -> str:
    """Format a quantile as a human-readable percentile label."""
    return f"{quantile * 100:g}th-percentile"


def strict_percentile_gate(
    real_score: float,
    control_scores: Sequence[float],
    *,
    quantile: float,
    interpretation: str,
) -> dict[str, Any]:
    """Compare a real score strictly against a caller-defined percentile."""
    threshold = percentile(control_scores, quantile)
    return {
        "real_score": float(real_score),
        "percentile": quantile,
        "threshold": threshold,
        "comparison": "real_score > threshold",
        "interpretation": interpretation,
        "passed": float(real_score) > threshold,
    }


def derive_subseed(
    base_seed: int,
    layer_index: int,
    role: str,
    *,
    namespace: str,
) -> int:
    """Derive a stable layer-and-role-specific seed within a namespace."""
    if role not in {"source", "target"}:
        raise ValueError("Role must be 'source' or 'target'")
    if not namespace:
        raise ValueError("Random namespace must be non-empty")
    payload = f"{namespace}:{base_seed}:{layer_index}:{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _norm_tolerances(
    dtype: torch.dtype,
    *,
    norm_atol: float,
    norm_rtol: float,
    low_precision_norm_atol: float,
    low_precision_norm_rtol: float,
) -> tuple[float, float]:
    if dtype in {torch.float16, torch.bfloat16}:
        return low_precision_norm_atol, low_precision_norm_rtol
    return norm_atol, norm_rtol


def _matched_random_vector(
    real_vector: torch.Tensor,
    *,
    base_seed: int,
    layer_index: int,
    role: str,
    namespace: str,
    norm_atol: float,
    norm_rtol: float,
    low_precision_norm_atol: float,
    low_precision_norm_rtol: float,
    max_attempts: int,
) -> torch.Tensor:
    real_cpu = real_vector.detach().to(device="cpu", dtype=torch.float32)
    if real_cpu.numel() == 0:
        raise ValueError("Cannot match the norm of an empty vector")
    real_norm = torch.linalg.vector_norm(real_cpu)
    if real_norm.item() == 0.0:
        return torch.zeros_like(real_vector)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        derive_subseed(
            base_seed,
            layer_index,
            role,
            namespace=namespace,
        )
    )
    atol, rtol = _norm_tolerances(
        real_vector.dtype,
        norm_atol=norm_atol,
        norm_rtol=norm_rtol,
        low_precision_norm_atol=low_precision_norm_atol,
        low_precision_norm_rtol=low_precision_norm_rtol,
    )
    for _ in range(max_attempts):
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
        converted = matched_cpu.to(
            device=real_vector.device,
            dtype=real_vector.dtype,
        )
        converted_norm = torch.linalg.vector_norm(converted.detach().float())
        if torch.isfinite(converted).all() and math.isclose(
            real_norm.item(),
            converted_norm.item(),
            abs_tol=atol,
            rel_tol=rtol,
        ):
            return converted

    raise RuntimeError(
        "Unable to generate a finite norm-matched random vector after "
        f"{max_attempts} conversion attempts"
    )


def matched_random_vectors(
    real_vectors: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    *,
    base_seed: int,
    namespace: str,
    norm_atol: float,
    norm_rtol: float,
    low_precision_norm_atol: float,
    low_precision_norm_rtol: float,
    max_attempts: int,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """Generate deterministic norm-matched vector pairs for each layer."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    generated: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    report: dict[str, dict[str, dict[str, Any]]] = {}
    for layer in sorted(real_vectors):
        source_vector, target_vector = real_vectors[layer]
        random_pair = (
            _matched_random_vector(
                source_vector,
                base_seed=base_seed,
                layer_index=layer,
                role="source",
                namespace=namespace,
                norm_atol=norm_atol,
                norm_rtol=norm_rtol,
                low_precision_norm_atol=low_precision_norm_atol,
                low_precision_norm_rtol=low_precision_norm_rtol,
                max_attempts=max_attempts,
            ),
            _matched_random_vector(
                target_vector,
                base_seed=base_seed,
                layer_index=layer,
                role="target",
                namespace=namespace,
                norm_atol=norm_atol,
                norm_rtol=norm_rtol,
                low_precision_norm_atol=low_precision_norm_atol,
                low_precision_norm_rtol=low_precision_norm_rtol,
                max_attempts=max_attempts,
            ),
        )
        generated[layer] = random_pair
        report[str(layer)] = {}
        for real_vector, random_vector, role in zip(
            (source_vector, target_vector),
            random_pair,
            ("source", "target"),
            strict=True,
        ):
            real_norm = torch.linalg.vector_norm(real_vector.detach().float()).item()
            random_norm = torch.linalg.vector_norm(
                random_vector.detach().float()
            ).item()
            atol, rtol = _norm_tolerances(
                real_vector.dtype,
                norm_atol=norm_atol,
                norm_rtol=norm_rtol,
                low_precision_norm_atol=low_precision_norm_atol,
                low_precision_norm_rtol=low_precision_norm_rtol,
            )
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


def decode_token(tokenizer: Any, token_id: int) -> str:
    """Decode one token without tokenizer cleanup."""
    return tokenizer.decode(
        [int(token_id)],
        clean_up_tokenization_spaces=False,
    )


def _encoded_ids(tokenizer: Any, surfaces: Sequence[str]) -> set[int]:
    return {
        int(token_id)
        for surface in surfaces
        for token_id in tokenizer.encode(surface, add_special_tokens=False)
    }


def build_random_target_exclusions(
    tokenizer: Any,
    *,
    source_surfaces: Sequence[str],
    target_surfaces: Sequence[str],
    clean_answer_surfaces: Sequence[str],
    intended_answer_surfaces: Sequence[str],
    formatting_token_ids: Iterable[int],
    existing_excluded_ids: Iterable[int] = (),
) -> dict[str, list[int]]:
    """Build an auditable token-ID exclusion report for random targets."""
    vocabulary_ids = sorted(
        {int(token_id) for token_id in tokenizer.get_vocab().values()}
    )
    added_control_ids = {
        int(token_id) for token_id in getattr(tokenizer, "added_tokens_decoder", {})
    }
    categories = {
        "sources": _encoded_ids(tokenizer, source_surfaces),
        "targets": _encoded_ids(tokenizer, target_surfaces),
        "clean_answers": _encoded_ids(tokenizer, clean_answer_surfaces),
        "intended_answers": _encoded_ids(tokenizer, intended_answer_surfaces),
        "formatting": {int(token_id) for token_id in formatting_token_ids},
        "reserved_special": {
            int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())
        }
        | added_control_ids,
        "decoded_formatting": {
            token_id
            for token_id in vocabulary_ids
            if not decode_token(tokenizer, token_id).strip()
        },
        "existing_filter": {int(token_id) for token_id in existing_excluded_ids},
    }
    all_ids: set[int] = set()
    for token_ids in categories.values():
        all_ids.update(token_ids)
    return {
        **{name: sorted(token_ids) for name, token_ids in categories.items()},
        "all": sorted(all_ids),
    }


def select_random_targets(
    tokenizer: Any,
    *,
    excluded_ids: Iterable[int],
    seeds: Sequence[int],
    output_vocab_size: int,
    namespace: str,
) -> list[dict[str, Any]]:
    """Select deterministic random target tokens from an eligible vocabulary."""
    if not namespace:
        raise ValueError("Random namespace must be non-empty")
    if output_vocab_size < 1:
        raise ValueError("Model output vocabulary must contain at least one token")
    excluded = {int(token_id) for token_id in excluded_ids}
    eligible = sorted(
        {
            int(token_id)
            for token_id in tokenizer.get_vocab().values()
            if 0 <= int(token_id) < output_vocab_size and int(token_id) not in excluded
        }
    )
    if not eligible:
        raise ValueError("No eligible random-target tokens remain")

    remaining = list(eligible)
    selected = []
    for seed in seeds:
        if not remaining:
            remaining = list(eligible)
        digest = hashlib.sha256(f"{namespace}:{seed}".encode()).digest()
        index = int.from_bytes(digest[:8], "big") % len(remaining)
        token_id = remaining.pop(index)
        selected.append(
            {
                "seed": int(seed),
                "token_id": token_id,
                "token": decode_token(tokenizer, token_id),
            }
        )
    return selected


def require_exact_case_keys(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
) -> None:
    """Require result keys to exactly match caller-owned keys and ordering."""
    actual_keys = [str(result["key"]) for result in results]
    expected = list(expected_keys)
    if actual_keys != expected:
        raise ValueError(f"Expected exact case keys {expected!r}, got {actual_keys!r}")
