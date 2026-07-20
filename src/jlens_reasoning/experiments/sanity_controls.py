"""Deterministic negative-control calculations for J-Lens sanity runs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
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
CONTROL_CASE_KEYS = (
    "spider",
    "france_capital",
    "france_language",
    "france_continent",
    "france_currency",
)
IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 1e-5
NORM_ATOL = 1e-6
NORM_RTOL = 1e-5
PERCENTILE_QUANTILE = 0.95
PERCENTILE_INTERPRETATION = "deterministic sanity check; not statistical significance"
CONTROL_CHECK_MAP = (
    ("identity", "identity_control"),
    ("matched_random_vector", "matched_random_vector_control"),
    ("wrong_concept", "wrong_concept_control"),
    ("random_target", "random_target_control"),
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
        return torch.zeros_like(real_vector)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_subseed(base_seed, layer_index, role))
    atol, rtol = _norm_tolerances(real_vector.dtype)
    for _ in range(1024):
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
        "Unable to generate a finite norm-matched random vector after conversion"
    )


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


def decode_token(tokenizer: Any, token_id: int) -> str:
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
    vocabulary_ids = sorted(
        {int(token_id) for token_id in tokenizer.get_vocab().values()}
    )
    added_special_ids = {
        int(token_id)
        for token_id, token in getattr(tokenizer, "added_tokens_decoder", {}).items()
        if getattr(token, "special", False)
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
        | added_special_ids,
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
) -> list[dict[str, Any]]:
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
        digest = hashlib.sha256(f"jlens-random-target-v1:{seed}".encode()).digest()
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


def require_exact_cases(
    results: Sequence[Mapping[str, Any]],
) -> None:
    actual_keys = [str(result["key"]) for result in results]
    expected = list(CONTROL_CASE_KEYS)
    if actual_keys != expected:
        raise ValueError(f"Expected exact case keys {expected!r}, got {actual_keys!r}")


def summarize_wrong_concept(
    matched_cases: Sequence[Mapping[str, Any]],
    mismatched_cases: Sequence[Mapping[str, Any]],
    *,
    required_winning_case_count: int = 4,
) -> dict[str, Any]:
    require_exact_cases(matched_cases)
    require_exact_cases(mismatched_cases)
    cases = []
    for matched, mismatched in zip(
        matched_cases,
        mismatched_cases,
        strict=True,
    ):
        matched_gain = float(matched["log_rank_gain"])
        mismatched_gain = float(mismatched["log_rank_gain"])
        cases.append(
            {
                "key": matched["key"],
                "matched_log_rank_gain": matched_gain,
                "mismatched_log_rank_gain": mismatched_gain,
                "comparison": "matched_log_rank_gain > mismatched_log_rank_gain",
                "matched_wins": matched_gain > mismatched_gain,
            }
        )
    matched_mean = mean([float(case["log_rank_gain"]) for case in matched_cases])
    mismatched_mean = mean([float(case["log_rank_gain"]) for case in mismatched_cases])
    winning_case_count = sum(bool(case["matched_wins"]) for case in cases)
    aggregate_condition = matched_mean > mismatched_mean
    case_condition = winning_case_count >= required_winning_case_count
    return {
        "cases": cases,
        "matched_mean_log_rank_gain": matched_mean,
        "mismatched_mean_log_rank_gain": mismatched_mean,
        "aggregate_comparison": (
            "matched_mean_log_rank_gain > mismatched_mean_log_rank_gain"
        ),
        "aggregate_condition": aggregate_condition,
        "matched_winning_case_count": winning_case_count,
        "required_winning_case_count": required_winning_case_count,
        "case_condition": case_condition,
        "passed": aggregate_condition and case_condition,
    }


def controls_passed(controls: Mapping[str, Mapping[str, Any]]) -> bool:
    return all(bool(controls[name]["passed"]) for name, _ in CONTROL_CHECK_MAP)


def _control_failure(name: str, control: Mapping[str, Any]) -> str:
    if name == "identity":
        return (
            "identity control failed: "
            f"{control.get('passed_case_count', 0)}/"
            f"{control.get('required_case_count', 5)} cases passed; "
            "required every identity comparison to pass; "
            "maximum absolute logit difference="
            f"{control.get('maximum_absolute_logit_difference')!r}"
        )
    if name == "matched_random_vector":
        return (
            "matched random vector control failed: real mean log-rank gain="
            f"{control.get('real_mean_log_rank_gain')!r}; required strictly > "
            "95th-percentile sanity threshold="
            f"{control.get('percentile_95_threshold')!r}"
        )
    if name == "wrong_concept":
        return (
            "wrong concept control failed: matched mean log-rank gain="
            f"{control.get('matched_mean_log_rank_gain')!r}, mismatched mean="
            f"{control.get('mismatched_mean_log_rank_gain')!r}; matched wins="
            f"{control.get('matched_winning_case_count')!r}; required matched "
            "mean > mismatched mean and at least "
            f"{control.get('required_winning_case_count', 4)} strict case wins"
        )
    if name == "random_target":
        return (
            "random target control failed: real mean log-rank gain="
            f"{control.get('real_mean_log_rank_gain')!r}; required strictly > "
            "95th-percentile sanity threshold="
            f"{control.get('percentile_95_threshold')!r}"
        )
    raise KeyError(f"Unknown control: {name}")


def aggregate_all_checks(
    existing_checks: Mapping[str, bool],
    existing_failures: Sequence[str],
    controls: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bool], list[str], bool]:
    checks = dict(existing_checks)
    failures = list(existing_failures)
    for control_name, check_name in CONTROL_CHECK_MAP:
        if control_name not in controls:
            raise KeyError(f"Missing control payload: {control_name}")
        passed = bool(controls[control_name]["passed"])
        checks[check_name] = passed
        if not passed:
            failures.append(_control_failure(control_name, controls[control_name]))
    return checks, failures, all(checks.values())
