"""Token-count lens-validity preflight for long prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from experiments.flenqa_length_drift.constants import SUMMARY_POSITION_BUDGET
from jlens_reasoning.evaluation_utils import best_token_rank


def _token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    encoded = tokenizer(text, truncation=False)
    if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
        raise ValueError("tokenizer must return input_ids")
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if (
        isinstance(ids, Sequence)
        and ids
        and isinstance(ids[0], Sequence)
        and not isinstance(ids[0], (str, bytes))
    ):
        if len(ids) != 1:
            raise ValueError("preflight tokenizer output must have one batch")
        ids = ids[0]
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)):
        raise ValueError("preflight tokenizer input_ids must be a sequence")
    return tuple(int(token_id) for token_id in ids)


def _token_count(tokenizer: Any, text: str) -> int:
    return len(_token_ids(tokenizer, text))


def _runner_ids(value: Any) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], Sequence):
        if len(value) != 1:
            raise RuntimeError("lens preflight input IDs must have one batch")
        value = value[0]
    return tuple(int(token_id) for token_id in value)


def _preflight_positions(
    *,
    actual_tokens: int,
    semantic_tokens: int,
) -> tuple[int, ...]:
    if semantic_tokens > SUMMARY_POSITION_BUDGET:
        raise ValueError(
            "preflight semantic suffix exceeds the summary position budget"
        )
    if semantic_tokens > actual_tokens:
        raise ValueError("preflight semantic suffix exceeds the padded prompt")

    semantic_start = actual_tokens - semantic_tokens
    semantic_positions = tuple(range(semantic_start, actual_tokens))
    padding_budget = SUMMARY_POSITION_BUDGET - semantic_tokens
    padding_count = min(semantic_start, padding_budget)
    if padding_count == semantic_start:
        padding_positions = tuple(range(semantic_start))
    elif padding_count == 1:
        padding_positions = (0,)
    elif padding_count > 1:
        padding_positions = tuple(
            index * (semantic_start - 1) // (padding_count - 1)
            for index in range(padding_count)
        )
    else:
        padding_positions = ()
    return padding_positions + semantic_positions


def evaluate_lens_preflight(
    prompt: str,
    *,
    tokenizer: Any,
    jacobian_runner: Any,
    logit_runner: Any,
    target_surfaces: Sequence[str],
    positions: Sequence[int],
    max_seq_len: int = 4096,
) -> tuple[int, int]:
    """Return best target ranks across layers/positions for both lens modes."""
    input_ids = _token_ids(tokenizer, prompt)
    if len(input_ids) > max_seq_len:
        raise ValueError("preflight prompt exceeds the lens sequence limit")
    selected_positions = tuple(positions)
    if not selected_positions:
        raise ValueError("lens preflight positions must be nonempty")
    if (
        len(selected_positions) > SUMMARY_POSITION_BUDGET
        or selected_positions != tuple(sorted(set(selected_positions)))
        or selected_positions[0] < 0
        or selected_positions[-1] >= len(input_ids)
    ):
        raise ValueError("lens preflight positions are invalid")
    jacobian = jacobian_runner.run(
        prompt,
        positions=selected_positions,
        max_seq_len=max_seq_len,
    )
    logit = logit_runner.run(
        prompt,
        positions=selected_positions,
        max_seq_len=max_seq_len,
    )
    if (
        _runner_ids(jacobian.input_ids) != input_ids
        or _runner_ids(logit.input_ids) != input_ids
    ):
        raise RuntimeError("lens preflight token IDs differ")
    if not torch.equal(jacobian.model_logits, logit.model_logits):
        raise RuntimeError("lens preflight model logits differ")
    target_ids = tuple(
        int(encoded[0])
        for surface in target_surfaces
        if len(
            encoded := tokenizer.encode(
                surface,
                add_special_tokens=False,
            )
        )
        == 1
    )
    if not target_ids:
        raise ValueError("preflight target has no single-token surface")

    def best(logits_by_layer: Mapping[int, torch.Tensor]) -> int:
        return min(
            best_token_rank(row, target_ids)
            for layer_logits in logits_by_layer.values()
            for row in layer_logits
        )

    return best(jacobian.logits_by_layer), best(logit.logits_by_layer)


def _prefix(filler: str, repeats: int, prompt: str) -> str:
    return f"{' '.join([filler] * repeats)} {prompt}" if repeats else prompt


def pad_to_token_count(
    prompt: str,
    *,
    filler: str,
    target_tokens: int,
    tokenizer: Any,
) -> str:
    """Prefix filler until the untruncated tokenizer reaches an exact target."""
    if type(target_tokens) is not int or target_tokens <= 0:
        raise ValueError("target_tokens must be a positive integer")
    if not prompt or not filler.strip():
        raise ValueError("prompt and filler must be nonempty")
    current = _token_count(tokenizer, prompt)
    if current > target_tokens:
        raise ValueError(f"Prompt token count {current} exceeds target {target_tokens}")
    if current == target_tokens:
        return prompt

    low = 0
    high = 1
    while _token_count(tokenizer, _prefix(filler, high, prompt)) < target_tokens:
        low = high
        high *= 2
        if high > target_tokens * 4:
            raise ValueError("filler does not increase the prompt token count")
    while low + 1 < high:
        middle = (low + high) // 2
        if _token_count(tokenizer, _prefix(filler, middle, prompt)) < target_tokens:
            low = middle
        else:
            high = middle
    padded = _prefix(filler, high, prompt)
    actual = _token_count(tokenizer, padded)
    if actual != target_tokens:
        raise ValueError(
            f"Filler cannot reach exact target {target_tokens}; reached {actual}"
        )
    return padded


@dataclass(frozen=True, slots=True)
class PreflightCaseResult:
    target_tokens: int
    actual_tokens: int
    jacobian_rank: int
    logit_rank: int
    passed: bool


@dataclass(frozen=True, slots=True)
class PreflightResult:
    results: tuple[PreflightCaseResult, ...]
    passed: bool


class PreflightEvaluator(Protocol):
    def __call__(
        self,
        prompt: str,
        *,
        positions: Sequence[int],
    ) -> tuple[int, int]: ...


def run_preflight(
    *,
    prompt: str,
    filler: str,
    target_tokens: Sequence[int],
    tokenizer: Any,
    evaluate: PreflightEvaluator,
) -> PreflightResult:
    """Require the Jacobian target rank to beat logit lens at every token length."""
    results: list[PreflightCaseResult] = []
    semantic_tokens = _token_count(tokenizer, prompt)
    for target in target_tokens:
        padded = pad_to_token_count(
            prompt,
            filler=filler,
            target_tokens=target,
            tokenizer=tokenizer,
        )
        actual = _token_count(tokenizer, padded)
        positions = _preflight_positions(
            actual_tokens=actual,
            semantic_tokens=semantic_tokens,
        )
        jacobian_rank, logit_rank = evaluate(padded, positions=positions)
        passed = jacobian_rank < logit_rank
        results.append(
            PreflightCaseResult(
                target_tokens=target,
                actual_tokens=actual,
                jacobian_rank=jacobian_rank,
                logit_rank=logit_rank,
                passed=passed,
            )
        )
    return PreflightResult(
        results=tuple(results),
        passed=all(result.passed for result in results),
    )
