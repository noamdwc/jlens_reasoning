"""Deterministic True/False scoring for FLenQA prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from jlens_reasoning.evaluation_utils import (
    answer_token_variants,
    best_token_rank,
    extract_answer,
    match_reference,
    normalize_text,
)


@dataclass(frozen=True, slots=True)
class BinaryAnswerScore:
    logit_true: float
    logit_false: float
    rank_true: int
    rank_false: int
    predicted: bool
    correct: bool
    generated_text: str | None
    extracted: bool | None
    generated_correct: bool | None
    agrees: bool | None


def extract_generated_verdict(text: str) -> bool | None:
    """Extract a front-loaded binary verdict without consulting the gold label."""
    extracted = extract_answer(text)
    if extracted is None:
        return None
    matched = match_reference(
        normalize_text(extracted),
        ("true", "false"),
    )
    if matched is None:
        return None
    return normalize_text(matched) == "true"


def _variant_ids(tokenizer: Any, reference: str) -> tuple[int, ...]:
    return tuple(
        token_id
        for token_id, _surface in answer_token_variants(tokenizer, (reference,))
    )


def score_binary_answer(
    final_logits: torch.Tensor,
    *,
    tokenizer: Any,
    label: bool,
    generated_text: str | None = None,
) -> BinaryAnswerScore:
    """Score constrained logits and an optional short generated verdict."""
    if final_logits.ndim != 1:
        raise ValueError("binary scoring expects one final logits vector")
    true_ids = _variant_ids(tokenizer, "True")
    false_ids = _variant_ids(tokenizer, "False")
    if set(true_ids) & set(false_ids):
        raise ValueError("True and False token variants must be disjoint")
    if max((*true_ids, *false_ids)) >= final_logits.numel():
        raise ValueError("answer token variant is outside the logits vocabulary")

    rank_true = best_token_rank(final_logits, true_ids)
    rank_false = best_token_rank(final_logits, false_ids)
    logit_true = max(float(final_logits[token_id].item()) for token_id in true_ids)
    logit_false = max(float(final_logits[token_id].item()) for token_id in false_ids)
    predicted = rank_true < rank_false
    extracted = (
        None if generated_text is None else extract_generated_verdict(generated_text)
    )
    generated_correct = None if extracted is None else extracted is label
    agrees = None if extracted is None else extracted is predicted
    return BinaryAnswerScore(
        logit_true=logit_true,
        logit_false=logit_false,
        rank_true=rank_true,
        rank_false=rank_false,
        predicted=predicted,
        correct=predicted is label,
        generated_text=generated_text,
        extracted=extracted,
        generated_correct=generated_correct,
        agrees=agrees,
    )
