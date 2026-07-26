import math
import re
import unicodedata
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import torch


class ReasoningStatus(StrEnum):
    NOT_PRESENT = "not_present"
    PARSED = "parsed"
    MALFORMED = "malformed_reasoning"


def no_reasoning(text: str) -> tuple[str, ReasoningStatus]:
    """Return text unchanged while marking reasoning as absent."""
    return text, ReasoningStatus.NOT_PRESENT


# Matches complete, non-nested reasoning spans. Any leftover tag is malformed.
_THINK_SPAN = re.compile(r"<think>(?:(?!</?think>).)*</think>", re.DOTALL)


def parse_think_tags(text: str) -> tuple[str, ReasoningStatus]:
    """Remove a valid leading think block and classify its structure."""
    if "<think>" not in text and "</think>" not in text:
        return text, ReasoningStatus.NOT_PRESENT
    visible = _THINK_SPAN.sub("", text)
    if "<think>" in visible or "</think>" in visible:
        return "", ReasoningStatus.MALFORMED
    return visible, ReasoningStatus.PARSED


def normalize_text(text: str) -> str:
    """Normalize answer text for case-insensitive factual comparison."""
    return unicodedata.normalize("NFC", text).strip().casefold().rstrip(".!?")


def answer_token_variants(
    tokenizer: Any,
    accepted_references: Sequence[str],
) -> tuple[tuple[int, str], ...]:
    """Resolve accepted answers to unique case-and-space single-token variants.

    Returns ordered ``(token_id, surface)`` pairs, keeping the first surface per ID.
    """
    variants: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    for reference in accepted_references:
        if not isinstance(reference, str) or not normalize_text(reference):
            raise ValueError("Accepted references must normalize to non-empty text")
        bases = (
            reference,
            reference.lower(),
            reference.capitalize(),
            reference.upper(),
        )
        for base in dict.fromkeys(bases):
            for surface in (base, f" {base}"):
                token_ids = tokenizer.encode(surface, add_special_tokens=False)
                if len(token_ids) == 1 and int(token_ids[0]) not in seen_ids:
                    token_id = int(token_ids[0])
                    seen_ids.add(token_id)
                    variants.append((token_id, surface))
    if not variants:
        raise ValueError("Accepted references have no single-token variants")
    return tuple(variants)


def best_token_rank(logits: torch.Tensor, target_ids: Sequence[int]) -> int:
    """Find the best deterministic one-based rank among target token IDs.

    Returns the smallest target rank, breaking logit ties by lower vocabulary ID.
    """
    if logits.ndim != 1:
        raise ValueError("best_token_rank expects one logits vector")
    if not target_ids:
        raise ValueError("best_token_rank needs at least one target token")
    token_ids = torch.arange(logits.numel(), device=logits.device)
    ranks = []
    for target_id in target_ids:
        target_logit = logits[target_id]
        higher = (logits > target_logit).sum()
        earlier_ties = ((logits == target_logit) & (token_ids < target_id)).sum()
        ranks.append(1 + int(higher.item()) + int(earlier_ties.item()))
    return min(ranks)


def top_token_values(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    k: int,
) -> tuple[tuple[int, str, float], ...]:
    """Select and decode the highest-logit tokens.

    Returns ordered ``(token_id, decoded_surface, logit)`` tuples.
    """
    if k < 0:
        raise ValueError("top-token count must be non-negative")
    values, indices = torch.topk(logits, k=min(k, logits.numel()))
    return tuple(
        (
            int(token_id),
            tokenizer.decode(
                [int(token_id)],
                clean_up_tokenization_spaces=False,
            ),
            float(value),
        )
        for value, token_id in zip(values.tolist(), indices.tolist(), strict=True)
    )


def log_rank_gain(baseline_rank: int, candidate_rank: int) -> float:
    """Measure the natural-log rank change from baseline to candidate.

    Returns ``log(baseline_rank) - log(candidate_rank)``; positive means improvement.
    """
    if baseline_rank < 1 or candidate_rank < 1:
        raise ValueError("Ranks must be positive one-based integers")
    return math.log(baseline_rank) - math.log(candidate_rank)


def extract_answer(evaluation_text: str) -> str | None:
    """Extract the first sentence-like answer segment from evaluated text."""
    answer = re.split(r"[,.!?\n]", evaluation_text, maxsplit=1)[0].strip()
    return answer or None


def match_reference(normalized_answer: str, references: tuple[str, ...]) -> str | None:
    """Return the accepted reference matching a normalized answer, if any."""
    return next((r for r in references if normalize_text(r) == normalized_answer), None)


def safe_truncated_text(text: str) -> str | None:
    """Keep only complete sentence-like text from a truncated generation."""
    boundary = max(text.rfind(character) for character in ".!?\n")
    return None if boundary < 0 else text[: boundary + 1].strip()
