"""Differentiable output objectives usable by probe evaluation and analyses."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def token_margin(
    logits: torch.Tensor,
    positive_ids: Sequence[int],
    negative_ids: Sequence[int],
) -> torch.Tensor:
    """Mean positive-token logit minus mean negative-token logit, with gradients."""
    if logits.ndim != 1:
        raise ValueError("Output margin requires vector logits")
    if not positive_ids or not negative_ids:
        raise ValueError("Output margin requires nonempty token groups")
    if set(positive_ids) & set(negative_ids):
        raise ValueError("Output margin token groups must be disjoint")
    if any(i < 0 or i >= logits.numel() for i in (*positive_ids, *negative_ids)):
        raise ValueError("Output margin token is outside the vocabulary")
    positive_mean = logits[list(positive_ids)].float().mean()
    negative_mean = logits[list(negative_ids)].float().mean()
    return positive_mean - negative_mean
