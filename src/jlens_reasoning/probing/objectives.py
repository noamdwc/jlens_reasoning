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
    if (
        logits.ndim != 1
        or not positive_ids
        or not negative_ids
        or set(positive_ids) & set(negative_ids)
    ):
        raise ValueError(
            "Output margin requires nonempty disjoint token groups and vector logits"
        )
    if any(i < 0 or i >= logits.numel() for i in (*positive_ids, *negative_ids)):
        raise ValueError("Output margin token is outside the vocabulary")
    return (
        logits[list(positive_ids)].float().mean()
        - logits[list(negative_ids)].float().mean()
    )
