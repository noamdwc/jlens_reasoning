"""Model and workspace validation shared by experiments."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch


def validate_model_lens(model: Any, lens: Any) -> None:
    if model.d_model != lens.d_model:
        raise ValueError(
            f"Model/lens residual width mismatch: {model.d_model} != {lens.d_model}"
        )
    invalid = [layer for layer in lens.source_layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"Lens fitted layers {invalid} are outside model depth {model.n_layers}"
        )


def workspace_loading(
    activations_by_layer: Mapping[int, torch.Tensor],
    vectors_by_layer: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int],
) -> float:
    similarities = []
    for layer, vector in vectors_by_layer.items():
        hidden = activations_by_layer[layer][0, list(positions)].float()
        direction = vector.to(hidden.device, dtype=torch.float32).expand_as(hidden)
        similarities.append(
            torch.nn.functional.cosine_similarity(hidden, direction, dim=-1)
        )
    return float(torch.cat(similarities).mean().item())


def workspace_layers(
    n_layers: int,
    source_layers: Iterable[int],
    *,
    lower_fraction: float,
    upper_fraction: float,
) -> list[int]:
    if not 0.0 <= lower_fraction <= upper_fraction <= 1.0:
        raise ValueError("Workspace fractions must satisfy 0 <= lower <= upper <= 1")
    lower = math.ceil(lower_fraction * n_layers)
    upper = math.floor(upper_fraction * n_layers)
    return [layer for layer in source_layers if lower <= layer <= upper]
