"""J-Lens intervention mechanics shared by experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch import nn


def jlens_vector(
    lens: Any,
    unembedding_weight: torch.Tensor,
    *,
    layer: int,
    token_id: int,
) -> torch.Tensor:
    jacobian = lens.jacobians[layer].to(
        device=unembedding_weight.device,
        dtype=torch.float32,
    )
    unembedding_row = unembedding_weight[token_id].to(dtype=torch.float32)
    return jacobian.T @ unembedding_row


def coordinate_swap(
    hidden: torch.Tensor,
    source_vector: torch.Tensor,
    target_vector: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if hidden.shape[-1] != source_vector.numel():
        raise ValueError("Source vector width does not match hidden width")
    if source_vector.shape != target_vector.shape:
        raise ValueError("Source and target vectors must have the same shape")

    working = hidden.float()
    vectors = torch.stack(
        (
            source_vector.to(device=hidden.device, dtype=torch.float32),
            target_vector.to(device=hidden.device, dtype=torch.float32),
        ),
        dim=-1,
    )
    coordinates = working @ torch.linalg.pinv(vectors).T
    delta = (coordinates.flip(-1) - coordinates) @ vectors.T
    return (working + float(alpha) * delta).to(dtype=hidden.dtype)


class LensCoordinateSwapper:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        *,
        alpha: float,
    ) -> None:
        self._blocks = blocks
        self._vectors_by_layer = dict(vectors_by_layer)
        self._alpha = alpha
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, layer: int):
        source_vector, target_vector = self._vectors_by_layer[layer]

        def patch(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            hidden = output if torch.is_tensor(output) else output[0]
            patched = coordinate_swap(
                hidden,
                source_vector,
                target_vector,
                alpha=self._alpha,
            )
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        return patch

    def __enter__(self) -> LensCoordinateSwapper:
        try:
            for layer in sorted(self._vectors_by_layer):
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._hook(layer))
                )
        except Exception:
            self._remove()
            raise
        return self

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __exit__(self, *exc: Any) -> None:
        self._remove()


def execute_intervention(
    *,
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    scoring_input: torch.Tensor,
    vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
) -> torch.Tensor:
    with (
        torch.inference_mode(),
        LensCoordinateSwapper(model.layers, vectors_by_layer, alpha=alpha),
    ):
        return forward_next_token(scoring_input)


def single_token_vectors_by_layer(
    *,
    lens: Any,
    unembedding_weight: torch.Tensor,
    layers: Sequence[int],
    token_id: int,
) -> dict[int, torch.Tensor]:
    return {
        layer: jlens_vector(
            lens,
            unembedding_weight,
            layer=layer,
            token_id=token_id,
        )
        for layer in layers
    }


def token_vectors_by_layer(
    *,
    lens: Any,
    unembedding_weight: torch.Tensor,
    layers: Sequence[int],
    source_token_id: int,
    target_token_id: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    source_vectors = single_token_vectors_by_layer(
        lens=lens,
        unembedding_weight=unembedding_weight,
        layers=layers,
        token_id=source_token_id,
    )
    target_vectors = single_token_vectors_by_layer(
        lens=lens,
        unembedding_weight=unembedding_weight,
        layers=layers,
        token_id=target_token_id,
    )
    return {layer: (source_vectors[layer], target_vectors[layer]) for layer in layers}
