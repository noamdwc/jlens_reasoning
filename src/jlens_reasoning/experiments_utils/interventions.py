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
    """Project one unembedding row into a fitted J-Lens layer direction."""
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
    """Swap source and target coordinates in the spanned residual subspace."""
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


def _prepare_vectors(
    hidden: torch.Tensor,
    vectors: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    """Normalize selected vectors into a [hidden_width, direction_count] matrix."""
    if hidden.ndim == 0:
        raise ValueError("Hidden activation must have at least one dimension")
    if torch.is_tensor(vectors):
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(-1)
        elif vectors.ndim != 2:
            raise ValueError("J-Lens vectors must be a vector matrix")
        working_vectors = vectors
    else:
        if not vectors:
            raise ValueError("At least one J-Lens vector is required")
        if any(vector.ndim != 1 for vector in vectors):
            raise ValueError("J-Lens vectors must be one-dimensional")
        if any(vector.numel() != hidden.shape[-1] for vector in vectors):
            raise ValueError("J-Lens vector width does not match hidden width")
        working_vectors = torch.stack(tuple(vectors), dim=-1)

    if working_vectors.shape[0] != hidden.shape[-1]:
        raise ValueError("J-Lens vector width does not match hidden width")
    if working_vectors.shape[1] == 0:
        raise ValueError("At least one J-Lens vector is required")
    return working_vectors.to(device=hidden.device, dtype=torch.float32)


def lens_coordinates(
    hidden: torch.Tensor,
    vectors: torch.Tensor | Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return the pseudoinverse coordinates of hidden in selected J-Lens directions."""
    working_vectors = _prepare_vectors(hidden, vectors)
    return hidden.float() @ torch.linalg.pinv(working_vectors).T


def coordinate_patch(
    hidden: torch.Tensor,
    vectors: torch.Tensor | Sequence[torch.Tensor],
    target_coordinates: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    """Patch selected J-Lens coordinates toward explicitly provided coefficients."""
    working_vectors = _prepare_vectors(hidden, vectors)
    expected_shape = (*hidden.shape[:-1], working_vectors.shape[1])
    if target_coordinates.shape != expected_shape:
        raise ValueError(
            "Target coordinates shape must match hidden prefix and vector count"
        )

    if alpha == 0.0:
        return hidden

    working = hidden.float()
    pinv = torch.linalg.pinv(working_vectors)
    source_coordinates = working @ pinv.T
    target_coordinates = target_coordinates.to(
        device=hidden.device,
        dtype=torch.float32,
    )
    delta = (target_coordinates - source_coordinates) @ working_vectors.T
    return (working + float(alpha) * delta).to(dtype=hidden.dtype)


def coordinate_patch_from_activation(
    hidden: torch.Tensor,
    target: torch.Tensor,
    vectors: torch.Tensor | Sequence[torch.Tensor],
    *,
    alpha: float,
) -> torch.Tensor:
    """Patch selected J-Lens coordinates from a same-shaped target activation."""
    if hidden.shape != target.shape:
        raise ValueError("Source and target activations must have the same shape")
    target_coordinates = lens_coordinates(target, vectors)
    return coordinate_patch(
        hidden,
        vectors,
        target_coordinates,
        alpha=alpha,
    )


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
            """Patch one hooked layer output with the configured coordinate swap."""
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


class LensCoordinatePatcher:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        vectors_by_layer: Mapping[int, torch.Tensor | Sequence[torch.Tensor]],
        target_coordinates_by_layer: Mapping[int, torch.Tensor],
        *,
        alpha: float,
    ) -> None:
        self._blocks = blocks
        self._vectors_by_layer = dict(vectors_by_layer)
        self._target_coordinates_by_layer = dict(target_coordinates_by_layer)
        self._alpha = alpha
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, layer: int):
        vectors = self._vectors_by_layer[layer]
        target_coordinates = self._target_coordinates_by_layer[layer]

        def patch(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            """Patch one hooked layer output toward configured target coordinates."""
            del module, inputs
            hidden = output if torch.is_tensor(output) else output[0]
            patched = coordinate_patch(
                hidden,
                vectors,
                target_coordinates,
                alpha=self._alpha,
            )
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        return patch

    def __enter__(self) -> LensCoordinatePatcher:
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
    """Patch configured model layers while evaluating one intervention condition."""
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
    """Build one token's J-Lens direction for every requested layer."""
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
    """Build paired source and target J-Lens directions by layer."""
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
