from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jlens_reasoning.experiments_utils.interventions import (
    LensCoordinateSwapper,
    coordinate_swap,
    execute_intervention,
    jlens_vector,
)


def test_jlens_vector_composes_jacobian_and_unembedding() -> None:
    lens = SimpleNamespace(jacobians={1: torch.tensor([[1.0, 2.0], [3.0, 4.0]])})
    unembedding = torch.tensor([[0.0, 0.0], [5.0, 6.0]])

    assert torch.equal(
        jlens_vector(lens, unembedding, layer=1, token_id=1),
        torch.tensor([23.0, 34.0]),
    )


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [
        (0.0, [1.0, 0.0, 7.0]),
        (1.0, [0.0, 1.0, 7.0]),
        (2.0, [-1.0, 2.0, 7.0]),
    ],
)
def test_coordinate_swap_strength_and_orthogonal_component(
    alpha: float,
    expected: list[float],
) -> None:
    actual = coordinate_swap(
        torch.tensor([1.0, 0.0, 7.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        alpha=alpha,
    )

    assert torch.allclose(actual, torch.tensor(expected))


def test_coordinate_swap_preserves_shape_and_dtype() -> None:
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.5, 0.25]]],
        dtype=torch.bfloat16,
    )

    actual = coordinate_swap(
        hidden,
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        alpha=1.0,
    )

    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype


class TensorBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class TupleBlock(nn.Module):
    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, str]:
        return hidden, "cache"


def test_swapper_patches_all_positions_and_preserves_tuple_members() -> None:
    blocks = nn.ModuleList([TensorBlock(), TupleBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
        1: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
        first = blocks[0](hidden)
        second, cache = blocks[1](hidden)

    assert torch.equal(first, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert torch.equal(second, torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]))
    assert cache == "cache"
    assert all(not block._forward_hooks for block in blocks)


def test_swapper_removes_hooks_after_exception() -> None:
    blocks = nn.ModuleList([TensorBlock()])
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }

    with pytest.raises(RuntimeError, match="stop"):
        with LensCoordinateSwapper(blocks, vectors, alpha=1.0):
            raise RuntimeError("stop")

    assert not blocks[0]._forward_hooks


def test_intervention_executor_removes_hooks_when_forward_raises() -> None:
    model = SimpleNamespace(layers=nn.ModuleList([TensorBlock()]))
    vectors = {
        0: (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])),
    }

    def fail(input_ids: torch.Tensor) -> torch.Tensor:
        del input_ids
        model.layers[0](torch.tensor([[[1.0, 0.0]]]))
        raise RuntimeError("forward failed")

    with pytest.raises(RuntimeError, match="forward failed"):
        execute_intervention(
            model=model,
            forward_next_token=fail,
            scoring_input=torch.tensor([[1]]),
            vectors_by_layer=vectors,
            alpha=1.0,
        )

    assert not model.layers[0]._forward_hooks
