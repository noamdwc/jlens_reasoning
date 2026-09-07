from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jlens_reasoning.experiments_utils.interventions import (
    LensCoordinatePatcher,
    LensCoordinateSwapper,
    coordinate_patch,
    coordinate_patch_from_activation,
    coordinate_swap,
    execute_intervention,
    jlens_vector,
    lens_coordinates,
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


def test_coordinate_patch_handles_multiple_non_orthogonal_directions() -> None:
    vectors = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    source = torch.tensor([2.0, 3.0, 7.0])
    target_coordinates = torch.tensor([-6.0, 11.0])

    actual = coordinate_patch(source, vectors, target_coordinates, alpha=1.0)
    expected = torch.tensor([5.0, 11.0, 7.0])

    assert torch.allclose(actual, expected)


def test_lens_coordinates_uses_pseudoinverse_for_non_orthogonal_vectors() -> None:
    vectors = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

    actual = lens_coordinates(torch.tensor([2.0, 3.0, 7.0]), vectors)

    assert torch.allclose(actual, torch.tensor([-1.0, 3.0]))


def test_coordinate_patch_from_activation_matches_explicit_coordinates() -> None:
    vectors = torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    source = torch.tensor([2.0, 3.0, 7.0])
    target = torch.tensor([5.0, 11.0, -4.0])

    actual = coordinate_patch_from_activation(source, target, vectors, alpha=1.0)
    expected = coordinate_patch(
        source,
        vectors,
        lens_coordinates(target, vectors),
        alpha=1.0,
    )

    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("alpha", [0.0, 1.0])
def test_coordinate_patch_preserves_shape_dtype_and_unselected_component(
    alpha: float,
) -> None:
    source = torch.tensor([[[1.0, 2.0, 9.0], [3.0, 4.0, 8.0]]], dtype=torch.bfloat16)
    target = torch.tensor([[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]])
    vectors = (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0]))

    target_coordinates = lens_coordinates(target, vectors)
    actual = coordinate_patch(source, vectors, target_coordinates, alpha=alpha)

    assert actual.shape == source.shape
    assert actual.dtype == source.dtype
    assert torch.equal(actual[..., 2], source[..., 2])
    if alpha == 0.0:
        assert torch.equal(actual, source)
    else:
        assert torch.equal(actual[0, 0, :2], torch.tensor([10.0, 20.0], dtype=torch.bfloat16))


@pytest.mark.parametrize(
    ("vectors", "match"),
    [
        (torch.ones(3), "vector width does not match"),
        (torch.ones((2, 0)), "At least one J-Lens vector"),
    ],
)
def test_coordinate_patch_rejects_invalid_shapes(
    vectors: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        coordinate_patch(torch.zeros(2), vectors, torch.zeros(2), alpha=1.0)


def test_coordinate_patch_rejects_invalid_target_coordinate_shape() -> None:
    with pytest.raises(ValueError, match="Target coordinates shape"):
        coordinate_patch(
            torch.zeros(2, 3),
            torch.eye(3),
            torch.zeros(2),
            alpha=1.0,
        )


def test_coordinate_patch_from_activation_rejects_activation_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="activations must have the same shape"):
        coordinate_patch_from_activation(
            torch.zeros(2),
            torch.zeros(3),
            torch.eye(2),
            alpha=1.0,
        )


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


def test_patcher_uses_target_coordinates_and_preserves_tuple_members() -> None:
    blocks = nn.ModuleList([TensorBlock(), TupleBlock()])
    vectors = {
        0: torch.tensor([[1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]),
        1: (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])),
    }
    target_coordinates = {
        0: torch.tensor([[[-6.0, 11.0]]]),
        1: torch.tensor([[[10.0, 20.0]]]),
    }
    hidden = torch.tensor([[[2.0, 3.0, 7.0]]])

    with LensCoordinatePatcher(blocks, vectors, target_coordinates, alpha=1.0):
        first = blocks[0](hidden)
        second, cache = blocks[1](hidden)

    assert torch.allclose(first, torch.tensor([[[5.0, 11.0, 7.0]]]))
    assert torch.equal(second, torch.tensor([[[10.0, 20.0, 7.0]]]))
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
