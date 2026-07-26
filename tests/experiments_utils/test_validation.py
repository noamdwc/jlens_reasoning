from types import SimpleNamespace

import pytest
import torch

from jlens_reasoning.experiments_utils.validation import (
    validate_model_lens,
    workspace_layers,
    workspace_loading,
)


def test_workspace_layers_use_caller_bounds() -> None:
    assert workspace_layers(
        20,
        range(20),
        lower_fraction=0.35,
        upper_fraction=0.80,
    ) == list(range(7, 17))


def test_workspace_layers_reject_invalid_bounds() -> None:
    with pytest.raises(ValueError, match="0 <= lower <= upper <= 1"):
        workspace_layers(
            20,
            range(20),
            lower_fraction=0.9,
            upper_fraction=0.2,
        )


def test_workspace_loading_averages_layers_and_positions() -> None:
    activations = {
        2: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        3: torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
    }
    vectors = {
        2: torch.tensor([1.0, 0.0]),
        3: torch.tensor([1.0, 0.0]),
    }

    assert workspace_loading(
        activations,
        vectors,
        positions=[0, 1],
    ) == pytest.approx(0.75)


def test_validate_model_lens_rejects_width_and_layer_mismatches() -> None:
    lens = SimpleNamespace(d_model=4, source_layers=[0, 1, 2, 3])
    with pytest.raises(ValueError, match="residual width"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=5), lens)

    lens.source_layers = [0, 4]
    with pytest.raises(ValueError, match="fitted layers"):
        validate_model_lens(SimpleNamespace(n_layers=4, d_model=4), lens)
