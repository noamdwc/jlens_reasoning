from pathlib import Path

import pytest
import torch

from jlens_reasoning.config import create_artifact_paths
from jlens_reasoning.environments.common import create_runtime_context
from jlens_reasoning.runtime import select_device


def test_cuda_is_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert select_device().type == "cuda"


def test_mps_is_used_for_lightweight_mac_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert select_device().type == "mps"


def test_cpu_is_the_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert select_device().type == "cpu"


def test_required_cuda_fails_instead_of_falling_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was required"):
        select_device(require_cuda=True)


def test_runtime_context_exposes_notebook_facing_fields(tmp_path: Path) -> None:
    paths = create_artifact_paths(tmp_path)
    context = create_runtime_context(
        paths=paths,
        device=torch.device("cpu"),
        wandb_enabled=False,
    )

    assert context.artifact_root == tmp_path
    assert context.datasets_dir == tmp_path / "datasets"
    assert context.huggingface_cache == tmp_path / "cache" / "huggingface"
    assert context.lenses_dir == tmp_path / "lenses"
    assert context.checkpoints_dir == tmp_path / "checkpoints"
    assert context.runs_dir == tmp_path / "runs"
    assert context.device.type == "cpu"
    assert context.wandb_enabled is False
