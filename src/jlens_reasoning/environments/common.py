"""Shared environment context returned to notebooks and scripts."""

from dataclasses import dataclass
from pathlib import Path

import torch

from jlens_reasoning.config import ArtifactPaths


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    device: torch.device
    artifact_root: Path
    datasets_dir: Path
    huggingface_cache: Path
    lenses_dir: Path
    checkpoints_dir: Path
    runs_dir: Path
    wandb_enabled: bool


def create_runtime_context(
    *,
    paths: ArtifactPaths,
    device: torch.device,
    wandb_enabled: bool,
) -> RuntimeContext:
    return RuntimeContext(
        device=device,
        artifact_root=paths.root,
        datasets_dir=paths.datasets,
        huggingface_cache=paths.huggingface_cache,
        lenses_dir=paths.lenses,
        checkpoints_dir=paths.checkpoints,
        runs_dir=paths.runs,
        wandb_enabled=wandb_enabled,
    )
