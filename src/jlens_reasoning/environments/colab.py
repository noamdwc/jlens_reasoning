"""Reusable initialization for Google Colab notebooks."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import torch

from jlens_reasoning.config import ARTIFACT_ROOT_ENV, create_artifact_paths
from jlens_reasoning.environments.common import RuntimeContext, create_runtime_context
from jlens_reasoning.runtime import select_device
from jlens_reasoning.tracking import authenticate_wandb

DEFAULT_COLAB_ARTIFACT_ROOT = Path("/content/drive/MyDrive/jlens-reasoning")


def _get_colab_secret(name: str) -> str:
    from google.colab import userdata

    return userdata.get(name)


def _mount_google_drive() -> None:
    from google.colab import drive

    drive.mount("/content/drive")


def _required_secret(
    name: str,
    secret_getter: Callable[[str], str],
) -> str:
    try:
        value = secret_getter(name)
    except Exception:
        raise RuntimeError(f"Required Colab secret {name} is unavailable") from None

    if not value:
        raise RuntimeError(f"Required Colab secret {name} is unavailable")

    return value


def initialize_colab(
    *,
    enable_wandb: bool = True,
    require_cuda: bool = False,
    artifact_root: str | Path = DEFAULT_COLAB_ARTIFACT_ROOT,
    secret_getter: Callable[[str], str] | None = None,
    drive_mounter: Callable[[], None] | None = None,
    wandb_authenticator: Callable[..., bool] = authenticate_wandb,
    device_selector: Callable[..., torch.device] = select_device,
) -> RuntimeContext:
    """Mount Drive, optionally authenticate W&B, and return runtime paths."""

    secret_getter = secret_getter or _get_colab_secret
    drive_mounter = drive_mounter or _mount_google_drive

    try:
        drive_mounter()
    except Exception:
        raise RuntimeError("Google Drive mount failed") from None

    os.environ[ARTIFACT_ROOT_ENV] = str(artifact_root)
    paths = create_artifact_paths(artifact_root)
    os.environ["HF_HOME"] = str(paths.huggingface_cache)

    wandb_enabled = False
    if enable_wandb:
        wandb_key = _required_secret("WANDB_API_KEY", secret_getter)
        try:
            wandb_enabled = wandb_authenticator(
                api_key=wandb_key,
                enabled=True,
            )
        except Exception:
            raise RuntimeError("W&B authentication failed") from None

    device = device_selector(require_cuda=require_cuda)
    return create_runtime_context(
        paths=paths,
        device=device,
        wandb_enabled=wandb_enabled,
    )
