"""Reusable initialization for Google Colab notebooks."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch

from jlens_reasoning.config import ARTIFACT_ROOT_ENV, create_artifact_paths
from jlens_reasoning.environments.common import RuntimeContext, create_runtime_context
from jlens_reasoning.runtime import select_device
from jlens_reasoning.tracking import authenticate_wandb

DEFAULT_COLAB_ARTIFACT_ROOT = Path("/content/drive/MyDrive/jlens-reasoning")


def source_bundle_sha256(project_dir: Path) -> str:
    """Hash the code and locked inputs sent to a Colab runtime."""
    source_files = [project_dir / "pyproject.toml", project_dir / "uv.lock"]
    for directory in ("src", "experiments", "notebooks"):
        source_files.extend(sorted((project_dir / directory).rglob("*.py")))
        source_files.extend(sorted((project_dir / directory).rglob("*.ipynb")))

    digest = hashlib.sha256()
    for path in source_files:
        contents = path.read_bytes()
        digest.update(path.relative_to(project_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def download_r2_inputs(
    *,
    client: Any,
    bucket: str,
    prefix: str,
    destination: Path,
    paths: Sequence[str],
) -> None:
    """Download selected files and directories beneath one R2 project prefix."""
    destination = destination.resolve()
    root_prefix = prefix.strip("/")
    for relative in paths:
        parts = relative.rstrip("/").split("/")
        if not root_prefix or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"Invalid R2 input path: {relative!r}")

        key = f"{root_prefix}/{relative}"
        if relative.endswith("/"):
            pages = client.get_paginator("list_objects_v2").paginate(
                Bucket=bucket, Prefix=key
            )
            keys = (item["Key"] for page in pages for item in page.get("Contents", []))
        else:
            keys = (key,)

        for object_key in keys:
            if object_key.endswith("/"):
                continue
            object_relative = object_key[len(root_prefix) + 1 :]
            if not object_key.startswith(f"{root_prefix}/{relative}") or ".." in (
                object_relative.split("/")
            ):
                raise ValueError(f"Invalid R2 object key: {object_key}")
            target = (destination / object_relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"R2 object escapes data directory: {object_key}")
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, object_key, str(target))


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
