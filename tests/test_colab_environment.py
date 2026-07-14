from pathlib import Path

import pytest
import torch

from jlens_reasoning.environments.colab import initialize_colab


def test_colab_initialization_mounts_drive_and_authenticates(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    secrets = {
        "HF_READ_TOKEN": "hf-secret",
        "WANDB_API_KEY": "wandb-secret",
    }

    context = initialize_colab(
        artifact_root=tmp_path,
        secret_getter=secrets.__getitem__,
        drive_mounter=lambda: events.append("drive-mounted"),
        hf_authenticator=lambda **kwargs: events.append(("hf", kwargs)),
        wandb_authenticator=lambda **kwargs: events.append(("wandb", kwargs)) or True,
        device_selector=lambda **_: torch.device("cuda"),
    )

    assert events == [
        "drive-mounted",
        (
            "hf",
            {
                "token": "hf-secret",
                "add_to_git_credential": False,
                "skip_if_logged_in": False,
            },
        ),
        (
            "wandb",
            {
                "api_key": "wandb-secret",
                "enabled": True,
            },
        ),
    ]
    assert context.device.type == "cuda"
    assert context.artifact_root == tmp_path
    assert context.wandb_enabled is True


def test_wandb_is_enabled_by_default_and_failure_is_fatal(tmp_path: Path) -> None:
    secrets = {
        "HF_READ_TOKEN": "hf-secret",
        "WANDB_API_KEY": "bad-wandb-secret",
    }

    with pytest.raises(RuntimeError, match="W&B authentication failed") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=secrets.__getitem__,
            drive_mounter=lambda: None,
            hf_authenticator=lambda **_: None,
            wandb_authenticator=lambda **_: (_ for _ in ()).throw(
                RuntimeError("W&B authentication failed")
            ),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "bad-wandb-secret" not in str(error.value)
    assert error.value.__cause__ is None


def test_wandb_can_be_explicitly_disabled(tmp_path: Path) -> None:
    requested: list[str] = []
    secrets = {"HF_READ_TOKEN": "hf-secret"}

    def get_secret(name: str) -> str:
        requested.append(name)
        return secrets[name]

    context = initialize_colab(
        enable_wandb=False,
        artifact_root=tmp_path,
        secret_getter=get_secret,
        drive_mounter=lambda: None,
        hf_authenticator=lambda **_: None,
        wandb_authenticator=lambda **_: (_ for _ in ()).throw(
            AssertionError("W&B authentication must be skipped")
        ),
        device_selector=lambda **_: torch.device("cuda"),
    )

    assert requested == ["HF_READ_TOKEN"]
    assert context.wandb_enabled is False


def test_required_secret_error_does_not_include_secret_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="HF_READ_TOKEN is unavailable") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=lambda _: (_ for _ in ()).throw(KeyError("private")),
            drive_mounter=lambda: None,
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "private" not in str(error.value)
    assert error.value.__cause__ is None


def test_drive_mount_failure_is_fatal_and_redacted(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Google Drive mount failed") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=lambda _: "unused",
            drive_mounter=lambda: (_ for _ in ()).throw(
                RuntimeError("sensitive mount detail")
            ),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "sensitive mount detail" not in str(error.value)
    assert error.value.__cause__ is None


def test_huggingface_failure_is_fatal_and_redacted(tmp_path: Path) -> None:
    with pytest.raises(
        RuntimeError, match="Hugging Face authentication failed"
    ) as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=lambda _: "hf-sensitive-value",
            drive_mounter=lambda: None,
            hf_authenticator=lambda **_: (_ for _ in ()).throw(
                RuntimeError("hf-sensitive-value")
            ),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "hf-sensitive-value" not in str(error.value)
    assert error.value.__cause__ is None
