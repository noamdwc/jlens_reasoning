from pathlib import Path

import pytest
import torch

from jlens_reasoning.environments.colab import (
    download_r2_inputs,
    initialize_colab,
    source_bundle_sha256,
)


def test_source_bundle_hash_tracks_project_code(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'example'\n")
    (tmp_path / "uv.lock").write_text("locked")
    module = tmp_path / "src" / "example.py"
    module.parent.mkdir()
    module.write_text("VALUE = 1\n")

    original = source_bundle_sha256(tmp_path)
    module.write_text("VALUE = 2\n")

    assert source_bundle_sha256(tmp_path) != original


def test_r2_download_selects_only_requested_inputs(tmp_path: Path) -> None:
    objects = {
        "jlens/runs/flenqa-full-run/model_outputs.parquet": b"answers",
        "jlens/runs/flenqa-full-run/topk/shard-1.parquet": b"large shard",
        "jlens/assets/models/qwen/config.json": b"model",
    }
    downloaded = []

    class Client:
        def get_paginator(self, operation):
            assert operation == "list_objects_v2"
            return self

        def paginate(self, *, Bucket, Prefix):
            assert Bucket == "research"
            return [
                {
                    "Contents": [
                        {"Key": key} for key in objects if key.startswith(Prefix)
                    ]
                }
            ]

        def download_file(self, bucket, key, target):
            assert bucket == "research"
            downloaded.append(key)
            Path(target).write_bytes(objects[key])

    download_r2_inputs(
        client=Client(),
        bucket="research",
        prefix="jlens",
        destination=tmp_path,
        paths=(
            "runs/flenqa-full-run/model_outputs.parquet",
            "assets/models/qwen/",
        ),
    )

    assert len(downloaded) == 2
    assert not (tmp_path / "runs/flenqa-full-run/topk").exists()
    assert (tmp_path / "runs/flenqa-full-run/model_outputs.parquet").read_bytes() == (
        b"answers"
    )
    assert (tmp_path / "assets/models/qwen/config.json").read_bytes() == b"model"

    with pytest.raises(ValueError, match="Invalid R2 input path"):
        download_r2_inputs(
            client=Client(),
            bucket="research",
            prefix="jlens",
            destination=tmp_path,
            paths=("../outside/",),
        )


def test_colab_initialization_authenticates_wandb(
    tmp_path: Path,
) -> None:
    events: list[object] = []
    secrets = {"WANDB_API_KEY": "wandb-secret"}

    context = initialize_colab(
        artifact_root=tmp_path,
        secret_getter=secrets.__getitem__,
        wandb_authenticator=lambda **kwargs: events.append(("wandb", kwargs)) or True,
        device_selector=lambda **_: torch.device("cuda"),
    )

    assert events == [
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
    secrets = {"WANDB_API_KEY": "bad-wandb-secret"}

    with pytest.raises(RuntimeError, match="W&B authentication failed") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=secrets.__getitem__,
            wandb_authenticator=lambda **_: (_ for _ in ()).throw(
                RuntimeError("W&B authentication failed")
            ),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "bad-wandb-secret" not in str(error.value)
    assert error.value.__cause__ is None


def test_wandb_can_be_explicitly_disabled(tmp_path: Path) -> None:
    requested: list[str] = []

    def get_secret(name: str) -> str:
        requested.append(name)
        raise AssertionError(f"Unexpected secret request: {name}")

    context = initialize_colab(
        enable_wandb=False,
        artifact_root=tmp_path,
        secret_getter=get_secret,
        wandb_authenticator=lambda **_: (_ for _ in ()).throw(
            AssertionError("W&B authentication must be skipped")
        ),
        device_selector=lambda **_: torch.device("cuda"),
    )

    assert requested == []
    assert context.wandb_enabled is False


def test_required_secret_error_does_not_include_secret_value(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="WANDB_API_KEY is unavailable") as error:
        initialize_colab(
            artifact_root=tmp_path,
            secret_getter=lambda _: (_ for _ in ()).throw(KeyError("private")),
            device_selector=lambda **_: torch.device("cuda"),
        )

    assert "private" not in str(error.value)
    assert error.value.__cause__ is None


def test_wandb_key_can_come_from_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WANDB_API_KEY", "env-wandb-key")
    events: list[object] = []

    def get_secret(name: str) -> str:
        raise AssertionError(f"secrets should not be used: {name}")

    context = initialize_colab(
        artifact_root=tmp_path,
        secret_getter=get_secret,
        wandb_authenticator=lambda **kwargs: events.append(("wandb", kwargs)) or True,
        device_selector=lambda **_: torch.device("cuda"),
    )

    assert events == [
        (
            "wandb",
            {
                "api_key": "env-wandb-key",
                "enabled": True,
            },
        ),
    ]
    assert context.wandb_enabled is True
