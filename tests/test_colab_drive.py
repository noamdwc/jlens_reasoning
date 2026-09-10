from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from jlens_reasoning.environments import colab_drive


def test_drive_layout_ready_requires_both_project_folders(tmp_path: Path) -> None:
    mydrive = tmp_path / "MyDrive"
    assert colab_drive.drive_layout_ready(mydrive) is False
    (mydrive / "jlens-reasoning").mkdir(parents=True)
    assert colab_drive.drive_layout_ready(mydrive) is False
    (mydrive / "data" / "jlens-reasoning").mkdir(parents=True)
    assert colab_drive.drive_layout_ready(mydrive) is True


def test_load_and_apply_env_files(tmp_path: Path) -> None:
    env_file = tmp_path / "jlens.env"
    env_file.write_text(
        "WANDB_API_KEY=from-file\nJLENS_DRIVE_ROOT_FOLDER_ID=abc\n",
        encoding="utf-8",
    )
    environ = {"WANDB_API_KEY": ""}
    loaded = colab_drive.apply_env_files(paths=(env_file,), environ=environ)
    assert loaded["WANDB_API_KEY"] == "from-file"
    assert environ["JLENS_DRIVE_ROOT_FOLDER_ID"] == "abc"


def test_resolve_wandb_api_key_prefers_env_over_secrets() -> None:
    requested: list[str] = []

    def get_secret(name: str) -> str:
        requested.append(name)
        raise AssertionError("secrets should not be consulted")

    key = colab_drive.resolve_wandb_api_key(
        environ={"WANDB_API_KEY": "env-key"},
        secret_getter=get_secret,
    )
    assert key == "env-key"
    assert requested == []


def test_resolve_wandb_api_key_falls_back_to_secrets() -> None:
    key = colab_drive.resolve_wandb_api_key(
        environ={},
        secret_getter=lambda name: "secret-key" if name == "WANDB_API_KEY" else "",
    )
    assert key == "secret-key"


def test_ensure_colab_drive_uses_service_account_when_json_present(
    tmp_path: Path,
) -> None:
    sa_json = tmp_path / "drive-sa.json"
    sa_json.write_text("{}", encoding="utf-8")
    mydrive = tmp_path / "MyDrive"
    events: list[str] = []

    def sa_mounter(path: Path, **_: object) -> None:
        assert path == sa_json
        events.append("sa")
        (mydrive / "jlens-reasoning").mkdir(parents=True)
        (mydrive / "data" / "jlens-reasoning").mkdir(parents=True)

    mode = colab_drive.ensure_colab_drive(
        sa_json=sa_json,
        mydrive_root=mydrive,
        sa_mounter=sa_mounter,
        interactive_mounter=lambda: events.append("interactive"),
    )
    assert mode == "service_account"
    assert events == ["sa"]


def test_ensure_colab_drive_falls_back_to_interactive_mount(tmp_path: Path) -> None:
    mydrive = tmp_path / "MyDrive"
    events: list[str] = []

    def interactive() -> None:
        events.append("interactive")
        (mydrive / "jlens-reasoning").mkdir(parents=True)
        (mydrive / "data" / "jlens-reasoning").mkdir(parents=True)

    mode = colab_drive.ensure_colab_drive(
        sa_json=tmp_path / "missing.json",
        mydrive_root=mydrive,
        interactive_mounter=interactive,
        sa_mounter=lambda *_args, **_kwargs: events.append("sa"),
    )
    assert mode == "interactive"
    assert events == ["interactive"]


def test_ensure_colab_drive_is_idempotent_when_layout_exists(tmp_path: Path) -> None:
    mydrive = tmp_path / "MyDrive"
    (mydrive / "jlens-reasoning").mkdir(parents=True)
    (mydrive / "data" / "jlens-reasoning").mkdir(parents=True)
    events: list[str] = []

    mode = colab_drive.ensure_colab_drive(
        sa_json=tmp_path / "missing.json",
        mydrive_root=mydrive,
        interactive_mounter=lambda: events.append("interactive"),
    )
    assert mode == "interactive"
    assert events == []


def test_resolve_root_folder_id_from_env(tmp_path: Path) -> None:
    sa_json = tmp_path / "drive-sa.json"
    sa_json.write_text("{}", encoding="utf-8")

    folder_id = colab_drive.resolve_root_folder_id(
        sa_json,
        rclone="/usr/bin/rclone",
        runner=lambda _cmd: (_ for _ in ()).throw(AssertionError("rclone")),
        environ={colab_drive.ROOT_FOLDER_ID_ENV: "root-from-env"},
    )
    assert folder_id == "root-from-env"


def test_resolve_root_folder_id_from_rclone_listing(tmp_path: Path) -> None:
    sa_json = tmp_path / "drive-sa.json"
    sa_json.write_text("{}", encoding="utf-8")
    payload = [
        {
            "Name": "jlens-colab-root",
            "ID": "folder-xyz",
            "IsDir": True,
            "MimeType": "application/vnd.google-apps.folder",
        }
    ]

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        assert command[0] == "rclone"
        assert command[1] == "lsjson"
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(payload), stderr=""
        )

    folder_id = colab_drive.resolve_root_folder_id(
        sa_json,
        rclone="rclone",
        runner=runner,
        environ={},
    )
    assert folder_id == "folder-xyz"


def test_mount_drive_with_service_account_invokes_rclone_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sa_json = tmp_path / "drive-sa.json"
    sa_json.write_text("{}", encoding="utf-8")
    mydrive = tmp_path / "MyDrive"
    commands: list[list[str]] = []
    fake_rclone = tmp_path / "bin" / "rclone"
    fake_rclone.parent.mkdir()
    fake_rclone.write_text("#!/bin/bash\n", encoding="utf-8")
    fake_rclone.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_rclone.parent}:{tmp_path}")
    monkeypatch.setattr(
        colab_drive.shutil,
        "which",
        lambda name: str(fake_rclone) if name == "rclone" else None,
    )

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        if len(command) > 1 and command[1] == "mount":
            (mydrive / "jlens-reasoning").mkdir(parents=True, exist_ok=True)
            (mydrive / "data" / "jlens-reasoning").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    colab_drive.mount_drive_with_service_account(
        sa_json,
        mydrive_root=mydrive,
        runner=runner,
        environ={colab_drive.ROOT_FOLDER_ID_ENV: "root-123"},
        sleep=lambda _seconds: None,
    )

    assert commands
    mount = commands[-1]
    assert Path(mount[0]).name == "rclone"
    assert mount[1] == "mount"
    assert "root_folder_id=root-123" in mount[2]
    assert mount[3] == str(mydrive)
    assert "--daemon" in mount


def test_service_account_mount_failure_is_redacted(tmp_path: Path) -> None:
    sa_json = tmp_path / "drive-sa.json"
    sa_json.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="service-account mount failed") as error:
        colab_drive.ensure_colab_drive(
            sa_json=sa_json,
            mydrive_root=tmp_path / "MyDrive",
            sa_mounter=lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("sensitive rclone detail")
            ),
        )

    assert "sensitive rclone detail" not in str(error.value)
    assert error.value.__cause__ is None
