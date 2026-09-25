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


@pytest.mark.parametrize("root", ["", "bad,scope=drive", "a/b"])
def test_mount_rejects_missing_or_invalid_root(tmp_path, monkeypatch, root):
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    monkeypatch.setattr(colab_drive, "_ensure_rclone", lambda _: "rclone")
    with pytest.raises(RuntimeError, match="JLENS_DRIVE_ROOT_FOLDER_ID"):
        colab_drive.mount_drive_with_service_account(
            sa,
            mydrive_root=tmp_path / "drive",
            environ={
                "JLENS_DRIVE_ROOT_FOLDER_ID": root,
                "JLENS_DRIVE_SHARED_DRIVE_ID": "team",
            },
        )


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
        if command[1] == "deletefile" and "--drive-use-trash=false" in command:
            return subprocess.CompletedProcess(
                command, 1, "", "Content manager cannot permanently delete"
            )
        if command[1] == "cat":
            return subprocess.CompletedProcess(
                command, 0, stdout=command[2].rsplit(":", 1)[1], stderr=""
            )
        if len(command) > 1 and command[1] == "mount":
            (mydrive / "jlens-reasoning").mkdir(parents=True, exist_ok=True)
            (mydrive / "data" / "jlens-reasoning").mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    colab_drive.mount_drive_with_service_account(
        sa_json,
        mydrive_root=mydrive,
        runner=runner,
        environ={
            colab_drive.ROOT_FOLDER_ID_ENV: "root-123",
            colab_drive.SHARED_DRIVE_ID_ENV: "team-456",
        },
        marker=tmp_path / "mounted",
        sleep=lambda _seconds: None,
    )

    assert commands
    mount = commands[-1]
    assert Path(mount[0]).name == "rclone"
    assert mount[1] == "mount"
    assert "root_folder_id=root-123" in mount[2]
    assert mount[3] == str(mydrive)
    assert "--daemon" in mount
    assert "team_drive=team-456" in mount[2]
    assert mount[mount.index("--rc-addr") + 1] == "127.0.0.1:5572"
    assert (tmp_path / "mounted").is_file()
    assert [command[1] for command in commands] == [
        "copyto",
        "cat",
        "deletefile",
        "mount",
    ]


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


def test_service_account_requires_shared_drive_before_mount(tmp_path, monkeypatch):
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    monkeypatch.setattr(colab_drive, "_ensure_rclone", lambda _: "rclone")
    with pytest.raises(RuntimeError, match="JLENS_DRIVE_SHARED_DRIVE_ID"):
        colab_drive.mount_drive_with_service_account(
            sa,
            mydrive_root=tmp_path / "drive",
            environ={"JLENS_DRIVE_ROOT_FOLDER_ID": "folder"},
            runner=lambda cmd: subprocess.CompletedProcess(
                cmd, 1, "", "unexpected command"
            ),
        )


def test_remote_write_failure_prevents_mount(tmp_path, monkeypatch):
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    monkeypatch.setattr(colab_drive, "_ensure_rclone", lambda _: "rclone")
    commands = []

    def runner(cmd):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, "", "storageQuotaExceeded")

    with pytest.raises(RuntimeError, match="write probe"):
        colab_drive.mount_drive_with_service_account(
            sa,
            mydrive_root=tmp_path / "drive",
            environ={
                "JLENS_DRIVE_ROOT_FOLDER_ID": "folder",
                "JLENS_DRIVE_SHARED_DRIVE_ID": "team",
            },
            runner=runner,
        )
    assert not any(cmd[1] == "mount" for cmd in commands)


def vfs_stats(queued=0, uploading=0, errors=0):
    return json.dumps(
        {
            "diskCache": {
                "uploadsQueued": queued,
                "uploadsInProgress": uploading,
                "erroredFiles": errors,
                "outOfSpace": False,
            },
            "inUse": 0,
        }
    )


def test_flush_waits_for_queued_and_active_uploads():
    replies = iter([vfs_stats(queued=1), vfs_stats(uploading=1), vfs_stats()])
    sleeps = []
    colab_drive.wait_for_drive_uploads(
        runner=lambda cmd: subprocess.CompletedProcess(cmd, 0, next(replies), ""),
        sleep=sleeps.append,
    )
    assert len(sleeps) == 2


@pytest.mark.parametrize("payload", [vfs_stats(errors=1), "{}", '{"diskCache": {}}'])
def test_flush_does_not_treat_errors_or_missing_stats_as_success(payload):
    with pytest.raises(RuntimeError):
        colab_drive.wait_for_drive_uploads(
            runner=lambda cmd: subprocess.CompletedProcess(cmd, 0, payload, ""),
        )


def test_flush_times_out_with_pending_uploads():
    with pytest.raises(RuntimeError, match="Timed out"):
        colab_drive.wait_for_drive_uploads(
            runner=lambda cmd: subprocess.CompletedProcess(
                cmd, 0, vfs_stats(queued=1), ""
            ),
            timeout_seconds=0,
        )


def test_flush_accepts_idle_but_mounted_vfs():
    stats = json.loads(vfs_stats())
    stats["inUse"] = 1
    colab_drive.wait_for_drive_uploads(
        runner=lambda cmd: subprocess.CompletedProcess(cmd, 0, json.dumps(stats), ""),
        timeout_seconds=0,
    )
