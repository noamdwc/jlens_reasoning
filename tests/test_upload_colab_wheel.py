import subprocess
from pathlib import Path

import pytest

from scripts.upload_colab_wheel import build_and_upload_wheel


class RecordingRunner:
    def __init__(self, *, preflight_returncode: int = 0) -> None:
        self.preflight_returncode = preflight_returncode
        self.calls: list[list[str]] = []
        self.build_directory: Path | None = None
        self.build_source: Path | None = None
        self.source_had_build_inputs = False
        self.wheel_existed_during_upload = False

    def __call__(self, command, **kwargs):
        self.calls.append(command)

        if command[:2] == ["rclone", "lsd"]:
            return subprocess.CompletedProcess(
                command,
                self.preflight_returncode,
                stdout="",
                stderr="missing remote",
            )

        if command[:3] == ["uv", "build", "--wheel"]:
            output_directory = Path(command[command.index("--out-dir") + 1])
            output_directory.mkdir(parents=True, exist_ok=True)
            self.build_directory = output_directory
            self.build_source = Path(command[-1])
            self.source_had_build_inputs = all(
                (self.build_source / path).exists()
                for path in ("pyproject.toml", "README.md", "src")
            )
            (output_directory / "jlens_reasoning-0.1.0-py3-none-any.whl").touch()

        if command[:2] == ["rclone", "copyto"]:
            self.wheel_existed_during_upload = Path(command[2]).is_file()

        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


def test_builds_uploads_and_cleans_the_wheel() -> None:
    runner = RecordingRunner()

    remote_wheel = build_and_upload_wheel(runner=runner)

    assert runner.calls[0] == [
        "rclone",
        "lsd",
        "jlens:data/jlens-reasoning",
    ]
    assert runner.calls[1][:4] == ["uv", "build", "--wheel", "--out-dir"]
    assert runner.calls[2][0:2] == ["rclone", "copyto"]
    assert runner.calls[2][3] == (
        "jlens:data/jlens-reasoning/wheels/jlens_reasoning-0.1.0-py3-none-any.whl"
    )
    assert runner.calls[2][-2:] == ["--ignore-times", "--progress"]
    assert remote_wheel == runner.calls[2][3]
    assert runner.source_had_build_inputs
    assert runner.build_source is not None
    assert runner.build_source != Path.cwd()
    assert runner.wheel_existed_during_upload
    assert runner.build_directory is not None
    assert not runner.build_source.exists()
    assert not runner.build_directory.exists()


def test_missing_remote_stops_before_building() -> None:
    runner = RecordingRunner(preflight_returncode=1)

    with pytest.raises(RuntimeError, match="rclone remote"):
        build_and_upload_wheel(runner=runner)

    assert len(runner.calls) == 1
