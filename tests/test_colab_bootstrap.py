import subprocess
from pathlib import Path
from typing import Any

import pytest

from scripts.colab_bootstrap import clone_repository, install_locked_environment


class RecordingRunner:
    def __init__(
        self,
        wheel_names: tuple[str, ...] = ("jlens_reasoning-0.1.0-py3-none-any.whl",),
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.wheel_names = wheel_names

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
        self.calls.append((command, kwargs))
        if command[1:2] == ["build"]:
            output_dir = Path(command[command.index("--out-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            for wheel_name in self.wheel_names:
                (output_dir / wheel_name).touch()
        stdout = (
            "https://github.com/noamdwc/jlens-reasoning.git\n"
            if command[-2:] == ["get-url", "origin"]
            else ""
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_clone_uses_header_auth_without_putting_token_in_commands(
    tmp_path: Path,
) -> None:
    runner = RecordingRunner()
    token = "github-secret-token"
    project_dir = tmp_path / "checkout"

    clone_repository(
        project_ref="experiment-branch",
        github_token=token,
        project_dir=project_dir,
        runner=runner,
    )

    rendered_commands = "\n".join(" ".join(call[0]) for call in runner.calls)
    assert token not in rendered_commands
    assert "experiment-branch" in rendered_commands

    fetch_call = next(call for call in runner.calls if "fetch" in call[0])
    fetch_environment = fetch_call[1]["env"]
    assert fetch_environment["GIT_CONFIG_COUNT"] == "1"
    assert fetch_environment["GIT_CONFIG_KEY_0"].endswith(".extraheader")
    assert fetch_environment["GIT_CONFIG_VALUE_0"].startswith("AUTHORIZATION: basic ")


def test_locked_install_exports_experiment_dependencies(tmp_path: Path) -> None:
    runner = RecordingRunner()

    install_locked_environment(
        project_dir=tmp_path,
        runner=runner,
        uv_bin="uv",
    )

    commands = [call[0] for call in runner.calls]
    assert commands[0][-2:] == ["install", "uv==0.11.28"]
    assert commands[1] == [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--extra",
        "experiment",
        "--prune",
        "torch",
        "--prune",
        "numpy",
        "--no-emit-project",
        "--format",
        "requirements.txt",
        "--output-file",
        "/tmp/jlens-requirements.txt",
        "--project",
        str(tmp_path),
    ]
    assert commands[2] == [
        "uv",
        "pip",
        "install",
        "--system",
        "--requirement",
        "/tmp/jlens-requirements.txt",
    ]
    wheel = tmp_path / "dist" / "jlens_reasoning-0.1.0-py3-none-any.whl"
    assert commands[3] == [
        "uv",
        "build",
        "--wheel",
        "--clear",
        "--out-dir",
        str(tmp_path / "dist"),
        str(tmp_path),
    ]
    assert commands[4] == [
        "uv",
        "pip",
        "install",
        "--system",
        "--no-deps",
        str(wheel),
    ]


def test_locked_install_requires_exactly_one_wheel(tmp_path: Path) -> None:
    runner = RecordingRunner(wheel_names=())

    with pytest.raises(RuntimeError, match="exactly one wheel"):
        install_locked_environment(
            project_dir=tmp_path,
            runner=runner,
            uv_bin="uv",
        )
