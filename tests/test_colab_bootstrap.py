import subprocess
from pathlib import Path
from typing import Any

from scripts.colab_bootstrap import clone_repository, install_locked_environment


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(
        self, command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess:
        self.calls.append((command, kwargs))
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
    assert commands[3] == [
        "uv",
        "pip",
        "install",
        "--system",
        "--no-deps",
        "--editable",
        str(tmp_path),
    ]
