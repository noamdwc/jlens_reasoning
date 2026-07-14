"""Clone an explicit project ref and install its locked Colab environment."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPOSITORY_URL = "https://github.com/noamdwc/jlens-reasoning.git"
DEFAULT_PROJECT_DIR = Path("/content/jlens-reasoning")
UV_VERSION = "0.11.28"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _git_auth_environment(token: str) -> dict[str, str]:
    encoded = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {encoded}",
        }
    )
    return environment


def _run(
    runner: Runner,
    command: list[str],
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            command,
            check=True,
            text=True,
            capture_output=True,
            **kwargs,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("Colab bootstrap command failed") from exc


def clone_repository(
    *,
    project_ref: str,
    github_token: str,
    project_dir: Path = DEFAULT_PROJECT_DIR,
    runner: Runner = subprocess.run,
) -> Path:
    """Clone exactly one branch, tag, or commit without exposing the token."""

    if not project_ref:
        raise RuntimeError("A non-empty project ref is required")
    if not github_token:
        raise RuntimeError("GITHUB_TOKEN_JLENS_REAS is unavailable")

    if project_dir.exists():
        shutil.rmtree(project_dir)
    project_dir.mkdir(parents=True)

    _run(runner, ["git", "init", str(project_dir)])
    _run(
        runner,
        ["git", "-C", str(project_dir), "remote", "add", "origin", REPOSITORY_URL],
    )
    _run(
        runner,
        [
            "git",
            "-C",
            str(project_dir),
            "fetch",
            "--depth",
            "1",
            "origin",
            project_ref,
        ],
        env=_git_auth_environment(github_token),
    )
    _run(
        runner,
        ["git", "-C", str(project_dir), "checkout", "--detach", "FETCH_HEAD"],
    )

    remote = _run(
        runner,
        ["git", "-C", str(project_dir), "remote", "get-url", "origin"],
    ).stdout.strip()
    if github_token in remote or remote != REPOSITORY_URL:
        raise RuntimeError("Repository remote URL failed security validation")

    return project_dir


def install_locked_environment(
    *,
    project_dir: Path,
    runner: Runner = subprocess.run,
    uv_bin: str | None = None,
) -> None:
    """Install the lockfile into Colab's active Python environment."""

    _run(
        runner,
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            f"uv=={UV_VERSION}",
        ],
    )
    uv_bin = uv_bin or shutil.which("uv") or "uv"
    requirements = Path("/tmp/jlens-requirements.txt")

    _run(
        runner,
        [
            uv_bin,
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
            str(requirements),
            "--project",
            str(project_dir),
        ],
    )
    _run(
        runner,
        [
            uv_bin,
            "pip",
            "install",
            "--system",
            "--requirement",
            str(requirements),
        ],
    )
    wheel_dir = project_dir / "dist"
    _run(
        runner,
        [
            uv_bin,
            "build",
            "--wheel",
            "--clear",
            "--out-dir",
            str(wheel_dir),
            str(project_dir),
        ],
    )
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Colab build must produce exactly one wheel")
    _run(
        runner,
        [
            uv_bin,
            "pip",
            "install",
            "--system",
            "--no-deps",
            str(wheels[0]),
        ],
    )


def bootstrap(
    *,
    project_ref: str,
    github_token: str,
    project_dir: Path = DEFAULT_PROJECT_DIR,
) -> Path:
    """Clone the requested source revision and install its locked environment."""

    checkout = clone_repository(
        project_ref=project_ref,
        github_token=github_token,
        project_dir=project_dir,
    )
    install_locked_environment(project_dir=checkout)
    return checkout
