# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
"""Build the project wheel and upload it to Google Drive."""

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from shutil import copy2, copytree

REPOSITORY = Path(__file__).resolve().parents[1]
DRIVE_ROOT = "data/jlens-reasoning"
Runner = Callable[..., subprocess.CompletedProcess[str]]


def _run(
    runner: Runner,
    command: list[str],
    *,
    error_message: str,
    show_output: bool = False,
) -> None:
    try:
        result = runner(
            command,
            check=False,
            text=True,
            capture_output=not show_output,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"{command[0]} is not installed") from error

    if result.returncode:
        detail = (result.stderr or "").strip()
        raise RuntimeError(f"{error_message}: {detail}" if detail else error_message)


def build_and_upload_wheel(
    *,
    remote: str = "jlens",
    runner: Runner = subprocess.run,
) -> str:
    """Build one wheel in temporary storage and upload it to Drive."""

    remote_root = f"{remote.removesuffix(':')}:{DRIVE_ROOT}"
    _run(
        runner,
        ["rclone", "lsd", remote_root],
        error_message=f"rclone remote is unavailable: {remote_root}",
    )

    with tempfile.TemporaryDirectory(prefix="jlens-wheel-") as temp:
        workspace = Path(temp)
        source = workspace / "source"
        build_directory = workspace / "dist"
        source.mkdir()
        copy2(REPOSITORY / "pyproject.toml", source)
        copy2(REPOSITORY / "README.md", source)
        copytree(REPOSITORY / "src", source / "src")

        _run(
            runner,
            [
                "uv",
                "build",
                "--wheel",
                "--out-dir",
                str(build_directory),
                str(source),
            ],
            error_message="wheel build failed",
        )

        wheels = list(build_directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError("wheel build must produce exactly one wheel")

        wheel = wheels[0]
        remote_wheel = f"{remote_root}/wheels/{wheel.name}"
        _run(
            runner,
            [
                "rclone",
                "copyto",
                str(wheel),
                remote_wheel,
                "--ignore-times",
                "--progress",
            ],
            error_message="wheel upload failed",
            show_output=True,
        )

    return remote_wheel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--remote",
        default="jlens",
        help="name of the configured rclone Google Drive remote",
    )
    args = parser.parse_args()

    try:
        remote_wheel = build_and_upload_wheel(remote=args.remote)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"uploaded {remote_wheel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
