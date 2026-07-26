import os
import re
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "upload_colab_wheel.sh"


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def fake_tools(tmp_path: Path) -> tuple[Path, Path]:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    command_log = tmp_path / "commands.log"

    write_executable(
        bin_directory / "rclone",
        """#!/bin/bash
set -eu
printf 'rclone' >> "$COMMAND_LOG"
printf '\\t%s' "$@" >> "$COMMAND_LOG"
printf '\\n' >> "$COMMAND_LOG"

if [ "$1" = "lsd" ] && [ "${RCLONE_PREFLIGHT_STATUS:-0}" -ne 0 ]; then
    printf 'missing remote\\n' >&2
    exit "$RCLONE_PREFLIGHT_STATUS"
fi

if [ "$1" = "copyto" ] && [ ! -f "$2" ]; then
    printf 'wheel does not exist\\n' >&2
    exit 1
fi
""",
    )
    write_executable(
        bin_directory / "uv",
        """#!/bin/bash
set -eu
printf 'uv' >> "$COMMAND_LOG"
printf '\\t%s' "$@" >> "$COMMAND_LOG"
printf '\\n' >> "$COMMAND_LOG"

output_directory=
while [ "$#" -gt 0 ]; do
    if [ "$1" = "--out-dir" ]; then
        output_directory=$2
        shift 2
        continue
    fi
    build_source=$1
    shift
done

for input in pyproject.toml README.md src; do
    [ -e "$build_source/$input" ]
done

printf '%s\\n' "$build_source" > "$BUILD_SOURCE_LOG"
printf '%s\\n' "$output_directory" > "$BUILD_DIRECTORY_LOG"
mkdir -p "$output_directory"
touch "$output_directory/jlens_reasoning-0.1.0-py3-none-any.whl"
""",
    )
    return bin_directory, command_log


def run_uploader(
    tmp_path: Path,
    *arguments: str,
    preflight_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    bin_directory, command_log = fake_tools(tmp_path)
    build_source_log = tmp_path / "build-source.log"
    build_directory_log = tmp_path / "build-directory.log"
    environment = {
        **os.environ,
        "PATH": f"{bin_directory}{os.pathsep}{os.environ['PATH']}",
        "COMMAND_LOG": str(command_log),
        "BUILD_SOURCE_LOG": str(build_source_log),
        "BUILD_DIRECTORY_LOG": str(build_directory_log),
        "RCLONE_PREFLIGHT_STATUS": str(preflight_status),
    }
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), *arguments],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, command_log, build_source_log, build_directory_log


def test_builds_uploads_and_cleans_the_wheel(tmp_path: Path) -> None:
    result, command_log, build_source_log, build_directory_log = run_uploader(
        tmp_path
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0] == "rclone\tlsd\tjlens:data/jlens-reasoning"
    assert commands[1].startswith("uv\tbuild\t--wheel\t--out-dir\t")
    assert commands[2].endswith(
        "\tjlens:data/jlens-reasoning/wheels/"
        "jlens_reasoning-0.1.0-py3-none-any.whl"
        "\t--ignore-times\t--progress"
    )
    assert (
        "uploaded jlens:data/jlens-reasoning/wheels/"
        "jlens_reasoning-0.1.0-py3-none-any.whl"
        in result.stdout
    )

    build_source = Path(build_source_log.read_text(encoding="utf-8").strip())
    build_directory = Path(
        build_directory_log.read_text(encoding="utf-8").strip()
    )
    assert build_source != REPOSITORY
    assert not build_source.exists()
    assert not build_directory.exists()


def test_reports_phase_progress_and_timing(tmp_path: Path) -> None:
    result, _, _, _ = run_uploader(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Plan: check Drive -> prepare source -> build wheel -> upload" in result.stdout
    assert "rclone displays live upload speed and ETA" in result.stdout
    assert "[#####---------------]  25% Drive access confirmed" in result.stdout
    assert "[##########----------]  50% Build source ready" in result.stdout
    assert "[###############-----]  75% Wheel ready:" in result.stdout
    assert "[####################] 100% Upload complete" in result.stdout
    assert re.search(r"Total time: \d+s", result.stdout)


def test_missing_remote_stops_before_building(tmp_path: Path) -> None:
    result, command_log, _, _ = run_uploader(tmp_path, preflight_status=1)

    assert result.returncode != 0
    assert "rclone remote is unavailable" in result.stderr
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "rclone\tlsd\tjlens:data/jlens-reasoning"
    ]


def test_custom_remote_is_normalized(tmp_path: Path) -> None:
    result, command_log, _, _ = run_uploader(
        tmp_path,
        "--remote",
        "research:",
    )

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0] == "rclone\tlsd\tresearch:data/jlens-reasoning"
    assert "uploaded research:data/jlens-reasoning/wheels/" in result.stdout


def test_help_exits_without_running_commands(tmp_path: Path) -> None:
    result, command_log, _, _ = run_uploader(tmp_path, "--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert not command_log.exists()
