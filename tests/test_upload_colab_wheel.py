import os
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
    (tmp_path / "uploaded").mkdir()

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
if [ "$1" = "copyto" ]; then
    cp "$2" "$UPLOAD_DIRECTORY/$(basename "$3")"
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

if [ "$1" = "export" ]; then
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "--output-file" ]; then
            requirements_file=$2
            break
        fi
        shift
    done
    printf 'transformers==5.5.0\\n' > "$requirements_file"
    exit 0
fi

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

for input in pyproject.toml README.md src experiments; do
    [ -e "$build_source/$input" ]
done

printf '%s\\n' "$build_source" > "$BUILD_SOURCE_LOG"
printf '%s\\n' "$output_directory" > "$BUILD_DIRECTORY_LOG"
mkdir -p "$output_directory"
touch "$output_directory/jlens_reasoning-0.1.0-py3-none-any.whl"
""",
    )
    return bin_directory, command_log


def create_repository(tmp_path: Path, *, dirty: bool) -> Path:
    repository = tmp_path / "repository"
    scripts_directory = repository / "scripts"
    scripts_directory.mkdir(parents=True)
    wrapper = scripts_directory / SCRIPT.name
    wrapper.write_bytes(SCRIPT.read_bytes())
    wrapper.chmod(0o755)

    (repository / "pyproject.toml").write_text(
        '[project]\nname = "jlens-reasoning"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )
    (repository / "README.md").write_text("# test\n", encoding="utf-8")
    (repository / "src").mkdir()
    (repository / "experiments").mkdir()

    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.name", "Test"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "commit", "-q", "-m", "initial"],
        check=True,
    )
    if dirty:
        (repository / "src" / "dirty.py").write_text("DIRTY = True\n", encoding="utf-8")
    return repository


def run_uploader(
    tmp_path: Path,
    *arguments: str,
    dirty: bool = False,
    preflight_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path, Path, Path]:
    repository = create_repository(tmp_path, dirty=dirty)
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
        "UPLOAD_DIRECTORY": str(tmp_path / "uploaded"),
    }
    result = subprocess.run(
        ["/bin/bash", str(repository / "scripts" / SCRIPT.name), *arguments],
        cwd=repository,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, command_log, build_source_log, build_directory_log


def test_exports_builds_uploads_and_cleans_the_colab_bundle(
    tmp_path: Path,
) -> None:
    result, command_log, build_source_log, build_directory_log = run_uploader(tmp_path)

    assert result.returncode == 0, result.stderr
    commands = command_log.read_text(encoding="utf-8").splitlines()
    assert commands[0] == "rclone\tlsd\tjlens:data/jlens-reasoning"
    export_command = commands[1].split("\t")
    assert export_command[:19] == [
        "uv",
        "export",
        "--frozen",
        "--no-dev",
        "--prune",
        "torch",
        "--prune",
        "numpy",
        "--prune",
        "fsspec",
        "--prune",
        "rich",
        "--prune",
        "colorama",
        "--no-emit-project",
        "--no-hashes",
        "--format",
        "requirements.txt",
        "--output-file",
    ]
    assert Path(export_command[19]).name == "requirements-colab.txt"
    repository = tmp_path / "repository"
    assert export_command[20:] == ["--project", str(repository)]
    assert commands[2].startswith("uv\tbuild\t--wheel\t--out-dir\t")
    assert commands[3].endswith(
        "\tjlens:data/jlens-reasoning/wheels/requirements-colab.txt"
        "\t--ignore-times\t--progress"
    )
    assert commands[4].endswith(
        "\tjlens:data/jlens-reasoning/wheels/"
        "jlens_reasoning-0.1.0-py3-none-any.whl"
        "\t--ignore-times\t--progress"
    )
    assert commands[5].endswith(
        "\tjlens:data/jlens-reasoning/wheels/project-dirty.txt"
        "\t--ignore-times\t--progress"
    )
    assert commands[6].endswith(
        "\tjlens:data/jlens-reasoning/wheels/project-commit.txt"
        "\t--ignore-times\t--progress"
    )
    assert (
        "uploaded jlens:data/jlens-reasoning/wheels/"
        "jlens_reasoning-0.1.0-py3-none-any.whl" in result.stdout
    )

    build_source = Path(build_source_log.read_text(encoding="utf-8").strip())
    build_directory = Path(build_directory_log.read_text(encoding="utf-8").strip())
    assert build_source != repository
    assert not build_source.exists()
    assert not build_directory.exists()
    assert (tmp_path / "uploaded" / "project-dirty.txt").read_text(
        encoding="utf-8"
    ) == "false\n"


def test_dirty_working_tree_stops_before_remote_preflight(tmp_path: Path) -> None:
    result, command_log, _, _ = run_uploader(tmp_path, dirty=True)

    assert result.returncode != 0
    assert "working tree has uncommitted changes" in result.stderr
    assert "--allow-dirty" in result.stderr
    assert not command_log.exists()


def test_allow_dirty_uploads_base_commit_and_dirty_marker(tmp_path: Path) -> None:
    result, _, _, _ = run_uploader(tmp_path, "--allow-dirty", dirty=True)

    assert result.returncode == 0, result.stderr
    assert "warning: uploading uncommitted code" in result.stderr
    repository = tmp_path / "repository"
    expected_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert (tmp_path / "uploaded" / "project-commit.txt").read_text(
        encoding="utf-8"
    ) == expected_commit
    assert (tmp_path / "uploaded" / "project-dirty.txt").read_text(
        encoding="utf-8"
    ) == "true\n"


def test_leaves_upload_progress_to_rclone(tmp_path: Path) -> None:
    result, _, _, _ = run_uploader(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Plan:" not in result.stdout
    assert "[#####" not in result.stdout
    assert "Total time:" not in result.stdout


def test_missing_remote_stops_before_building(tmp_path: Path) -> None:
    result, command_log, _, _ = run_uploader(tmp_path, preflight_status=1)

    assert result.returncode != 0
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
    assert "--allow-dirty" in result.stdout
    assert not command_log.exists()
