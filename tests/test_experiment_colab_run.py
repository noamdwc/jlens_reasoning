import os
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "experiment_colab_run.sh"


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def run_orchestrator(
    tmp_path: Path,
    *arguments: str,
    append_experiment: bool = True,
    create_notebook: bool = True,
    experiment: str = "jlens_readout_sanity",
    run_status: int = 0,
    upload_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    assert SCRIPT.is_file(), f"missing script: {SCRIPT}"

    repository = tmp_path / "repository"
    scripts_directory = repository / "scripts"
    experiment_directory = repository / "experiments" / experiment
    scripts_directory.mkdir(parents=True)
    experiment_directory.mkdir(parents=True)
    notebook = experiment_directory / f"{experiment}.ipynb"
    if create_notebook:
        notebook.touch()

    wrapper = scripts_directory / SCRIPT.name
    wrapper.write_bytes(SCRIPT.read_bytes())
    wrapper.chmod(0o755)

    command_log = tmp_path / "commands.log"
    upload_script = """#!/bin/bash
set -eu
printf 'upload' >> "$COMMAND_LOG"
if [ "$#" -gt 0 ]; then
    printf '\\t%s' "$@" >> "$COMMAND_LOG"
fi
printf '\\n' >> "$COMMAND_LOG"
exit "$JLENS_UPLOAD_STATUS"
"""
    write_executable(
        scripts_directory / "upload_colab_wheel.sh",
        upload_script,
    )
    run_script = """#!/bin/bash
set -eu
printf 'run' >> "$COMMAND_LOG"
if [ "$#" -gt 0 ]; then
    printf '\\t%s' "$@" >> "$COMMAND_LOG"
fi
printf '\\n' >> "$COMMAND_LOG"
exit "$JLENS_RUN_STATUS"
"""
    write_executable(
        scripts_directory / "run_colab_notebook.sh",
        run_script,
    )

    environment = {
        **os.environ,
        "COMMAND_LOG": str(command_log),
        "JLENS_UPLOAD_STATUS": str(upload_status),
        "JLENS_RUN_STATUS": str(run_status),
    }

    result = subprocess.run(
        [
            "/bin/bash",
            str(wrapper),
            *arguments,
            *([experiment] if append_experiment else []),
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, command_log, notebook


def test_uploads_wheel_then_runs_experiment_notebook(tmp_path: Path) -> None:
    result, command_log, notebook = run_orchestrator(tmp_path)

    assert result.returncode == 0, result.stderr
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "upload",
        f"run\t{notebook}",
    ]


def test_forwards_upload_and_colab_options(tmp_path: Path) -> None:
    result, command_log, notebook = run_orchestrator(
        tmp_path,
        "--remote",
        "research",
        "--gpu",
        "T4",
        "--session",
        "custom-session",
        "--timeout",
        "300",
        "--keep",
        "--allow-dirty",
    )

    assert result.returncode == 0, result.stderr
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "upload\t--remote\tresearch\t--allow-dirty",
        (
            "run\t--gpu\tT4\t--session\tcustom-session"
            f"\t--timeout\t300\t--keep\t{notebook}"
        ),
    ]


def test_does_not_run_notebook_when_upload_fails(tmp_path: Path) -> None:
    result, command_log, _ = run_orchestrator(tmp_path, upload_status=7)

    assert result.returncode == 7
    assert command_log.read_text(encoding="utf-8").splitlines() == ["upload"]


def test_rejects_missing_experiment_notebook(tmp_path: Path) -> None:
    result, command_log, notebook = run_orchestrator(
        tmp_path,
        create_notebook=False,
    )

    assert result.returncode != 0
    assert f"experiment notebook does not exist: {notebook}" in result.stderr
    assert not command_log.exists()


def test_help_exits_without_running_commands(tmp_path: Path) -> None:
    result, command_log, _ = run_orchestrator(
        tmp_path,
        "--help",
        append_experiment=False,
    )

    assert result.returncode == 0
    assert "usage: experiment_colab_run.sh [OPTIONS] EXPERIMENT" in result.stdout
    assert "--remote NAME" in result.stdout
    assert "--gpu TYPE" in result.stdout
    assert "--keep" in result.stdout
    assert "--allow-dirty" in result.stdout
    assert not command_log.exists()


def test_rejects_unknown_option_before_running_commands(tmp_path: Path) -> None:
    result, command_log, _ = run_orchestrator(tmp_path, "--unknown")

    assert result.returncode == 2
    assert "unknown option: --unknown" in result.stderr
    assert not command_log.exists()
