import json
import os
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "scripts" / "run_colab_notebook.sh"
DRIVE_HELPER = (
    REPOSITORY / "src" / "jlens_reasoning" / "environments" / "colab_drive.py"
)


def write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def run_runner(
    tmp_path: Path,
    *arguments: str,
    append_notebook: bool = True,
    create_notebook: bool = True,
    drivemount_status: int = 0,
    exec_status: int = 0,
    notebook_name: str = "example.ipynb",
    output_error: bool = False,
    invalid_ssl_cert: bool = False,
    stop_status: int = 0,
    supply_ssl_cert: bool = True,
    write_output: bool = True,
    provide_sa: bool = True,
    allow_interactive: bool = False,
    wandb_api_key: str | None = "wandb-from-env",
    root_folder_id: str | None = "folder-123",
) -> tuple[subprocess.CompletedProcess[str], Path, Path | None]:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    command_log = tmp_path / "commands.log"
    upload_log = tmp_path / "uploads.log"
    notebook = tmp_path / notebook_name
    if create_notebook:
        notebook.write_text(
            json.dumps(
                {
                    "cells": [],
                    "metadata": {},
                    "nbformat": 4,
                    "nbformat_minor": 5,
                }
            ),
            encoding="utf-8",
        )

    sa_json = tmp_path / "drive-sa.json"
    if provide_sa:
        sa_json.write_text(
            json.dumps({"type": "service_account", "client_email": "sa@example.com"}),
            encoding="utf-8",
        )

    write_executable(
        bin_directory / "colab",
        """#!/bin/bash
set -eu
command=$1
printf '%s' "$command" >> "$COMMAND_LOG"
shift
printf '\\t%s' "$@" >> "$COMMAND_LOG"
printf '\\n' >> "$COMMAND_LOG"
printf '%s\\n' "${SSL_CERT_FILE:-}" >> "$CERTIFICATE_LOG"

if [ "$command" = "upload" ]; then
    printf '%s\\t%s\\n' "$1" "$2" >> "$UPLOAD_LOG"
fi

if [ "$command" = "exec" ] && [ "${COLAB_WRITE_OUTPUT:-1}" -eq 1 ]; then
    notebook=
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "-f" ]; then
            notebook=$2
            break
        fi
        shift
    done
    if [ -n "$notebook" ]; then
        output_notebook="${notebook%.ipynb}_output.ipynb"
        if [ "${COLAB_OUTPUT_ERROR:-0}" -eq 1 ]; then
            printf '%s\\n' '{"cells":[{"id":"broken","outputs":[{"output_type":"error","ename":"RuntimeError","evalue":"boom"}]}]}' > "$output_notebook"
        else
            printf '%s\\n' '{"cells":[{"id":"ok","outputs":[]}]}' > "$output_notebook"
        fi
    fi
fi

case "$command" in
    drivemount) exit "${COLAB_DRIVEMOUNT_STATUS:-0}" ;;
    exec) exit "${COLAB_EXEC_STATUS:-0}" ;;
    stop) exit "${COLAB_STOP_STATUS:-0}" ;;
    upload) exit 0 ;;
esac
""",
    )
    certificate_file = tmp_path / "cacert.pem"
    certificate_file.touch()
    environment = {
        **os.environ,
        "PATH": f"{bin_directory}{os.pathsep}{os.environ['PATH']}",
        "COMMAND_LOG": str(command_log),
        "UPLOAD_LOG": str(upload_log),
        "CERTIFICATE_LOG": str(tmp_path / "certificates.log"),
        "COLAB_DRIVEMOUNT_STATUS": str(drivemount_status),
        "COLAB_EXEC_STATUS": str(exec_status),
        "COLAB_OUTPUT_ERROR": "1" if output_error else "0",
        "COLAB_STOP_STATUS": str(stop_status),
        "COLAB_WRITE_OUTPUT": "1" if write_output else "0",
        "JLENS_COLAB_OUTPUT_DIR": str(tmp_path / "artifacts"),
        "HOME": str(tmp_path / "home"),
    }
    (tmp_path / "home").mkdir()
    environment.pop("JLENS_DRIVE_SA_JSON", None)
    environment.pop("JLENS_COLAB_ALLOW_INTERACTIVE_DRIVEMOUNT", None)
    environment.pop("WANDB_API_KEY", None)
    environment.pop("JLENS_DRIVE_ROOT_FOLDER_ID", None)
    environment.pop("JLENS_DRIVE_ROOT_FOLDER_NAME", None)
    if provide_sa:
        environment["JLENS_DRIVE_SA_JSON"] = str(sa_json)
    if allow_interactive:
        environment["JLENS_COLAB_ALLOW_INTERACTIVE_DRIVEMOUNT"] = "1"
    if wandb_api_key is not None:
        environment["WANDB_API_KEY"] = wandb_api_key
    if root_folder_id is not None:
        environment["JLENS_DRIVE_ROOT_FOLDER_ID"] = root_folder_id
    if supply_ssl_cert:
        environment["SSL_CERT_FILE"] = str(
            tmp_path / "missing-cacert.pem" if invalid_ssl_cert else certificate_file
        )
    else:
        environment.pop("SSL_CERT_FILE", None)

    cli_arguments = list(arguments)
    if allow_interactive and "--allow-interactive-drivemount" not in cli_arguments:
        # env flag alone is enough; keep CLI optional
        pass

    result = subprocess.run(
        [
            "/bin/bash",
            str(SCRIPT),
            *cli_arguments,
            *([str(notebook)] if append_notebook else []),
        ],
        cwd=REPOSITORY,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, command_log, upload_log if upload_log.exists() else None


def test_runs_local_notebook_with_service_account_and_stops_session(
    tmp_path: Path,
) -> None:
    result, command_log, upload_log = run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    notebook = tmp_path / "example.ipynb"
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "new\t-s\tjlens-example\t--gpu\tL4"
    assert lines[1].startswith("exec\t-s\tjlens-example")
    assert "drivemount" not in "\n".join(lines)
    assert any(
        line.startswith("upload\t-s\tjlens-example\t")
        and line.endswith("/content/jlens-credentials/drive-sa.json")
        for line in lines
    )
    assert any(
        line.endswith("/content/jlens-credentials/colab_drive.py") for line in lines
    )
    assert any(line.endswith("/content/jlens-credentials/jlens.env") for line in lines)
    assert lines[-2] == (f"exec\t-s\tjlens-example\t--timeout\t7200\t-f\t{notebook}")
    assert lines[-1] == "stop\t-s\tjlens-example"
    assert "Uploaded service-account Drive credentials" in result.stdout
    assert upload_log is not None
    assert DRIVE_HELPER.is_file()
    output_notebook = tmp_path / "artifacts" / "jlens-example_output.ipynb"
    assert f"Executed notebook: {output_notebook}" in result.stdout
    assert output_notebook.is_file()
    assert not (tmp_path / "example_output.ipynb").exists()


def test_keep_preserves_session(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(tmp_path, "--keep")

    assert result.returncode == 0, result.stderr
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "new\t-s\tjlens-example\t--gpu\tL4"
    assert "drivemount" not in "\n".join(lines)
    assert lines[-1] == (
        f"exec\t-s\tjlens-example\t--timeout\t7200\t-f\t{tmp_path / 'example.ipynb'}"
    )
    assert "Keeping Colab session: jlens-example" in result.stdout


def test_accepts_gpu_session_and_timeout_overrides(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(
        tmp_path,
        "--gpu",
        "T4",
        "--session",
        "custom-run",
        "--timeout",
        "300",
    )

    assert result.returncode == 0, result.stderr
    lines = command_log.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "new\t-s\tcustom-run\t--gpu\tT4"
    assert lines[-2] == (
        f"exec\t-s\tcustom-run\t--timeout\t300\t-f\t{tmp_path / 'example.ipynb'}"
    )
    assert lines[-1] == "stop\t-s\tcustom-run"


def test_fails_when_executed_notebook_contains_a_cell_error(
    tmp_path: Path,
) -> None:
    result, command_log, _ = run_runner(tmp_path, output_error=True)

    assert result.returncode != 0
    assert "executed notebook contains cell errors" in result.stderr
    assert command_log.read_text(encoding="utf-8").splitlines()[-1] == (
        "stop\t-s\tjlens-example"
    )
    assert (tmp_path / "artifacts" / "jlens-example_output.ipynb").is_file()
    assert not (tmp_path / "example_output.ipynb").exists()


def test_uses_jq_instead_of_python3_for_output_validation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "command -v jq" in source
    assert "python3" not in source
    assert (
        '[.cells[].outputs[]? | select(.output_type == "error")] | length == 0'
        in source
    )


def test_help_exits_without_creating_a_session(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(
        tmp_path,
        "--help",
        append_notebook=False,
        provide_sa=False,
        wandb_api_key=None,
        root_folder_id=None,
    )

    assert result.returncode == 0
    assert "usage: run_colab_notebook.sh [OPTIONS] NOTEBOOK.ipynb" in result.stdout
    assert "--gpu TYPE" in result.stdout
    assert "--keep" in result.stdout
    assert "--allow-interactive-drivemount" in result.stdout
    assert "JLENS_DRIVE_SA_JSON" in result.stdout
    assert not command_log.exists()


def test_missing_notebook_stops_before_creating_a_session(
    tmp_path: Path,
) -> None:
    result, command_log, _ = run_runner(tmp_path, create_notebook=False)

    assert result.returncode != 0
    assert f"notebook does not exist: {tmp_path / 'example.ipynb'}" in (result.stderr)
    assert not command_log.exists()


def test_unknown_option_stops_before_creating_a_session(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(tmp_path, "--unknown")

    assert result.returncode == 2
    assert "unknown option: --unknown" in result.stderr
    assert not command_log.exists()


def test_configures_certifi_bundle_when_ssl_cert_is_unset(
    tmp_path: Path,
) -> None:
    result, _, _ = run_runner(tmp_path, supply_ssl_cert=False)

    assert result.returncode == 0, result.stderr
    certificate_paths = (
        (tmp_path / "certificates.log").read_text(encoding="utf-8").splitlines()
    )
    assert certificate_paths
    assert all(certificate_paths)
    assert all(Path(path).is_file() for path in certificate_paths)


def test_replaces_invalid_ssl_cert_file(tmp_path: Path) -> None:
    invalid_certificate = tmp_path / "missing-cacert.pem"

    result, _, _ = run_runner(tmp_path, invalid_ssl_cert=True)

    assert result.returncode == 0, result.stderr
    certificate_paths = (
        (tmp_path / "certificates.log").read_text(encoding="utf-8").splitlines()
    )
    assert certificate_paths
    assert all(Path(path).is_file() for path in certificate_paths)
    assert str(invalid_certificate) not in certificate_paths


def test_missing_sa_fails_before_creating_a_session(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(
        tmp_path,
        provide_sa=False,
        wandb_api_key=None,
        root_folder_id=None,
    )

    assert result.returncode != 0
    assert "unattended Colab CLI Drive access requires a service-account JSON" in (
        result.stderr
    )
    assert "JLENS_DRIVE_SA_JSON" in result.stderr
    assert not command_log.exists()


def test_interactive_drivemount_fallback(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(
        tmp_path,
        "--allow-interactive-drivemount",
        provide_sa=False,
        allow_interactive=True,
        wandb_api_key=None,
        root_folder_id=None,
    )

    assert result.returncode == 0, result.stderr
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "new\t-s\tjlens-example\t--gpu\tL4",
        "drivemount\t-s\tjlens-example",
        (f"exec\t-s\tjlens-example\t--timeout\t7200\t-f\t{tmp_path / 'example.ipynb'}"),
        "stop\t-s\tjlens-example",
    ]


def test_mount_failure_still_stops_session(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(
        tmp_path,
        "--allow-interactive-drivemount",
        provide_sa=False,
        allow_interactive=True,
        drivemount_status=7,
        wandb_api_key=None,
        root_folder_id=None,
    )

    assert result.returncode == 7
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "new\t-s\tjlens-example\t--gpu\tL4",
        "drivemount\t-s\tjlens-example",
        "stop\t-s\tjlens-example",
    ]


def test_keep_preserves_failed_session(tmp_path: Path) -> None:
    result, command_log, _ = run_runner(
        tmp_path,
        "--keep",
        "--allow-interactive-drivemount",
        provide_sa=False,
        allow_interactive=True,
        drivemount_status=7,
        wandb_api_key=None,
        root_folder_id=None,
    )

    assert result.returncode == 7
    assert command_log.read_text(encoding="utf-8").splitlines() == [
        "new\t-s\tjlens-example\t--gpu\tL4",
        "drivemount\t-s\tjlens-example",
    ]
    assert "Keeping Colab session: jlens-example" in result.stdout


def test_fails_clearly_when_cli_does_not_create_output_notebook(
    tmp_path: Path,
) -> None:
    result, command_log, _ = run_runner(tmp_path, write_output=False)

    expected_output = tmp_path / "example_output.ipynb"
    assert result.returncode != 0
    assert f"executed notebook was not created: {expected_output}" in (result.stderr)
    assert command_log.read_text(encoding="utf-8").splitlines()[-1] == (
        "stop\t-s\tjlens-example"
    )


def test_rejects_non_notebook_input_before_creating_a_session(
    tmp_path: Path,
) -> None:
    result, command_log, _ = run_runner(
        tmp_path,
        notebook_name="example.py",
    )

    assert result.returncode == 2
    assert "notebook must use the .ipynb extension" in result.stderr
    assert not command_log.exists()
