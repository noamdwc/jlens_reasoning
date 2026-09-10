#!/bin/bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd -P)
gpu="L4"
keep=0
notebook=
session=
timeout=7200
allow_interactive_drivemount=0

usage() {
    cat <<USAGE
usage: $(basename "$0") [OPTIONS] NOTEBOOK.ipynb

Unattended My Drive access prefers the user OAuth remote [jlens] from:
  \$JLENS_RCLONE_CONFIG, \$RCLONE_CONFIG, or ~/.config/rclone/rclone.conf
Only this remote is uploaded; its OAuth scope determines VM access.
Secondary SA mode: \$JLENS_DRIVE_SA_JSON or ~/.config/jlens/drive-sa.json,
with JLENS_DRIVE_SHARED_DRIVE_ID and JLENS_DRIVE_ROOT_FOLDER_ID.
Override selection with JLENS_DRIVE_AUTH=rclone|service_account|interactive.

Options:
  --gpu TYPE       Colab accelerator (default: L4)
  --session NAME   Session name (default: derived from notebook)
  --timeout SEC    Per-cell execution timeout (default: 7200)
  --keep           Keep the Colab session after the run
  --allow-interactive-drivemount
                   Fall back to interactive colab drivemount when no credentials are
                   configured (default: fail with a clear error)
  -h, --help       Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --gpu)
            gpu=$2
            shift 2
            ;;
        --keep)
            keep=1
            shift
            ;;
        --session)
            session=$2
            shift 2
            ;;
        --timeout)
            timeout=$2
            shift 2
            ;;
        --allow-interactive-drivemount)
            allow_interactive_drivemount=1
            shift
            ;;
        -*)
            printf 'error: unknown option: %s\n' "$1" >&2
            exit 2
            ;;
        *)
            if [ -n "$notebook" ]; then
                printf 'error: unexpected argument: %s\n' "$1" >&2
                exit 2
            fi
            notebook=$1
            shift
            ;;
    esac
done

if [ -z "$notebook" ]; then
    printf 'error: NOTEBOOK.ipynb is required\n' >&2
    usage >&2
    exit 2
fi
if [ ! -f "$notebook" ]; then
    printf 'error: notebook does not exist: %s\n' "$notebook" >&2
    exit 1
fi
if [[ "$notebook" != *.ipynb ]]; then
    printf 'error: notebook must use the .ipynb extension: %s\n' \
        "$notebook" >&2
    exit 2
fi
if ! colab_executable=$(command -v colab); then
    printf 'error: colab CLI is not installed\n' >&2
    exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
    printf 'error: jq is required to validate notebook output\n' >&2
    exit 1
fi

if [ ! -f "${SSL_CERT_FILE:-}" ]; then
    colab_python=$(head -n 1 "$colab_executable")
    colab_python=${colab_python#\#!}
    certificate_file=
    colab_environment=$(dirname "$(dirname "$colab_python")")

    for candidate in \
        "$colab_environment"/lib/python*/site-packages/certifi/cacert.pem \
        /etc/ssl/cert.pem \
        /etc/ssl/certs/ca-certificates.crt
    do
        if [ -f "$candidate" ]; then
            certificate_file=$candidate
            break
        fi
    done
    if [ ! -f "$certificate_file" ]; then
        printf 'error: could not locate a CA certificate bundle\n' >&2
        exit 1
    fi
    export SSL_CERT_FILE="$certificate_file"
fi

notebook_name=$(basename "$notebook" .ipynb)
session=${session:-"jlens-${notebook_name//_/-}"}

drive_helper="$repository/src/jlens_reasoning/environments/colab_drive.py"
if ! command -v python3 >/dev/null 2>&1; then
    printf 'error: python3 is required to prepare Colab credentials\n' >&2
    exit 1
fi
if [ ! -f "$drive_helper" ]; then
    printf 'error: missing Drive helper: %s\n' "$drive_helper" >&2
    exit 1
fi
credential_staging=$(mktemp -d "${TMPDIR:-/tmp}/jlens-colab-creds.XXXXXX")
# Also clean up when validation or VM allocation fails.
trap 'rm -rf "$credential_staging"' EXIT
if [ "$allow_interactive_drivemount" -eq 1 ]; then
    export JLENS_COLAB_ALLOW_INTERACTIVE_DRIVEMOUNT=1
fi
auth_mode=$(python3 "$drive_helper" --stage-credentials "$credential_staging")
notebook_started=0

cleanup() {
    status=$?
    trap - EXIT

    if [ "$auth_mode" != "interactive" ] && [ "$notebook_started" -eq 1 ]; then
        printf 'Waiting for Drive uploads before teardown...\n'
        if ! flush_output=$(printf '%s\n' \
            'import runpy' \
            'helper = runpy.run_path("/content/jlens-credentials/colab_drive.py")' \
            'helper["flush_colab_drive"]()' \
            'print("JLENS_DRIVE_UPLOADS_COMPLETE")' \
            | colab exec -s "$session" --timeout 660) || \
            [[ "$flush_output" != *JLENS_DRIVE_UPLOADS_COMPLETE* ]]; then
            printf 'error: Drive uploads could not be confirmed; preserving the VM and cached artifacts\n' >&2
            keep=1
            if [ "$status" -eq 0 ]; then
                status=1
            fi
        fi
    fi

    if [ -n "$credential_staging" ] && [ -d "$credential_staging" ]; then
        rm -rf "$credential_staging"
    fi

    if [ "$keep" -eq 1 ]; then
        printf 'Keeping Colab session: %s\n' "$session"
    else
        if ! colab stop -s "$session"; then
            printf 'error: failed to stop Colab session: %s\n' "$session" >&2
            if [ "$status" -eq 0 ]; then
                status=1
            fi
        fi
    fi

    exit "$status"
}

colab new -s "$session" --gpu "$gpu"
trap cleanup EXIT

if [ "$auth_mode" != "interactive" ]; then
    cp "$drive_helper" "$credential_staging/colab_drive.py"
    printf '%s\n' \
        'from pathlib import Path' \
        'credentials = Path("/content/jlens-credentials")' \
        'credentials.mkdir(parents=True, exist_ok=True, mode=0o700)' \
        'credentials.chmod(0o700)' \
        | colab exec -s "$session"
    for credential_file in "$credential_staging"/*; do
        colab upload -s "$session" "$credential_file" \
            "/content/jlens-credentials/$(basename "$credential_file")"
    done
    printf '%s\n' \
        'from pathlib import Path' \
        'for path in Path("/content/jlens-credentials").iterdir():' \
        '    path.chmod(0o600)' \
        | colab exec -s "$session"
    if [ "$auth_mode" = "service_account" ]; then
        printf 'Uploaded service-account Drive credentials for unattended mount\n'
    else
        printf 'Uploaded user rclone jlens credentials for unattended mount\n'
    fi
else
    colab drivemount -s "$session"
fi

notebook_started=1
colab exec -s "$session" --timeout "$timeout" -f "$notebook"

cli_output_notebook="${notebook%.ipynb}_output.ipynb"
if [ ! -f "$cli_output_notebook" ]; then
    printf 'error: executed notebook was not created: %s\n' \
        "$cli_output_notebook" >&2
    exit 1
fi

output_directory=${JLENS_COLAB_OUTPUT_DIR:-"$repository/artifacts/colab"}
mkdir -p "$output_directory"
output_notebook="$output_directory/${session}_output.ipynb"
mv -f "$cli_output_notebook" "$output_notebook"

if ! jq -e \
    '[.cells[].outputs[]? | select(.output_type == "error")] | length == 0' \
    "$output_notebook" >/dev/null
then
    printf 'error: executed notebook contains cell errors: %s\n' \
        "$output_notebook" >&2
    exit 1
fi

printf 'Executed notebook: %s\n' "$output_notebook"
