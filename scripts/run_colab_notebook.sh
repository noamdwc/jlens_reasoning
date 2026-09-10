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

Unattended Drive access uses a Google service-account JSON:
  \$JLENS_DRIVE_SA_JSON  or  ~/.config/jlens/drive-sa.json
Share a Drive folder named jlens-colab-root (containing jlens-reasoning/ and
data/jlens-reasoning/) with jlens-colab@j-lens-reasoning.iam.gserviceaccount.com,
or set JLENS_DRIVE_ROOT_FOLDER_ID.

Options:
  --gpu TYPE       Colab accelerator (default: L4)
  --session NAME   Session name (default: derived from notebook)
  --timeout SEC    Per-cell execution timeout (default: 7200)
  --keep           Keep the Colab session after the run
  --allow-interactive-drivemount
                   Fall back to interactive colab drivemount when no SA JSON
                   is configured (default: fail with a clear error)
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

resolve_sa_json() {
    if [ -n "${JLENS_DRIVE_SA_JSON:-}" ]; then
        printf '%s\n' "$JLENS_DRIVE_SA_JSON"
        return 0
    fi
    printf '%s\n' "${HOME}/.config/jlens/drive-sa.json"
}

sa_json=$(resolve_sa_json)
drive_helper="$repository/src/jlens_reasoning/environments/colab_drive.py"
credential_staging=
use_service_account=0

if [ -f "$sa_json" ]; then
    use_service_account=1
elif [ "$allow_interactive_drivemount" -eq 1 ] || \
    [ "${JLENS_COLAB_ALLOW_INTERACTIVE_DRIVEMOUNT:-0}" = "1" ]; then
    use_service_account=0
else
    printf 'error: unattended Colab CLI Drive access requires a service-account JSON\n' >&2
    printf 'error: set JLENS_DRIVE_SA_JSON or create ~/.config/jlens/drive-sa.json\n' >&2
    printf 'error: for interactive browser OAuth instead, pass --allow-interactive-drivemount\n' >&2
    exit 1
fi

cleanup() {
    status=$?
    trap - EXIT

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

if [ "$use_service_account" -eq 1 ]; then
    if [ ! -f "$drive_helper" ]; then
        printf 'error: missing Drive helper: %s\n' "$drive_helper" >&2
        exit 1
    fi
    credential_staging=$(mktemp -d "${TMPDIR:-/tmp}/jlens-colab-creds.XXXXXX")
    cp "$sa_json" "$credential_staging/drive-sa.json"
    cp "$drive_helper" "$credential_staging/colab_drive.py"

    env_file="$credential_staging/jlens.env"
    : > "$env_file"
    if [ -n "${WANDB_API_KEY:-}" ]; then
        printf 'WANDB_API_KEY=%s\n' "$WANDB_API_KEY" >> "$env_file"
    fi
    if [ -n "${JLENS_DRIVE_ROOT_FOLDER_ID:-}" ]; then
        printf 'JLENS_DRIVE_ROOT_FOLDER_ID=%s\n' \
            "$JLENS_DRIVE_ROOT_FOLDER_ID" >> "$env_file"
    fi
    if [ -n "${JLENS_DRIVE_ROOT_FOLDER_NAME:-}" ]; then
        printf 'JLENS_DRIVE_ROOT_FOLDER_NAME=%s\n' \
            "$JLENS_DRIVE_ROOT_FOLDER_NAME" >> "$env_file"
    fi

    printf 'from pathlib import Path\nPath("/content/jlens-credentials").mkdir(parents=True, exist_ok=True)\n' \
        | colab exec -s "$session"

    colab upload -s "$session" \
        "$credential_staging/drive-sa.json" \
        /content/jlens-credentials/drive-sa.json
    colab upload -s "$session" \
        "$credential_staging/colab_drive.py" \
        /content/jlens-credentials/colab_drive.py
    if [ -s "$env_file" ]; then
        colab upload -s "$session" \
            "$env_file" \
            /content/jlens-credentials/jlens.env
    fi
    printf 'Uploaded service-account Drive credentials for unattended mount\n'
else
    colab drivemount -s "$session"
fi

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
