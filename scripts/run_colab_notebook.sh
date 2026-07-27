#!/bin/bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd -P)
gpu="L4"
keep=0
notebook=
session=
timeout=7200

usage() {
    cat <<EOF
usage: $(basename "$0") [OPTIONS] NOTEBOOK.ipynb

Options:
  --gpu TYPE       Colab accelerator (default: L4)
  --session NAME   Session name (default: derived from notebook)
  --timeout SEC    Per-cell execution timeout (default: 7200)
  --keep           Keep the Colab session after the run
  -h, --help       Show this help
EOF
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

cleanup() {
    status=$?
    trap - EXIT

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

colab drivemount -s "$session"
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
