#!/bin/bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd -P)
upload_script="$repository/scripts/upload_colab_wheel.sh"
notebook_runner="$repository/scripts/run_colab_notebook.sh"
experiment=
upload_arguments=()
runner_arguments=()

usage() {
    cat <<EOF
usage: $(basename "$0") [OPTIONS] EXPERIMENT

Options:
  --remote NAME    rclone remote used for the wheel upload
  --gpu TYPE       Colab accelerator (default: L4)
  --session NAME   Colab session name
  --timeout SEC    Per-cell execution timeout
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
        --remote)
            if [ "$#" -lt 2 ]; then
                printf 'error: --remote requires a value\n' >&2
                exit 2
            fi
            upload_arguments+=("$1" "$2")
            shift 2
            ;;
        --gpu|--session|--timeout)
            if [ "$#" -lt 2 ]; then
                printf 'error: %s requires a value\n' "$1" >&2
                exit 2
            fi
            runner_arguments+=("$1" "$2")
            shift 2
            ;;
        --keep)
            runner_arguments+=("$1")
            shift
            ;;
        -*)
            printf 'error: unknown option: %s\n' "$1" >&2
            exit 2
            ;;
        *)
            if [ -n "$experiment" ]; then
                printf 'error: unexpected argument: %s\n' "$1" >&2
                exit 2
            fi
            experiment=$1
            shift
            ;;
    esac
done

if [ -z "$experiment" ]; then
    printf 'error: EXPERIMENT is required\n' >&2
    usage >&2
    exit 2
fi
if [[ ! "$experiment" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'error: invalid experiment name: %s\n' "$experiment" >&2
    exit 2
fi

notebook="$repository/experiments/$experiment/$experiment.ipynb"
if [ ! -f "$notebook" ]; then
    printf 'error: experiment notebook does not exist: %s\n' "$notebook" >&2
    exit 1
fi

printf 'Experiment: %s\n' "$experiment"
printf 'Notebook: %s\n' "$notebook"
printf '\nUploading the Colab wheel bundle...\n'
if [ "${#upload_arguments[@]}" -gt 0 ]; then
    "$upload_script" "${upload_arguments[@]}"
else
    "$upload_script"
fi

printf '\nRunning the experiment notebook in Colab...\n'
if [ "${#runner_arguments[@]}" -gt 0 ]; then
    "$notebook_runner" "${runner_arguments[@]}" "$notebook"
else
    "$notebook_runner" "$notebook"
fi
