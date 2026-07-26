#!/bin/bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd -P)
drive_root="data/jlens-reasoning"
remote="jlens"

usage() {
    printf 'usage: %s [--remote NAME]\n' "$(basename "$0")"
}

show_progress() {
    step=$1
    label=$2

    case "$step" in
        1) bar="#####---------------" ;;
        2) bar="##########----------" ;;
        3) bar="###############-----" ;;
        4) bar="####################" ;;
    esac

    printf '[%s] %3d%% %s (elapsed %ds)\n' \
        "$bar" \
        "$((step * 25))" \
        "$label" \
        "$((SECONDS - start_seconds))"
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
            remote=$2
            shift 2
            ;;
        *)
            printf 'error: unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

remote_root="${remote%:}:$drive_root"
start_seconds=$SECONDS

printf 'Colab wheel upload\n'
printf 'Repository: %s\n' "$repository"
printf 'Destination: %s/wheels/\n' "$remote_root"
printf 'Plan: check Drive -> prepare source -> build wheel -> upload\n'
printf 'Timing: build duration depends on the local cache; '
printf 'rclone displays live upload speed and ETA.\n'

printf '\nStep 1/4: Checking Google Drive access...\n'

if ! preflight_error=$(rclone lsd "$remote_root" 2>&1); then
    printf 'error: rclone remote is unavailable: %s\n' "$remote_root" >&2
    if [ -n "$preflight_error" ]; then
        printf '%s\n' "$preflight_error" >&2
    fi
    exit 1
fi
show_progress 1 "Drive access confirmed"

printf '\nStep 2/4: Preparing an isolated build source...\n'
workspace=$(mktemp -d "${TMPDIR:-/tmp}/jlens-wheel.XXXXXX")
trap 'rm -rf "$workspace"' EXIT

source_directory="$workspace/source"
build_directory="$workspace/dist"
mkdir -p "$source_directory"
cp "$repository/pyproject.toml" "$source_directory/"
cp "$repository/README.md" "$source_directory/"
cp -R "$repository/src" "$source_directory/src"
show_progress 2 "Build source ready"

printf '\nStep 3/4: Building the wheel with uv...\n'
if ! uv build \
    --wheel \
    --out-dir "$build_directory" \
    "$source_directory"; then
    printf 'error: wheel build failed\n' >&2
    exit 1
fi

wheels=("$build_directory"/*.whl)
if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
    printf 'error: wheel build must produce exactly one wheel\n' >&2
    exit 1
fi

wheel=${wheels[0]}
wheel_size=$(du -h "$wheel" | awk '{print $1}')
show_progress 3 "Wheel ready: $(basename "$wheel") ($wheel_size)"

remote_wheel="$remote_root/wheels/$(basename "$wheel")"
printf '\nStep 4/4: Uploading the wheel with rclone...\n'
printf 'Live transfer speed and ETA follow below.\n'
if ! rclone copyto \
    "$wheel" \
    "$remote_wheel" \
    --ignore-times \
    --progress; then
    printf 'error: wheel upload failed\n' >&2
    exit 1
fi

show_progress 4 "Upload complete"
printf 'uploaded %s\n' "$remote_wheel"
printf 'Total time: %ds\n' "$((SECONDS - start_seconds))"
