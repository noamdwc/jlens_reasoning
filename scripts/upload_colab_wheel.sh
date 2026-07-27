#!/bin/bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd -P)
drive_root="data/jlens-reasoning"
remote="jlens"

usage() {
    cat <<EOF
usage: $(basename "$0") [--remote NAME]
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
printf 'Uploading Colab wheel bundle to %s/wheels/\n' "$remote_root"
rclone lsd "$remote_root" >/dev/null

workspace=$(mktemp -d "${TMPDIR:-/tmp}/jlens-wheel.XXXXXX")
trap 'rm -rf "$workspace"' EXIT

source_directory="$workspace/source"
build_directory="$workspace/dist"
requirements_file="$workspace/requirements-colab.txt"
commit_file="$workspace/project-commit.txt"
mkdir -p "$source_directory"

git -C "$repository" rev-parse HEAD > "$commit_file"

printf 'Exporting locked requirements...\n'
uv export \
    --frozen \
    --no-dev \
    --prune torch \
    --prune numpy \
    --prune fsspec \
    --prune rich \
    --prune colorama \
    --no-emit-project \
    --no-hashes \
    --format requirements.txt \
    --output-file "$requirements_file" \
    --project "$repository"

cp "$repository/pyproject.toml" "$repository/README.md" "$source_directory/"
cp -R "$repository/src" "$repository/experiments" "$source_directory/"

printf 'Building wheel...\n'
uv build \
    --wheel \
    --out-dir "$build_directory" \
    "$source_directory"

wheels=("$build_directory"/*.whl)
if [ "${#wheels[@]}" -ne 1 ] || [ ! -f "${wheels[0]}" ]; then
    printf 'error: wheel build must produce exactly one wheel\n' >&2
    exit 1
fi

wheel=${wheels[0]}
remote_wheel="$remote_root/wheels/$(basename "$wheel")"
remote_requirements="$remote_root/wheels/requirements-colab.txt"
remote_commit="$remote_root/wheels/project-commit.txt"

upload() {
    local source_file=$1
    local destination=$2

    printf 'Uploading %s...\n' "$(basename "$source_file")"
    rclone copyto "$source_file" "$destination" --ignore-times --progress
}

upload "$requirements_file" "$remote_requirements"
upload "$wheel" "$remote_wheel"
upload "$commit_file" "$remote_commit"

printf 'uploaded %s\n' "$remote_requirements"
printf 'uploaded %s\n' "$remote_wheel"
printf 'uploaded %s\n' "$remote_commit"
