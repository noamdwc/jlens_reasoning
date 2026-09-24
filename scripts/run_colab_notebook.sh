#!/usr/bin/env bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd -P)
export R2_CREDENTIALS_FILE="${R2_CREDENTIALS_FILE:-$HOME/.config/colab-utils/r2.env}"

exec "$repository/../colab-utils/run.sh" "$@"
