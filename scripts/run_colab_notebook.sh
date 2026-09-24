#!/usr/bin/env bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/.." && pwd -P)
export R2_CREDENTIALS_FILE="${R2_CREDENTIALS_FILE:-$HOME/.config/colab-utils/r2.env}"

# The Colab CLI's Python has no default CA file on this macOS installation.
if [[ ! -f ${SSL_CERT_FILE:-} ]]; then
    colab_python=$(head -n 1 "$(command -v colab)")
    colab_python=${colab_python#\#!}
    export SSL_CERT_FILE=$("$colab_python" -c 'import certifi; print(certifi.where())')
fi

exec "$repository/../colab-utils/run.sh" "$@"
