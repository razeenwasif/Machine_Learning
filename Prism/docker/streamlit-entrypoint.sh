#!/usr/bin/env bash
set -euo pipefail

exec /opt/conda/bin/conda run --no-capture-output -n ml-rl-cuda12 "$@"
