#!/usr/bin/env bash
set -euo pipefail

exec conda run --no-capture-output -n ml-rl-cuda12 "$@"
