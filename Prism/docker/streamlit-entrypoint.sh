#!/usr/bin/env bash
set -euo pipefail

cmd="$1"
shift || true

exec "/opt/conda/envs/ml-rl-cuda12/bin/${cmd}" "$@"
