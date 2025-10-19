#!/usr/bin/env bash
set -euo pipefail

cmd="$1"
shift

exec "/opt/conda/envs/ml-rl-cuda12/bin/${cmd}" "$@"
