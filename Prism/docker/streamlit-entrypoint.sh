#!/usr/bin/env bash
set -euo pipefail

cmd="$1"
shift || true

exec "/opt/conda/bin/${cmd}" "$@"
