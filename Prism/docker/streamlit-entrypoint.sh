#!/usr/bin/env bash
set -euo pipefail

cmd="$1"
shift || true

uid="${PRISM_UID:-0}"
gid="${PRISM_GID:-0}"

if [[ "$uid" != "0" ]]; then
  group_name="prismgrp"
  if getent group "$gid" >/dev/null 2>&1; then
    group_name="$(getent group "$gid" | cut -d: -f1)"
  else
    groupadd -g "$gid" "$group_name"
  fi

  user_name="prismusr"
  if id -u "$user_name" >/dev/null 2>&1; then
    usermod -u "$uid" -g "$group_name" "$user_name"
  else
    useradd -M -u "$uid" -g "$group_name" "$user_name"
  fi

  exec gosu "$uid:$gid" "/opt/conda/envs/ml-rl-cuda12/bin/${cmd}" "$@"
fi

exec "/opt/conda/envs/ml-rl-cuda12/bin/${cmd}" "$@"
