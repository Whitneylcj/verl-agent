#!/usr/bin/env bash
set -euo pipefail

shared_data_root=${VERL_AGENT_SHARED_DATA_ROOT:-/root/autodl-tmp/data}
export APPWORLD_ROOT=${APPWORLD_ROOT:-${shared_data_root}/appworld}
export APPWORLD_PORT_FILE=${APPWORLD_PORT_FILE:-/root/autodl-tmp/config/appworld_ports.ports}
probe_path=${APPWORLD_EXACT_G_PROBE_PATH:-/root/autodl-tmp/config/exact/appworld-real-probe.json}

python3 -m recipe.exact.probe_appworld_exact_g \
  --port-file "${APPWORLD_PORT_FILE}" \
  --output "${probe_path}" \
  "$@"
python3 -m recipe.exact.probe_appworld_exact_g --verify "${probe_path}"
printf 'appworld_exact_g_probe=%s\n' "${probe_path}"
