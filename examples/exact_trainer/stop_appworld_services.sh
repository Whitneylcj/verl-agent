#!/usr/bin/env bash
set -euo pipefail

port_file=${APPWORLD_PORT_FILE:-/root/autodl-tmp/config/appworld_ports.ports}
session_prefix=${APPWORLD_SESSION_PREFIX:-exact-appworld}

if [[ ! -f "${port_file}" ]]; then
  echo "No AppWorld port manifest found: ${port_file}"
  exit 0
fi

while IFS= read -r port; do
  [[ "${port}" =~ ^[0-9]+$ ]] || continue
  screen -S "${session_prefix}-${port}" -X quit >/dev/null 2>&1 || true
done <"${port_file}"
rm -f "${port_file}"
echo "Stopped AppWorld services listed in ${port_file}"
