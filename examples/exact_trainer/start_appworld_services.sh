#!/usr/bin/env bash
set -euo pipefail

appworld_root=${APPWORLD_ROOT:-/root/autodl-tmp/data/appworld}
port_file=${APPWORLD_PORT_FILE:-/root/autodl-tmp/config/appworld_ports.ports}
log_root=${APPWORLD_LOG_ROOT:-/root/autodl-tmp/logs/appworld}
start_port=${APPWORLD_START_PORT:-8200}
server_count=${APPWORLD_SERVER_COUNT:-8}
session_prefix=${APPWORLD_SESSION_PREFIX:-exact-appworld}

appworld_bin=$(command -v appworld)
command -v screen >/dev/null
command -v curl >/dev/null
if [[ ! -f "${appworld_root}/data/version.txt" ]]; then
  echo "AppWorld data is missing under ${appworld_root}" >&2
  exit 2
fi
if (( server_count < 1 )); then
  echo "APPWORLD_SERVER_COUNT must be positive" >&2
  exit 2
fi

mkdir -p "$(dirname "${port_file}")" "${log_root}"
manifest_tmp="${port_file}.tmp.$$"
launched_sessions=()
cleanup_failed_start() {
  for session in "${launched_sessions[@]}"; do
    screen -S "${session}" -X quit >/dev/null 2>&1 || true
  done
  rm -f "${manifest_tmp}"
}
trap cleanup_failed_start ERR INT TERM

for (( offset=0; offset<server_count; offset++ )); do
  port=$(( start_port + offset ))
  session="${session_prefix}-${port}"
  if screen -ls | grep -Fq ".${session}"; then
    echo "AppWorld screen session already exists: ${session}" >&2
    exit 2
  fi
  if curl --silent --fail --max-time 1 "http://127.0.0.1:${port}/docs" >/dev/null 2>&1; then
    echo "Port ${port} is already serving HTTP" >&2
    exit 2
  fi
done

: >"${manifest_tmp}"
for (( offset=0; offset<server_count; offset++ )); do
  port=$(( start_port + offset ))
  session="${session_prefix}-${port}"
  screen -dmS "${session}" -L -Logfile "${log_root}/${session}.log" \
    bash -lc "export APPWORLD_ROOT='${appworld_root}' PYTHONWARNINGS=ignore; exec '${appworld_bin}' serve environment --no-show-usage --port '${port}' --root '${appworld_root}'"
  launched_sessions+=("${session}")
  printf '%s\n' "${port}" >>"${manifest_tmp}"
done

for port in $(<"${manifest_tmp}"); do
  ready=0
  for _ in $(seq 1 90); do
    if curl --silent --fail --max-time 1 "http://127.0.0.1:${port}/docs" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 1
  done
  if (( ready == 0 )); then
    echo "AppWorld service on port ${port} did not become ready" >&2
    cleanup_failed_start
    trap - ERR INT TERM
    exit 1
  fi
done

mv "${manifest_tmp}" "${port_file}"
trap - ERR INT TERM
echo "Started ${server_count} AppWorld services; manifest: ${port_file}"
