#!/usr/bin/env bash
set -euo pipefail

environment_name=${1:?Usage: launch_managed.sh sokoban|alfworld|webshop|appworld [Hydra overrides...]}
shift

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
run_dir=$(RUN_DIR_ONLY=1 bash "${script_dir}/run_exact.sh" "${environment_name}" "$@")
run_basename=${run_dir##*/}
session_default=${run_basename//[^[:alnum:]_.-]/_}
session_name=${SESSION_NAME:-${session_default:0:80}}
log_root=${VERL_AGENT_LOG_ROOT:-/root/autodl-tmp/logs/verl-agent}
log_file=${log_root}/${run_basename}.log

if ! command -v screen >/dev/null 2>&1; then
  echo "GNU screen is required for managed remote pilots" >&2
  exit 1
fi
if screen -ls 2>/dev/null | grep -Fq ".${session_name}"; then
  echo "screen session already exists: ${session_name}" >&2
  exit 1
fi
if [[ -e "${run_dir}" && "${RESUME_RUN:-0}" != "1" ]]; then
  echo "run directory already exists: ${run_dir}" >&2
  exit 1
fi
if [[ -e "${log_file}" && "${RESUME_RUN:-0}" != "1" ]]; then
  echo "console log already exists: ${log_file}" >&2
  exit 1
fi

mkdir -p "${log_root}"
export EXACT_CONSOLE_LOG=${log_file}
screen -DmS "${session_name}" -L -Logfile "${log_file}" \
  bash "${script_dir}/run_exact.sh" "${environment_name}" "$@"

sleep 1
if ! screen -ls 2>/dev/null | grep -Fq ".${session_name}"; then
  echo "managed pilot exited during startup; inspect ${log_file}" >&2
  exit 1
fi

printf 'session=%s\nrun_dir=%s\nconsole_log=%s\n' \
  "${session_name}" "${run_dir}" "${log_file}"
