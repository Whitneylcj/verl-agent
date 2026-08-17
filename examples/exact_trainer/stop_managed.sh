#!/usr/bin/env bash
set -euo pipefail

session_name=${1:?Usage: stop_managed.sh SCREEN_SESSION}
if ! screen -ls 2>/dev/null | grep -Fq ".${session_name}"; then
  echo "screen session is not running: ${session_name}" >&2
  exit 1
fi

# Send the same interrupt as an attached terminal. This gives the trainer a
# chance to release Ray workers and write its final log instead of killing all
# Python processes on the host.
screen -S "${session_name}" -X stuff $'\003'
printf 'interrupt sent to %s; verify screen -ls and the run heartbeat\n' "${session_name}"
