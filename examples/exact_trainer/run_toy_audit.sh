#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
audit_path=${EXACT_TOY_AUDIT_PATH:-/root/autodl-tmp/config/exact/toy-audit.json}

cd "${repo_root}"
python3 -m recipe.exact.toy_audit --output "${audit_path}"
python3 -m recipe.exact.toy_audit --verify "${audit_path}"
printf 'toy_audit=%s\n' "${audit_path}"
