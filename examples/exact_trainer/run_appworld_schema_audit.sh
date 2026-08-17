#!/usr/bin/env bash
set -euo pipefail

audit_path=${APPWORLD_SCHEMA_AUDIT_PATH:-/root/autodl-tmp/config/exact/appworld-schema-audit.json}

python3 -m recipe.exact.audit_appworld_schema --output "${audit_path}" "$@"
python3 -m recipe.exact.audit_appworld_schema --verify "${audit_path}"
printf 'appworld_schema_audit=%s\n' "${audit_path}"
