#!/usr/bin/env bash
set -euo pipefail

if [[ "${MODEL_DOWNLOAD_AUTHORIZED:-0}" != "1" ]]; then
  echo "Set MODEL_DOWNLOAD_AUTHORIZED=1 only after the model download and GPU smoke are approved" >&2
  exit 2
fi

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
model_id=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
model_revision=${MODEL_REVISION:-989aa7980e4cf806f80c7fef2b1adb7bc71aa306}
cache_dir=${HF_HOME:-/root/autodl-tmp/cache/huggingface}
manifest_root=${VERL_AGENT_MODEL_MANIFEST_ROOT:-/root/autodl-tmp/config/models}
model_tag=${MODEL_TAG:-${model_id##*/}}
model_tag=${model_tag//[^[:alnum:]_.-]/_}
manifest_path=${manifest_root}/${model_tag}-${model_revision}.json
smoke_script=${VERL_AGENT_MODEL_SMOKE:-/root/autodl-tmp/tools/model_smoke.py}

if [[ ! -f "${smoke_script}" ]]; then
  echo "model smoke script is missing: ${smoke_script}" >&2
  exit 1
fi

export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
cd "${repo_root}"
python3 -m recipe.exact.prepare_model \
  --model "${model_id}" \
  --revision "${model_revision}" \
  --cache-dir "${cache_dir}" \
  --output-manifest "${manifest_path}"

resolved_snapshot=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_snapshot"])' "${manifest_path}")
python3 "${smoke_script}" --backend transformers --model "${resolved_snapshot}"
python3 "${smoke_script}" --backend vllm --model "${resolved_snapshot}"
printf 'model_manifest=%s\nresolved_snapshot=%s\n' "${manifest_path}" "${resolved_snapshot}"
