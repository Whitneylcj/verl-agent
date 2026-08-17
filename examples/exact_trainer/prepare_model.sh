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
model_source=${MODEL_SOURCE:-huggingface}
source_revision=${MODEL_SOURCE_REVISION:-}
case "${model_source}" in
  modelscope)
    cache_dir=${MODELSCOPE_CACHE:-/root/autodl-tmp/cache/modelscope}
    source_revision=${source_revision:-master}
    ;;
  huggingface)
    cache_dir=${HF_HOME:-/root/autodl-tmp/cache/huggingface}
    source_revision=${source_revision:-${model_revision}}
    ;;
  *)
    echo "unsupported MODEL_SOURCE: ${model_source}" >&2
    exit 2
    ;;
esac
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
prepare_args=(
  --source "${model_source}"
  --source-revision "${source_revision}"
  --model "${model_id}"
  --revision "${model_revision}"
  --cache-dir "${cache_dir}"
  --output-manifest "${manifest_path}"
)
if [[ -n "${MODEL_WEIGHT_SHA256:-}" ]]; then
  prepare_args+=(--expected-weight-sha256 "model.safetensors=${MODEL_WEIGHT_SHA256}")
fi
if [[ -n "${MODEL_WEIGHT_HASHES:-}" ]]; then
  IFS=',' read -r -a weight_hashes <<< "${MODEL_WEIGHT_HASHES}"
  for weight_hash in "${weight_hashes[@]}"; do
    prepare_args+=(--expected-weight-sha256 "${weight_hash}")
  done
fi
python3 -m recipe.exact.prepare_model "${prepare_args[@]}"

resolved_snapshot=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["resolved_snapshot"])' "${manifest_path}")
python3 "${smoke_script}" --backend transformers --model "${resolved_snapshot}"
python3 "${smoke_script}" --backend vllm --model "${resolved_snapshot}"
printf 'model_manifest=%s\nresolved_snapshot=%s\n' "${manifest_path}" "${resolved_snapshot}"
