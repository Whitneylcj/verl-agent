#!/usr/bin/env bash
set -euo pipefail

shared_data_root=${VERL_AGENT_SHARED_DATA_ROOT:-/root/autodl-tmp/data}
data_root=${WEBSHOP_DATA_ROOT:-${shared_data_root}/webshop/data}
search_root=${WEBSHOP_SEARCH_ROOT:-${shared_data_root}/webshop/search_engine}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

command -v gdown >/dev/null
command -v java >/dev/null
python -c 'import flask, pyserini, spacy, thefuzz'
python -c 'import spacy; spacy.load("en_core_web_sm")'

mkdir -p "${data_root}" "${search_root}"
if [[ ! -f "${data_root}/items_shuffle_1000.json" ]]; then
  gdown 'https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib' \
    -O "${data_root}/items_shuffle_1000.json"
fi
if [[ ! -f "${data_root}/items_ins_v2_1000.json" ]]; then
  gdown 'https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu' \
    -O "${data_root}/items_ins_v2_1000.json"
fi

export WEBSHOP_DATA_ROOT=${data_root}
export WEBSHOP_SEARCH_ROOT=${search_root}
force_args=()
if [[ "${WEBSHOP_FORCE_REBUILD:-0}" == "1" ]]; then
  force_args+=(--force)
fi
python "${repo_root}/recipe/exact/prepare_webshop.py" \
  --data-root "${data_root}" \
  --search-root "${search_root}" \
  "${force_args[@]}"
