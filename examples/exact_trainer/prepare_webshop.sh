#!/usr/bin/env bash
set -euo pipefail

shared_data_root=${VERL_AGENT_SHARED_DATA_ROOT:-/root/autodl-tmp/data}
data_root=${WEBSHOP_DATA_ROOT:-${shared_data_root}/webshop/data}
search_root=${WEBSHOP_SEARCH_ROOT:-${shared_data_root}/webshop/search_engine}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

command -v gdown >/dev/null
command -v java >/dev/null
command -v curl >/dev/null
command -v sha256sum >/dev/null
python -c 'import faiss, flask, pyserini, spacy, thefuzz'
python -c 'import spacy; spacy.load("en_core_web_sm")'

mkdir -p "${data_root}" "${search_root}"
data_source=${WEBSHOP_DATA_SOURCE:-auto}
mirror_endpoint=${WEBSHOP_HF_ENDPOINT:-https://hf-mirror.com}
mirror_repo=YWZBrandon/webshop-data
mirror_commit=ce990fff5aee388db2706f07820c578ab68e0453

download_webshop_file() {
  local filename=$1
  local drive_id=$2
  local expected_sha256=$3
  local destination="${data_root}/${filename}"
  local temporary="${destination}.part"

  if [[ -f "${destination}" ]]; then
    printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status || {
      echo "Existing WebShop file failed its pinned SHA-256: ${destination}" >&2
      return 1
    }
    return 0
  fi
  rm -f "${temporary}"

  if [[ "${data_source}" != mirror ]] && timeout 120 gdown \
    "https://drive.google.com/uc?id=${drive_id}" -O "${temporary}"; then
    echo "Downloaded ${filename} from the official WebShop Google Drive link"
  elif [[ "${data_source}" != drive ]]; then
    rm -f "${temporary}"
    curl --fail --location --retry 3 --max-time 180 \
      "${mirror_endpoint}/datasets/${mirror_repo}/resolve/${mirror_commit}/${filename}" \
      --output "${temporary}"
    echo "Downloaded ${filename} from the pinned third-party WebShop mirror"
  else
    rm -f "${temporary}"
    echo "Official WebShop Google Drive download failed and mirror fallback is disabled" >&2
    return 1
  fi

  printf '%s  %s\n' "${expected_sha256}" "${temporary}" | sha256sum --check --status || {
    rm -f "${temporary}"
    echo "Downloaded WebShop file failed its pinned SHA-256: ${filename}" >&2
    return 1
  }
  mv "${temporary}" "${destination}"
}

download_webshop_file \
  items_shuffle_1000.json \
  1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib \
  30a4765c3a327af72d9a9a95a6b2486d516f0fa1d3ecd83681901ce82a21b269
download_webshop_file \
  items_ins_v2_1000.json \
  1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu \
  f88a36314a397b53b3d9c3fa5878e5f7b26d35019a51ec83fbedeca61a948f6f
download_webshop_file \
  items_human_ins.json \
  14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O \
  cf78667548a71786e1d9049c24b802e48e1084ad4bb021cae56ce1f6d96954a3

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
