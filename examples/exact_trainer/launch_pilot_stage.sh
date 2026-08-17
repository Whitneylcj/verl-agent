#!/usr/bin/env bash
set -euo pipefail

stage=${1:?Usage: launch_pilot_stage.sh repository_grpo|matched_grpo|exact ENVIRONMENT [Hydra overrides...]}
environment_name=${2:?Usage: launch_pilot_stage.sh repository_grpo|matched_grpo|exact ENVIRONMENT [Hydra overrides...]}
shift 2

if [[ "${PILOT_AUTHORIZED:-0}" != "1" ]]; then
  echo "Set PILOT_AUTHORIZED=1 only after model downloads and paid GPU rollout/training are approved" >&2
  exit 2
fi

case "${stage}" in
  repository_grpo)
    export ADV_ESTIMATOR=grpo
    export LOSS_AGG_MODE=token-mean
    export USE_INVALID_ACTION_PENALTY=True
    ;;
  matched_grpo)
    export ADV_ESTIMATOR=grpo
    export LOSS_AGG_MODE=seq-mean-token-sum
    export USE_INVALID_ACTION_PENALTY=False
    ;;
  exact)
    export ADV_ESTIMATOR=exact
    export LOSS_AGG_MODE=seq-mean-token-sum
    export USE_INVALID_ACTION_PENALTY=False
    ;;
  *)
    echo "unsupported pilot stage: ${stage}" >&2
    exit 2
    ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "${script_dir}/launch_managed.sh" "${environment_name}" "$@"
