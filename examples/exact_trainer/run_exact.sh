#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
cd "${repo_root}"

environment_name=${1:?Usage: run_exact.sh sokoban|alfworld|webshop|appworld}
shift

engine=${ENGINE:-vllm}
model_path=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
model_tag=${MODEL_TAG:-${model_path##*/}}
model_tag=${model_tag//[^[:alnum:]._-]/_}
algorithm_name=${ADV_ESTIMATOR:-exact}
exact_mode=${EXACT_MODE:-graph}
seed=${SEED:-0}
data_root=${VERL_AGENT_DATA_ROOT:-/root/autodl-tmp/data/verl-agent}
output_root=${VERL_AGENT_OUTPUT_ROOT:-/root/autodl-tmp/outputs/verl-agent}
shared_data_root=${VERL_AGENT_SHARED_DATA_ROOT:-/root/autodl-tmp/data}
train_size=${TRAIN_SIZE:-4}
validation_size=${VALIDATION_SIZE:-16}
group_size=${GROUP_SIZE:-4}
total_epochs=${TOTAL_EPOCHS:-999}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-5}
max_env_steps=${MAX_ENV_STEPS:-10000}
max_generated_tokens=${MAX_GENERATED_TOKENS:-1000000}
max_steps=20
max_prompt_length=2048

case "${environment_name}" in
  sokoban)
    env_name=Sokoban
    max_steps=15
    ;;
  alfworld)
    env_name=alfworld/AlfredTWEnv
    max_steps=30
    ;;
  webshop)
    env_name=Webshop
    max_steps=15
    max_prompt_length=4096
    ;;
  appworld)
    env_name=AppWorld
    max_steps=20
    max_prompt_length=4096
    train_size=${TRAIN_SIZE:-2}
    validation_size=${VALIDATION_SIZE:-4}
    group_size=${GROUP_SIZE:-2}
    ;;
  *)
    echo "Unsupported environment: ${environment_name}" >&2
    exit 2
    ;;
esac

loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-sum}
if [[ "${algorithm_name}" == "exact" && "${loss_agg_mode}" != "seq-mean-token-sum" ]]; then
  echo "EXACT requires LOSS_AGG_MODE=seq-mean-token-sum" >&2
  exit 2
fi

loss_tag=${loss_agg_mode//-/_}
experiment_name="${algorithm_name}_${exact_mode}_${loss_tag}_${environment_name}_${model_tag}_seed${seed}"
run_output_dir="${output_root}/${experiment_name}"
train_file="${data_root}/text/train.parquet"
validation_file="${data_root}/text/test.parquet"

common_overrides=(
  "algorithm.adv_estimator=${algorithm_name}"
  "algorithm.exact.mode=${exact_mode}"
  "algorithm.exact.potential.scale=10.0"
  "data.train_files=${train_file}"
  "data.val_files=${validation_file}"
  "data.train_batch_size=${train_size}"
  "data.val_batch_size=${validation_size}"
  "data.max_prompt_length=${max_prompt_length}"
  "data.max_response_length=256"
  "data.filter_overlong_prompts=True"
  "data.truncation=error"
  "data.return_raw_chat=True"
  "actor_rollout_ref.model.path=${model_path}"
  "actor_rollout_ref.model.lora_rank=32"
  "actor_rollout_ref.model.lora_alpha=32"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.optim.lr=1e-6"
  "actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}"
  "actor_rollout_ref.actor.ppo_mini_batch_size=32"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.shuffle=False"
  "actor_rollout_ref.actor.use_kl_loss=True"
  "actor_rollout_ref.actor.kl_loss_coef=0.01"
  "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
  "actor_rollout_ref.actor.use_invalid_action_penalty=False"
  "actor_rollout_ref.actor.fsdp_config.param_offload=False"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.name=${engine}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=0.55"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "algorithm.use_kl_in_reward=False"
  "env.env_name=${env_name}"
  "env.seed=${seed}"
  "env.max_steps=${max_steps}"
  "env.rollout.n=${group_size}"
  "env.sokoban.mode=tiny_rgb_array"
  "env.appworld.action_mode=json_api"
  "env.appworld.validation_split=dev"
  "env.resources_per_worker.num_cpus=0.1"
  "trainer.logger=['console','tensorboard']"
  "trainer.project_name=verl_agent_exact"
  "trainer.experiment_name=${experiment_name}"
  "trainer.n_gpus_per_node=1"
  "trainer.nnodes=1"
  "trainer.save_freq=${save_freq}"
  "trainer.test_freq=${test_freq}"
  "trainer.total_epochs=${total_epochs}"
  "trainer.val_before_train=True"
  "trainer.max_env_steps=${max_env_steps}"
  "trainer.max_generated_tokens=${max_generated_tokens}"
  "trainer.default_local_dir=${run_output_dir}/checkpoints"
  "trainer.rollout_data_dir=${run_output_dir}/rollouts"
  "trainer.validation_data_dir=${run_output_dir}/validation"
  "trainer.resolved_config_path=${run_output_dir}/resolved_config.yaml"
  "algorithm.exact.monitor.output_dir=${run_output_dir}/monitor"
  "algorithm.exact.monitor.enabled=True"
)

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export ALFWORLD_DATA=${ALFWORLD_DATA:-${shared_data_root}/alfworld}
export APPWORLD_ROOT=${APPWORLD_ROOT:-${shared_data_root}/appworld}
export APPWORLD_PORT_FILE=${APPWORLD_PORT_FILE:-/root/autodl-tmp/config/appworld_ports.ports}
export WEBSHOP_DATA_ROOT=${WEBSHOP_DATA_ROOT:-${shared_data_root}/webshop/data}
export WEBSHOP_SEARCH_ROOT=${WEBSHOP_SEARCH_ROOT:-${shared_data_root}/webshop/search_engine}

model_overrides=()
case "${model_path}" in
  *[Qq][Ww][Ee][Nn]3*)
    model_overrides+=("+data.apply_chat_template_kwargs.enable_thinking=False")
    ;;
esac

if [[ "${RUN_DIR_ONLY:-0}" == "1" ]]; then
  printf '%s\n' "${run_output_dir}"
  exit 0
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  python3 -m verl.trainer.main_ppo "${common_overrides[@]}" "${model_overrides[@]}" --cfg job "$@"
  exit 0
fi

if [[ ! -f "${train_file}" || ! -f "${validation_file}" ]]; then
  python3 -m examples.data_preprocess.prepare \
    --mode text \
    --local_dir "${data_root}" \
    --train_data_size "${train_size}" \
    --val_data_size "${validation_size}"
fi

manifest_args=(
  --output-dir "${run_output_dir}"
  --repo-root "${repo_root}"
  --name "${experiment_name}"
  --environment "${environment_name}"
  --algorithm "${algorithm_name}"
  --exact-mode "${exact_mode}"
  --model-path "${model_path}"
  --seed "${seed}"
  --loss-agg-mode "${loss_agg_mode}"
)
if [[ "${RESUME_RUN:-0}" == "1" ]]; then
  manifest_args+=(--resume)
fi
for override in "${common_overrides[@]}" "${model_overrides[@]}" "$@"; do
  manifest_args+=(--override "${override}")
done
python3 -m recipe.exact.run_manifest "${manifest_args[@]}"

python3 -m verl.trainer.main_ppo "${common_overrides[@]}" "${model_overrides[@]}" "$@"
