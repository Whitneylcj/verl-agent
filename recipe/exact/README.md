# EXACT: Credit by Construction

This recipe implements return-conserving credit for multi-step agent rollouts.
It is intentionally fail-closed: missing verifier state, changing factor schemas,
discontinuous checkpoints, unknown graph routes, or conservation failures stop
the update instead of silently falling back to a biased estimator.

## Data flow

1. Each environment manager records a verifier-only factor snapshot before and
   after every action. These values are never added to the model prompt.
2. `build_conserved_atoms` constructs factor-potential deltas and one terminal
   residual whose pathwise sum is exactly the episode return.
3. A prefix-predictable effect schema routes future atoms to response spans.
   Sokoban, ALFWorld, and WebShop conservatively retain all future factors
   because observation history can mediate later policy actions. AppWorld uses
   full Exact-G: its evaluator is compiled into per-test model read sets, its
   strict one-call JSON action is split into selector and prefix-known argument
   spans, and API possible writes are intersected with factor reads. Unknown
   evaluator constructs, unsupported AppWorld versions, and invalid actions
   widen routes instead of dropping causal edges.
4. `compute_exact_advantage` writes span credit onto response tokens and masks
   data-parallel copy padding. `seq-mean-token-sum` makes the score of a span the
   sum of its token log probabilities.
5. Graph-CV optionally fits bounded coefficients from detached credit proxies;
   coefficients activate only on the next rollout batch and are checkpointed.

The hard graph estimator is unbiased when its route is conservative. PPO ratio
clipping, KL regularization, and repeated optimizer steps are practical training
choices and are reported separately from the raw on-policy identity.
Programmatic 0–1 factors use the normalized potential scale `1.0` by default;
set `POTENTIAL_SCALE` only as a declared ablation, never as an implicit tuning
change between matched runs.

## Validation order

Run local theory tests first:

```bash
EXACT_TOY_AUDIT_PATH=/tmp/exact-toy-audit.json \
  bash examples/exact_trainer/run_toy_audit.sh
python -m pytest -q tests/recipe/exact
```

The stage-0 audit exactly enumerates an overwritable, branched Toy SCM with
environment noise and irrelevant actions. It checks pathwise conservation,
sound and corrupted graph gradients, the realized-mask counterexample,
potential corruption, Monte Carlo confidence intervals, and distractor
variance/MSE/SNR scaling. A paid pilot refuses to start unless the saved report
passed on the current tracked-clean Git commit.

On the prepared GPU host, compose a launch without starting environments:

```bash
PREFLIGHT_ONLY=1 bash examples/exact_trainer/run_sokoban.sh
```

The launcher defaults Sokoban, ALFWorld, and WebShop to `EXACT_MODE=temporal`
(Exact-T), and AppWorld to `EXACT_MODE=graph` (Exact-G). Set `EXACT_MODE`
explicitly only for a declared ablation; the resolved mode is recorded in the
experiment name, Hydra config, and run manifest.

Then run one environment probe, a repository-style GRPO pipeline smoke, a
strictly matched GRPO control, and an EXACT smoke before a longer pilot. The
repository-style stage uses token-mean loss and the repository's invalid-action
reward penalty; it verifies the inherited algorithm/config family but does not
reproduce a published result after single-GPU, LoRA, batch-size, or text-mode
adaptations. The matched control disables that extra reward shaping, uses
`seq-mean-token-sum`, and reuses the EXACT model, prompt, seed, rollout budget,
optimizer, schedule, and hardware. Matched GRPO versus EXACT is the estimator
comparison.

`MODEL_PATH` may be a Hugging Face model ID or a resolved local snapshot. The
launcher derives a distinct output tag from its basename; set `MODEL_TAG` when
two checkpoints share a basename. Qwen3 model IDs automatically disable the
thinking chat-template mode used by the repository's Qwen3 agent recipes. This
is only a compatibility setting: every new model family still requires a
tokenizer, action-format, Transformers, and vLLM smoke test before training.

After download authorization, prepare the pinned default model and exercise
both inference backends with:

```bash
MODEL_DOWNLOAD_AUTHORIZED=1 bash examples/exact_trainer/prepare_model.sh
```

The script resumes the official Hugging Face cache, requires the pinned Qwen
revision, validates every safetensors shard plus local config/tokenizer assets,
writes a model manifest under `/root/autodl-tmp/config/models/`, and passes the
resolved immutable snapshot path to both smoke tests. It does not train.

If the provider's Hugging Face large-file route is too slow, use the supported
ModelScope mirror while preserving the Hugging Face model identity:

```bash
MODEL_DOWNLOAD_AUTHORIZED=1 MODEL_SOURCE=modelscope \
  bash examples/exact_trainer/prepare_model.sh
```

For the pinned default Qwen model, the preparation gate requires the official
`model.safetensors` SHA-256 and records the hash, byte size, ModelScope source
revision, requested Hugging Face revision, tokenizer class, and resolved local
snapshot in the manifest. A mirror download without an expected weight hash is
rejected for other models.

Keep the first paid run to one conservatively reserved rollout batch. For a
Sokoban batch with two prompts, two trajectories per prompt, and 15 steps:

```bash
TRAIN_SIZE=2 VALIDATION_SIZE=2 GROUP_SIZE=2 \
MAX_ENV_STEPS=60 MAX_GENERATED_TOKENS=15360 TOTAL_EPOCHS=1 \
PILOT_AUTHORIZED=1 bash examples/exact_trainer/run_sokoban.sh
```

Prepared parquet inputs are size-addressed under
`data/verl-agent/trainN_valM/text/`. Changing `TRAIN_SIZE` or
`VALIDATION_SIZE` therefore cannot silently reuse a larger prior dataset; the
three matched estimator stages share the same size-addressed files.
These files contain only deterministic empty text prompts and sample indices,
because the actual task prompt comes from the selected agent environment. They
are generated locally and do not download the unrelated Geometry3k dataset.

The same `MODEL_PATH`, seed, sizes, budgets, optimizer, and schedule must be
used for the official GRPO baseline, the sequence-score-matched GRPO control,
and EXACT. Only start these GPU runs after explicit authorization.

On an SSH host, keep an authorized pilot alive across disconnects with the
stage launcher (run it after sourcing the prepared environment):

```bash
PILOT_AUTHORIZED=1 bash examples/exact_trainer/launch_pilot_stage.sh \
  repository_grpo sokoban
screen -ls
python -m recipe.exact.inspect_run /path/printed/by/launcher --window 20
```

Run and inspect `repository_grpo`, `matched_grpo`, and `exact` sequentially;
do not auto-launch the next stage after a failed or anomalous run. The launcher
requires the explicit `PILOT_AUTHORIZED=1` guard, refuses existing run/log
paths, starts one named GNU Screen session, and places the console log under
`/root/autodl-tmp/logs/verl-agent/`. To request a graceful interrupt, pass the
printed session name to `stop_managed.sh`; never kill every Python or Ray
process on a shared host.

## Remote environment assets

Keep environment data outside the Git checkout. `run_exact.sh` defaults to
`/root/autodl-tmp/data/{alfworld,appworld,webshop}` and accepts
`VERL_AGENT_SHARED_DATA_ROOT` to move the whole tree. After installing the
documented WebShop text dependencies (`faiss-cpu` is needed even for Pyserini's
Lucene search import) and `en_core_web_sm`, download and index the 1k-product
development set with:

```bash
bash examples/exact_trainer/prepare_webshop.sh
```

The script prefers the official Google Drive IDs, validates fixed SHA-256
digests, and falls back to a commit-pinned third-party Hugging Face mirror when
Drive is unavailable. Set `WEBSHOP_DATA_SOURCE=drive` to forbid that fallback or
`WEBSHOP_DATA_SOURCE=mirror` to select it explicitly. The files are then
validated as 1,000 unique, attribute-aligned products with non-empty matching
human instructions before indexing.

Set `APPWORLD_PORT_FILE` to the persistent port manifest produced by the
AppWorld service launcher. The client refuses to start when that manifest is
missing or too short for the requested train/validation workers.

After installing the pinned AppWorld checkout and its data, start only the
ports owned by this recipe with:

```bash
bash examples/exact_trainer/run_appworld_schema_audit.sh
bash examples/exact_trainer/start_appworld_services.sh
bash examples/exact_trainer/run_appworld_probe.sh
# later, stop exactly those sessions
bash examples/exact_trainer/stop_appworld_services.sh
```

The launcher defaults to eight loopback services on ports 8200-8207, checks
each `/docs` endpoint before publishing the manifest, and never kills unrelated
AppWorld or Python processes.

The schema audit covers every non-template task, requires exact agreement with
the canonical `test_data.json` requirements, and permits at most 1% of factors
to use the explicit all-model read fallback. Its report is commit- and
AppWorld-source-bound (currently `a072b7a86e7c1d5b1d7175659d750ebb9b79f10a`).
The real probe also exercises Ray reset/evaluate/step and selector/argument
resolution. An AppWorld EXACT pilot refuses stale schema or probe evidence.

## Monitoring artifacts

Every non-preflight launcher first writes `run_manifest.json` at the experiment
root and refuses tracked Git changes or an unidentified non-empty run directory.
The manifest records the commit, branch, model, seed, package/GPU inventory,
and redacted Hydra overrides. `resolved_config.yaml` is then atomically written
inside the Ray task before model loading. Set `RESUME_RUN=1` only for an
identical commit and override set; mismatched resumes fail closed.

The standard GRPO baseline, matched GRPO control, and EXACT all write the same
common agentic artifacts under `outputs/.../monitor/`:

- `heartbeat.json`: status, last step, warnings, and cumulative usage;
- `metrics.jsonl`: framework and agentic metrics per optimizer step;
- `validation_metrics.jsonl`: step-zero and periodic held-out metrics;
- `rollout_samples.jsonl`: deterministic reward/validity/advantage strata with
  readable prompts and actions;
- `trajectory_diagnostics.jsonl`: every trajectory indexed by invalid steps,
  termination, repeated actions, and success;
- `alerts.jsonl`: only steps that cross configurable PPO KL, clip fraction,
  gradient norm, response clipping, invalid-action, or EXACT warning thresholds.

EXACT additionally writes `credit_traces.jsonl.gz` with atoms, routes,
residuals, span credits, factor progress, and conservation health. Baselines do
not fabricate unavailable factor or causal-credit signals.

Persisted rollout text is size-bounded and redacts common credentials, email
addresses, and phone numbers by default. Diagnostic tags are deterministic
signals for investigation, not proof of a semantic root cause.

The run inspector also emits bounded `diagnoses`: each entry records the exact
metric or lifecycle state that crossed a deterministic threshold and the next
checks to perform. Treat these entries as investigation hypotheses; the
inspector never changes prompts, rewards, budgets, or optimizer settings.

Compare all estimators using `agent_diag/success_rate`, invalid-step and
trajectory rates, max-step/terminal-failure rates, repeated actions, reward,
episode length, KL/clip fraction, entropy, gradient norm, response clipping,
throughput, and validation metrics. For EXACT also track
`exact/conservation_error_max`, `exact/residual_ratio_mean`,
`exact/cone_density_mean`, `exact/schema_fallback_rate`, credit quantiles,
factor-progress rates, verifier snapshot cost, and active GPU-hours. AppWorld
also reports resource-graph span coverage, argument-span resolution, opaque
factor-read rate, schema-version support, and evaluator-compile fallback. Before
starting each rollout batch, the launcher
reserves its worst-case usage and stops conservatively when the next batch could
exceed the configured environment-step or generated-token budget.
The underlying verl `Tracking` interface supports console, W&B, TensorBoard,
MLflow, SwanLab, VEMLP W&B, and ClearML; the default EXACT launcher uses console
plus TensorBoard while retaining local JSONL artifacts for lossless diagnosis.
Poll a live or completed run without loading the model:

```bash
python -m recipe.exact.inspect_run /path/to/experiment --window 20
python -m recipe.exact.inspect_run /path/to/experiment --show-rollouts 3
```

After the three controlled pilots exist, audit unintended configuration drift
and compare aligned metrics before interpreting any apparent gain:

```bash
python -m recipe.exact.compare_runs \
  /path/to/standard_grpo /path/to/matched_grpo /path/to/exact \
  --window 20 --fail-on-uncontrolled-drift
```

Differences in estimator, loss aggregation, repository-style invalid-action
penalty, experiment name, and output paths are declared controls. A changed
model, seed, environment, learning rate, budget, prompt length, batch size, or
optimizer setting fails the audit. Interpret the repository-style stage as a
pipeline check; use matched GRPO versus EXACT for the causal estimator contrast.
