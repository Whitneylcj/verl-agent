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
   Current adapters conservatively retain all future factors because observation
   history can mediate later policy actions. Unknown schemas use all-to-all.
4. `compute_exact_advantage` writes span credit onto response tokens and masks
   data-parallel copy padding. `seq-mean-token-sum` makes the score of a span the
   sum of its token log probabilities.
5. Graph-CV optionally fits bounded coefficients from detached credit proxies;
   coefficients activate only on the next rollout batch and are checkpointed.

The hard graph estimator is unbiased when its route is conservative. PPO ratio
clipping, KL regularization, and repeated optimizer steps are practical training
choices and are reported separately from the raw on-policy identity.

## Validation order

Run local theory tests first:

```bash
python -m pytest -q tests/recipe/exact
```

On the prepared GPU host, compose a launch without starting environments:

```bash
PREFLIGHT_ONLY=1 bash examples/exact_trainer/run_sokoban.sh
```

Then run one environment probe, the repository's standard GRPO smoke baseline,
a loss-aggregation-matched GRPO control, and an EXACT smoke run before a pilot.
Set `ADV_ESTIMATOR=grpo LOSS_AGG_MODE=token-mean` for the standard GRPO loss;
omit `LOSS_AGG_MODE` for the matched sequence-score control used in comparisons.
The matched control reuses the EXACT model, prompt, seed, rollout budget,
optimizer, schedule, and hardware settings.

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

Set `APPWORLD_PORT_FILE` to the persistent port manifest produced by the
AppWorld service launcher. The client refuses to start when that manifest is
missing or too short for the requested train/validation workers.

After installing the pinned AppWorld checkout and its data, start only the
ports owned by this recipe with:

```bash
bash examples/exact_trainer/start_appworld_services.sh
# later, stop exactly those sessions
bash examples/exact_trainer/stop_appworld_services.sh
```

The launcher defaults to eight loopback services on ports 8200-8207, checks
each `/docs` endpoint before publishing the manifest, and never kills unrelated
AppWorld or Python processes.

## Monitoring artifacts

`outputs/.../monitor/` contains:

- `heartbeat.json`: status, last step, warnings, and cumulative usage;
- `metrics.jsonl`: framework and EXACT metrics per optimizer step;
- `credit_traces.jsonl.gz`: atoms, routes, residuals, and span credits;
- `rollout_samples.jsonl`: deterministic reward/validity/credit strata with
  readable prompts and actions;
- `trajectory_diagnostics.jsonl`: every trajectory indexed by invalid steps,
  termination, factor progress, repeated actions, success, and credit health;
- `alerts.jsonl`: only steps that cross configurable PPO KL, clip fraction,
  gradient norm, response clipping, invalid-action, or EXACT warning thresholds.

Persisted rollout text is size-bounded and redacts common credentials, email
addresses, and phone numbers by default. Diagnostic tags are deterministic
signals for investigation, not proof of a semantic root cause.

Track `exact/conservation_error_max`, `exact/residual_ratio_mean`,
`exact/cone_density_mean`, `exact/schema_fallback_rate`, credit quantiles,
episode success/reward/length, invalid actions, PPO KL/clip fraction, entropy,
gradient norm, throughput, and GPU memory. Before starting each rollout batch,
the launcher reserves its worst-case usage and stops conservatively when the
next batch could exceed 10,000 environment steps or 1,000,000 generated tokens.
The underlying verl `Tracking` interface supports console, W&B, TensorBoard,
MLflow, SwanLab, VEMLP W&B, and ClearML; the default EXACT launcher uses console
plus TensorBoard while retaining local JSONL artifacts for lossless diagnosis.
