"""Summarize an agentic baseline or EXACT run without loading its model."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_METRICS = (
    "agent_diag/success_rate",
    "episode/reward/mean",
    "episode/length/mean",
    "agent_diag/invalid_step_rate",
    "agent_diag/syntax_invalid_step_rate",
    "agent_diag/execution_error_step_rate",
    "agent_diag/invalid_trajectory_rate",
    "agent_diag/max_steps_rate",
    "agent_diag/terminal_failure_rate",
    "agent_diag/repeated_action_rate",
    "exact_diag/no_factor_progress_rate",
    "exact/residual_ratio_mean",
    "exact/cone_density_mean",
    "exact/schema_fallback_rate",
    "exact/resource_graph_span_rate",
    "exact/appworld_argument_span_rate",
    "exact/appworld_factor_opaque_rate",
    "exact/appworld_version_supported",
    "exact/appworld_source_supported",
    "exact/appworld_factor_compile_fallback_rate",
    "exact/credit_std",
    "exact/probe_seconds_per_snapshot",
    "exact/graph_compile_seconds",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/entropy_loss",
    "actor/grad_norm",
    "response_length/clip_ratio",
    "perf/time_per_step",
    "perf/throughput",
    "perf/active_gpu_hours_per_step",
    "training/cumulative_active_gpu_hours",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _tail_jsonl(path: Path, count: int) -> list[dict[str, Any]]:
    if count <= 0 or not path.exists():
        return []
    records: deque[dict[str, Any]] = deque(maxlen=count)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"expected a JSON object at {path}:{line_number}")
            records.append(value)
    return list(records)


def _finite_values(records: Iterable[Mapping[str, Any]], metric: str) -> list[float]:
    values = []
    for record in records:
        raw_value = record.get("metrics", {}).get(metric)
        if isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
            values.append(float(raw_value))
    return values


def _resolve_monitor_dir(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    monitor = root / "monitor"
    return monitor if monitor.is_dir() else root


def diagnose_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Map deterministic metric patterns to bounded investigation hypotheses."""

    trends = report.get("trends", {})

    def latest(metric: str) -> float | None:
        value = trends.get(metric, {}).get("latest")
        return float(value) if isinstance(value, (int, float)) else None

    validation_metrics = (report.get("latest_validation") or {}).get("metrics", {})

    def latest_or_validation(metric: str) -> tuple[str, float | None]:
        value = latest(metric)
        if value is not None:
            return metric, value
        validation_metric = f"val/{metric}"
        value = validation_metrics.get(validation_metric)
        return validation_metric, float(value) if isinstance(value, (int, float)) else None

    diagnoses = []

    def add(code: str, severity: str, evidence: Mapping[str, Any], next_checks: Sequence[str]) -> None:
        diagnoses.append(
            {
                "code": code,
                "severity": severity,
                "evidence": dict(evidence),
                "next_checks": list(next_checks),
            }
        )

    run_state = str(report.get("run_state", "unknown"))
    run_age = report.get("run_age_seconds")
    stale_after = report.get("stale_after_seconds")
    startup_is_stale = run_state == "unknown" or not isinstance(run_age, (int, float)) or not isinstance(stale_after, (int, float)) or run_age >= stale_after
    if run_state in {"launched_no_heartbeat", "ray_configured_no_heartbeat", "unknown"} and startup_is_stale:
        add(
            "startup_incomplete",
            "high",
            {"run_state": run_state},
            (
                "inspect the persisted console log",
                "verify model/runtime initialization before changing training hyperparameters",
            ),
        )
    if run_state == "failed":
        failure = (report.get("heartbeat") or {}).get("failure", {})
        add(
            "trainer_failed",
            "high",
            {
                "failure_type": failure.get("type"),
                "failure_message": failure.get("message"),
            },
            (
                "inspect the persisted console log around the final traceback",
                "fix the runtime or data cause before changing optimization settings",
            ),
        )
    if run_state == "safety_stopped":
        add(
            "exact_safety_stop",
            "high",
            {"warnings": (report.get("heartbeat") or {}).get("warnings", [])},
            (
                "inspect the failed EXACT invariant and credit trace",
                "do not resume from the rejected optimizer step until the invariant is restored",
            ),
        )
    heartbeat_age = report.get("heartbeat_age_seconds")
    stale_after = report.get("stale_after_seconds")
    if run_state in {"initializing", "validated", "credit_checked", "rollout_checked", "running"} and isinstance(heartbeat_age, (int, float)) and isinstance(stale_after, (int, float)) and heartbeat_age >= stale_after:
        add(
            "heartbeat_stale",
            "high",
            {
                "run_state": run_state,
                "heartbeat_age_seconds": heartbeat_age,
                "stale_after_seconds": stale_after,
            },
            (
                "inspect the managed screen session and persisted console log",
                "check GPU utilization and Ray workers before interrupting the run",
            ),
        )
    invalid_metric, invalid_rate = latest_or_validation("agent_diag/invalid_step_rate")
    if invalid_rate is not None and invalid_rate >= 0.2:
        add(
            "high_invalid_action_rate",
            "high",
            {invalid_metric: invalid_rate},
            (
                "inspect invalid-action rollout samples",
                "compare syntax-invalid and execution-error rates before changing the parser",
            ),
        )
    syntax_invalid_metric, syntax_invalid_rate = latest_or_validation("agent_diag/syntax_invalid_step_rate")
    if syntax_invalid_rate is not None and syntax_invalid_rate >= 0.2:
        add(
            "high_action_syntax_error_rate",
            "high",
            {syntax_invalid_metric: syntax_invalid_rate},
            (
                "inspect raw model responses against the advertised JSON contract",
                "check projection failures without weakening the one-call boundary",
            ),
        )
    execution_error_metric, execution_error_rate = latest_or_validation("agent_diag/execution_error_step_rate")
    if execution_error_rate is not None and execution_error_rate >= 0.2:
        add(
            "appworld_api_execution_errors",
            "high",
            {execution_error_metric: execution_error_rate},
            (
                "inspect AppWorld errors for hallucinated APIs and invalid arguments",
                "verify the policy queries API documentation after an execution error",
            ),
        )
    repeated_rate = latest("agent_diag/repeated_action_rate")
    if repeated_rate is not None and repeated_rate >= 0.2:
        add(
            "policy_action_loop",
            "medium",
            {"agent_diag/repeated_action_rate": repeated_rate},
            (
                "inspect consecutive responses and observations",
                "verify history contains the last action and changed state",
            ),
        )
    max_steps_rate = latest("agent_diag/max_steps_rate")
    if max_steps_rate is not None and max_steps_rate >= 0.5:
        add(
            "frequent_max_step_termination",
            "medium",
            {"agent_diag/max_steps_rate": max_steps_rate},
            (
                "separate invalid, repeated, and valid-but-unproductive trajectories",
                "do not increase max_steps before inspecting rollouts",
            ),
        )
    response_clip = latest("response_length/clip_ratio")
    if response_clip is not None and response_clip >= 0.1:
        add(
            "response_clipping",
            "high",
            {"response_length/clip_ratio": response_clip},
            (
                "inspect whether </action> is truncated",
                "adjust response budget only after checking verbosity and stop behavior",
            ),
        )
    ppo_kl = latest("actor/ppo_kl")
    if ppo_kl is not None and abs(ppo_kl) >= 0.1:
        add(
            "large_policy_update",
            "high",
            {"actor/ppo_kl": ppo_kl},
            (
                "check learning rate, PPO epochs, and KL coefficient",
                "compare reward/success before retaining the update",
            ),
        )
    clipfrac = latest("actor/pg_clipfrac")
    if clipfrac is not None and clipfrac >= 0.3:
        add(
            "high_policy_clip_fraction",
            "high",
            {"actor/pg_clipfrac": clipfrac},
            (
                "inspect advantage scale and ratio distribution",
                "consider a smaller update only after ruling out padding/mask errors",
            ),
        )
    grad_norm = latest("actor/grad_norm")
    if grad_norm is not None and grad_norm >= 100:
        add(
            "high_gradient_norm",
            "high",
            {"actor/grad_norm": grad_norm},
            (
                "check credit outliers and non-finite values",
                "verify clipping is active before changing optimization",
            ),
        )
    schema_fallback = latest("exact/schema_fallback_rate")
    if schema_fallback is not None and schema_fallback > 0:
        add(
            "opaque_exact_schema",
            "high",
            {"exact/schema_fallback_rate": schema_fallback},
            (
                "inspect environment effect-schema records",
                "treat graph attribution as all-to-all until the schema is repaired",
            ),
        )
    appworld_factor_opaque = latest("exact/appworld_factor_opaque_rate")
    if appworld_factor_opaque is not None and appworld_factor_opaque > 0:
        add(
            "partial_appworld_factor_graph",
            "medium",
            {"exact/appworld_factor_opaque_rate": appworld_factor_opaque},
            (
                "inspect task evaluator constructs that forced an all-model read set",
                "treat affected factors as conservative but not sparse",
            ),
        )
    appworld_version_supported = latest("exact/appworld_version_supported")
    if appworld_version_supported is not None and appworld_version_supported < 1:
        add(
            "unsupported_appworld_graph_version",
            "high",
            {"exact/appworld_version_supported": appworld_version_supported},
            (
                "verify the installed AppWorld source and data versions",
                "retain the global-write fallback until the new source is audited",
            ),
        )
    appworld_source_supported = latest("exact/appworld_source_supported")
    if appworld_source_supported is not None and appworld_source_supported < 1:
        add(
            "unsupported_appworld_graph_source",
            "high",
            {"exact/appworld_source_supported": appworld_source_supported},
            (
                "verify the installed AppWorld Git revision against the schema audit",
                "retain the global-write fallback until that exact source is audited",
            ),
        )
    appworld_compile_fallback = latest("exact/appworld_factor_compile_fallback_rate")
    if appworld_compile_fallback is not None and appworld_compile_fallback > 0:
        add(
            "appworld_factor_compile_failure",
            "high",
            {"exact/appworld_factor_compile_fallback_rate": appworld_compile_fallback},
            (
                "inspect evaluator parse errors in the rollout schema",
                "do not interpret factor-level sparsity while all factors read all models",
            ),
        )
    residual_ratio = latest("exact/residual_ratio_mean")
    if residual_ratio is not None and residual_ratio >= 0.8:
        add(
            "weak_exact_factor_explanation",
            "medium",
            {"exact/residual_ratio_mean": residual_ratio},
            (
                "inspect factor snapshots and terminal residual",
                "improve verifier factors without adding model-visible reward hints",
            ),
        )
    cone_density = latest("exact/cone_density_mean")
    if cone_density is not None and cone_density >= 0.9:
        add(
            "dense_exact_credit_cone",
            "medium",
            {"exact/cone_density_mean": cone_density},
            (
                "inspect routed factor descendants",
                "do not claim sparse causal localization from a nearly all-to-all graph",
            ),
        )
    no_progress = latest("exact_diag/no_factor_progress_rate")
    if no_progress is not None and no_progress >= 0.8:
        add(
            "little_verifier_progress",
            "medium",
            {"exact_diag/no_factor_progress_rate": no_progress},
            (
                "compare valid actions against factor changes",
                "inspect whether factors are too coarse or the policy never reaches subgoals",
            ),
        )
    return diagnoses


def build_run_report(
    run_dir: str | Path,
    window: int = 20,
    rollout_count: int = 0,
    stale_after_seconds: float = 1_800.0,
    now_unix: float | None = None,
) -> dict[str, Any]:
    """Return a compact machine-readable report for polling and diagnosis."""

    if window <= 0:
        raise ValueError("window must be positive")
    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    monitor_dir = _resolve_monitor_dir(run_dir)
    run_root = monitor_dir.parent if monitor_dir.name == "monitor" else monitor_dir
    manifest = _read_json(run_root / "run_manifest.json")
    heartbeat = _read_json(monitor_dir / "heartbeat.json")
    metric_records = _tail_jsonl(monitor_dir / "metrics.jsonl", window)
    validation_records = _tail_jsonl(monitor_dir / "validation_metrics.jsonl", window)
    diagnostic_records = _tail_jsonl(monitor_dir / "trajectory_diagnostics.jsonl", window * 64)
    alert_records = _tail_jsonl(monitor_dir / "alerts.jsonl", window)
    rollout_records = _tail_jsonl(monitor_dir / "rollout_samples.jsonl", rollout_count)

    active_steps = {int(record["step"]) for record in metric_records if "step" in record}
    if active_steps:
        diagnostic_records = [record for record in diagnostic_records if int(record.get("step", -1)) in active_steps]
        alert_records = [record for record in alert_records if int(record.get("step", -1)) in active_steps]

    trends = {}
    for metric in DEFAULT_METRICS:
        values = _finite_values(metric_records, metric)
        if values:
            trends[metric] = {
                "latest": values[-1],
                "window_mean": sum(values) / len(values),
                "window_delta": values[-1] - values[0],
                "observations": len(values),
            }

    tag_counts = Counter(str(tag) for record in diagnostic_records for tag in record.get("diagnostic_tags", ()))
    samples = []
    for record in rollout_records:
        samples.append(
            {
                key: value
                for key, value in record.items()
                if key
                in {
                    "step",
                    "trajectory_id",
                    "step_id",
                    "episode_return",
                    "is_action_valid",
                    "is_action_syntax_valid",
                    "is_action_execution_valid",
                    "credit_token_sum",
                    "diagnostic_tags",
                    "termination",
                    "prompt",
                    "response",
                }
            }
        )

    latest_metrics = metric_records[-1].get("metrics", {}) if metric_records else {}
    if heartbeat is not None:
        run_state = heartbeat.get("status", "unknown")
    elif manifest is not None and (run_root / "resolved_config.yaml").is_file():
        run_state = "ray_configured_no_heartbeat"
    elif manifest is not None:
        run_state = "launched_no_heartbeat"
    else:
        run_state = "unknown"
    current_unix = time.time() if now_unix is None else float(now_unix)
    updated_unix = (heartbeat or {}).get("updated_unix")
    heartbeat_age_seconds = max(current_unix - float(updated_unix), 0.0) if isinstance(updated_unix, (int, float)) else None
    created_unix = (manifest or {}).get("created_unix")
    run_age_seconds = max(current_unix - float(created_unix), 0.0) if isinstance(created_unix, (int, float)) else None
    report = {
        "run_root": str(run_root),
        "monitor_dir": str(monitor_dir),
        "run_state": run_state,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "run_age_seconds": run_age_seconds,
        "stale_after_seconds": float(stale_after_seconds),
        "manifest": manifest,
        "resolved_config_present": (run_root / "resolved_config.yaml").is_file(),
        "heartbeat": heartbeat,
        "latest_step": latest_metrics.get("training/global_step"),
        "latest_validation": validation_records[-1] if validation_records else None,
        "window_steps": sorted(active_steps),
        "trends": trends,
        "diagnostic_tag_counts": dict(sorted(tag_counts.items())),
        "recent_alerts": alert_records,
        "rollout_samples": samples,
    }
    report["diagnoses"] = diagnose_report(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Experiment directory or its monitor/ subdirectory")
    parser.add_argument("--window", type=int, default=20, help="Number of optimizer steps to summarize")
    parser.add_argument("--show-rollouts", type=int, default=0, help="Include this many recent redacted rollout samples")
    parser.add_argument(
        "--stale-after-seconds",
        type=float,
        default=1_800.0,
        help="Diagnose a nonterminal heartbeat as stale after this many seconds",
    )
    parser.add_argument("--fail-on-alert", action="store_true", help="Exit 2 when the latest heartbeat contains warnings")
    args = parser.parse_args()

    report = build_run_report(
        args.run_dir,
        window=args.window,
        rollout_count=args.show_rollouts,
        stale_after_seconds=args.stale_after_seconds,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    warnings = (report.get("heartbeat") or {}).get("warnings", ())
    return 2 if args.fail_on_alert and warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
