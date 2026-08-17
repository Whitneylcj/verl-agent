"""Compare controlled agentic pilots and reject unintended config drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from recipe.exact.inspect_run import build_run_report

_CONTROLLED_OVERRIDE_KEYS = {
    "algorithm.adv_estimator",
    "actor_rollout_ref.actor.loss_agg_mode",
    "actor_rollout_ref.actor.use_invalid_action_penalty",
    "trainer.experiment_name",
    "trainer.default_local_dir",
    "trainer.rollout_data_dir",
    "trainer.validation_data_dir",
    "trainer.resolved_config_path",
    "algorithm.exact.monitor.output_dir",
}
_COMPARISON_METRICS = (
    "agent_diag/success_rate",
    "episode/reward/mean",
    "episode/length/mean",
    "agent_diag/invalid_step_rate",
    "agent_diag/max_steps_rate",
    "agent_diag/terminal_failure_rate",
    "agent_diag/repeated_action_rate",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/entropy_loss",
    "actor/grad_norm",
    "response_length/clip_ratio",
    "perf/throughput",
    "exact/conservation_error_max",
    "exact/residual_ratio_mean",
    "exact/cone_density_mean",
    "exact/schema_fallback_rate",
    "exact/credit_std",
)


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing run manifest: {path}")
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"run manifest must be an object: {path}")
    return value


def _override_map(values: Sequence[str]) -> dict[str, str]:
    result = {}
    for value in values:
        if "=" not in value:
            continue
        key, raw_value = value.split("=", 1)
        result[key.lstrip("+")] = raw_value
    return result


def audit_run_fairness(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Identify config differences outside the declared estimator controls."""

    if len(manifests) < 2:
        raise ValueError("at least two run manifests are required")
    identity_fields = ("environment", "model_path", "seed", "exact_mode")
    identity_mismatches = {}
    for field in identity_fields:
        values = [manifest["experiment"].get(field) for manifest in manifests]
        if any(value != values[0] for value in values[1:]):
            identity_mismatches[field] = values
    runtime_fields = {
        "git.commit": tuple(manifest.get("git", {}).get("commit") for manifest in manifests),
        "runtime.python_version": tuple(manifest.get("runtime", {}).get("python_version") for manifest in manifests),
        "runtime.packages": tuple(manifest.get("runtime", {}).get("packages") for manifest in manifests),
        "runtime.gpus": tuple(manifest.get("runtime", {}).get("gpus") for manifest in manifests),
        "paths": tuple(manifest.get("paths") for manifest in manifests),
    }
    for field, values in runtime_fields.items():
        if any(value != values[0] for value in values[1:]):
            identity_mismatches[field] = list(values)

    override_maps = [_override_map(manifest.get("hydra_overrides", ())) for manifest in manifests]
    keys = set().union(*(values.keys() for values in override_maps))
    uncontrolled = {}
    controlled = {}
    for key in sorted(keys):
        values = [mapping.get(key) for mapping in override_maps]
        if all(value == values[0] for value in values[1:]):
            continue
        target = controlled if key in _CONTROLLED_OVERRIDE_KEYS else uncontrolled
        target[key] = values
    return {
        "passed": not identity_mismatches and not uncontrolled,
        "identity_mismatches": identity_mismatches,
        "uncontrolled_override_differences": uncontrolled,
        "controlled_override_differences": controlled,
    }


def build_comparison(run_dirs: Sequence[str | Path], window: int = 20) -> dict[str, Any]:
    """Return fairness evidence and aligned latest/window metrics."""

    paths = [Path(path).expanduser().resolve() for path in run_dirs]
    manifests = [_load_manifest(path) for path in paths]
    reports = [build_run_report(path, window=window) for path in paths]
    runs = []
    for path, manifest, report in zip(paths, manifests, reports):
        trends = report["trends"]
        runs.append(
            {
                "name": manifest["experiment"]["name"],
                "path": str(path),
                "algorithm": manifest["experiment"]["algorithm"],
                "loss_agg_mode": manifest["experiment"]["loss_agg_mode"],
                "heartbeat": report["heartbeat"],
                "latest_validation": report["latest_validation"],
                "diagnostic_tag_counts": report["diagnostic_tag_counts"],
                "metrics": {metric: trends[metric] for metric in _COMPARISON_METRICS if metric in trends},
            }
        )
    return {
        "fairness": audit_run_fairness(manifests),
        "window": window,
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--fail-on-uncontrolled-drift", action="store_true")
    args = parser.parse_args()
    comparison = build_comparison(args.run_dirs, window=args.window)
    print(json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True))
    if args.fail_on_uncontrolled_drift and not comparison["fairness"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
