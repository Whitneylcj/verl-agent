"""Durable local monitoring artifacts for EXACT training runs."""

from __future__ import annotations

import gzip
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class ExactSafetyStop(RuntimeError):
    """Raised before an optimizer step when a hard EXACT invariant fails."""


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"monitoring value is not finite: {value}")
    return value


class ExactObserver:
    """Write heartbeat, scalar metrics, credit traces, and readable rollouts."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        conservation_tolerance: float = 1e-8,
        residual_warning_ratio: float = 0.95,
        initial_env_steps: int = 0,
        initial_generated_tokens: int = 0,
        initial_step: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conservation_tolerance = float(conservation_tolerance)
        self.residual_warning_ratio = float(residual_warning_ratio)
        self.cumulative_env_steps = int(initial_env_steps)
        self.cumulative_generated_tokens = int(initial_generated_tokens)
        self._write_heartbeat(
            status="initializing",
            step=initial_step,
            metrics={},
            warnings=[],
        )

    def _validated_metrics(
        self,
        metrics: Mapping[str, Any],
        step: int,
    ) -> dict[str, Any]:
        try:
            validated = _json_value(dict(metrics))
        except (TypeError, ValueError) as error:
            reason = f"non-finite or non-serializable monitoring metric: {error}"
            self._write_heartbeat(
                status="safety_stopped",
                step=step,
                metrics={},
                warnings=[reason],
            )
            raise ExactSafetyStop(reason) from error
        if not isinstance(validated, dict):
            raise TypeError("validated monitoring metrics must remain a mapping")
        return validated

    def _atomic_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(_json_value(payload), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)

    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            json.dump(_json_value(payload), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()

    def _write_heartbeat(
        self,
        status: str,
        step: int,
        metrics: Mapping[str, Any],
        warnings: Sequence[str],
    ) -> None:
        self._atomic_json(
            self.output_dir / "heartbeat.json",
            {
                "status": status,
                "step": int(step),
                "updated_unix": time.time(),
                "cumulative_env_steps": self.cumulative_env_steps,
                "cumulative_generated_tokens": self.cumulative_generated_tokens,
                "warnings": list(warnings),
                "metrics": dict(metrics),
            },
        )

    def observe_credit(
        self,
        step: int,
        metrics: Mapping[str, float],
        traces: Sequence[Mapping[str, Any]],
        rollout_records: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        metrics = self._validated_metrics(metrics, step=step)
        conservation_error = float(metrics["exact/conservation_error_max"])
        warnings = []
        if float(metrics["exact/residual_ratio_mean"]) >= self.residual_warning_ratio:
            warnings.append("high_residual_ratio")
        if float(metrics["exact/schema_fallback_rate"]) > 0:
            warnings.append("opaque_schema_fallback")

        trace_payload = {
            "step": int(step),
            "recorded_unix": time.time(),
            "traces": list(traces),
        }
        with gzip.open(self.output_dir / "credit_traces.jsonl.gz", "at", encoding="utf-8") as handle:
            json.dump(_json_value(trace_payload), handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        for record in rollout_records:
            self._append_jsonl(
                self.output_dir / "rollout_samples.jsonl",
                {"step": int(step), "recorded_unix": time.time(), **dict(record)},
            )

        self._write_heartbeat(status="credit_checked", step=step, metrics=metrics, warnings=warnings)
        if conservation_error > self.conservation_tolerance:
            reason = (
                f"EXACT conservation error {conservation_error:.3e} exceeded "
                f"{self.conservation_tolerance:.3e}"
            )
            self._write_heartbeat(
                status="safety_stopped",
                step=step,
                metrics=metrics,
                warnings=[*warnings, reason],
            )
            raise ExactSafetyStop(reason)
        return warnings

    def complete_step(self, step: int, metrics: Mapping[str, Any], warnings: Sequence[str]) -> None:
        metrics = self._validated_metrics(metrics, step=step)
        exact_env_steps = int(float(metrics.get("exact/env_step_count", 0)))
        exact_tokens = int(float(metrics.get("exact/generated_token_count", 0)))
        self.cumulative_env_steps += exact_env_steps
        self.cumulative_generated_tokens += exact_tokens
        payload = {"step": int(step), "recorded_unix": time.time(), "metrics": dict(metrics)}
        self._append_jsonl(self.output_dir / "metrics.jsonl", payload)
        self._write_heartbeat(status="running", step=step, metrics=metrics, warnings=warnings)

    def mark_completed(self, step: int, metrics: Mapping[str, Any]) -> None:
        self._write_heartbeat(status="completed", step=step, metrics=metrics, warnings=[])

    def mark_budget_reached(self, step: int, metrics: Mapping[str, Any]) -> None:
        self._write_heartbeat(
            status="budget_reached",
            step=step,
            metrics=metrics,
            warnings=[],
        )


def build_rollout_records(tokenizer: Any, batch: Any, max_records: int = 8) -> list[dict[str, Any]]:
    """Select deterministic reward/validity/credit strata and decode only those rows."""

    padding = np.asarray(
        batch.non_tensor_batch.get("exact_padding", np.zeros(len(batch), dtype=bool)),
        dtype=bool,
    )
    candidates = np.flatnonzero(~padding)
    if candidates.size == 0:
        return []
    rewards = np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float64)
    invalid = ~np.asarray(
        batch.non_tensor_batch.get("is_action_valid", np.ones(len(batch), dtype=bool)),
        dtype=bool,
    )
    advantage_strength = batch.batch["advantages"].detach().abs().sum(dim=-1).cpu().numpy()

    selected: list[int] = []
    priority = [
        int(candidates[np.argmin(rewards[candidates])]),
        int(candidates[np.argmax(rewards[candidates])]),
        int(candidates[np.argmax(advantage_strength[candidates])]),
    ]
    invalid_candidates = candidates[invalid[candidates]]
    if invalid_candidates.size:
        priority.append(int(invalid_candidates[0]))
    seen_trajectories = set()
    for row in candidates:
        trajectory_id = str(batch.non_tensor_batch["traj_uid"][row])
        if trajectory_id not in seen_trajectories:
            priority.append(int(row))
            seen_trajectories.add(trajectory_id)
    for row in priority:
        if row not in selected:
            selected.append(row)
        if len(selected) >= max_records:
            break

    records = []
    for row in selected:
        records.append(
            {
                "row": row,
                "trajectory_id": str(batch.non_tensor_batch["traj_uid"][row]),
                "step_id": int(batch.non_tensor_batch["exact_step_id"][row]),
                "episode_return": float(rewards[row]),
                "is_action_valid": bool(not invalid[row]),
                "credit_token_sum": float(batch.batch["advantages"][row].sum().item()),
                "prompt": tokenizer.decode(batch.batch["prompts"][row], skip_special_tokens=True),
                "response": tokenizer.decode(batch.batch["responses"][row], skip_special_tokens=True),
            }
        )
    return records
