"""Durable local monitoring artifacts for EXACT training runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

_EMAIL_PATTERN = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE_PATTERN = re.compile(r"(?<!\w)(?:\+?\d[\d(). -]{7,}\d)(?!\w)")
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|password|passwd|secret)\b"
    r"(\s*[:=]\s*)([^\s,;\"'}]+|\"[^\"]*\"|'[^']*')"
)
_ACTION_PATTERN = re.compile(r"<action>\s*(.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


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


def redact_rollout_text(text: Any, max_chars: int = 20_000) -> str:
    """Redact common credentials/PII and bound persisted rollout text size."""

    value = str(text)
    value = _BEARER_PATTERN.sub("Bearer [REDACTED_TOKEN]", value)
    value = _SECRET_PATTERN.sub(r"\1\2[REDACTED_SECRET]", value)
    value = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
    value = _PHONE_PATTERN.sub("[REDACTED_PHONE]", value)
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    marker = "\n...[TRUNCATED]...\n"
    if max_chars <= len(marker):
        return marker[:max_chars]
    retained_chars = max(max_chars - len(marker), 0)
    head_chars = retained_chars * 3 // 5
    tail_chars = retained_chars - head_chars
    return f"{value[:head_chars]}{marker}{value[-tail_chars:] if tail_chars else ''}"


def _snapshot_values(record: Any) -> np.ndarray:
    if isinstance(record, Mapping):
        values = record.get("values", ())
    else:
        values = getattr(record, "values", ())
    return np.asarray(values, dtype=np.float64)


def _schema_uses_fallback(schema: Any) -> bool:
    if schema is None:
        return True
    if not isinstance(schema, Mapping):
        return False
    spans = schema.get("spans", (schema,))
    return any(isinstance(span, Mapping) and bool(span.get("opaque", "descendant_factor_ids" not in span)) for span in spans)


def _valid_response_signature(batch: Any, row: int) -> str:
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        response_length = batch.batch["responses"].shape[-1]
        response_mask = batch.batch["attention_mask"][:, -response_length:]
    token_ids = batch.batch["responses"][row][response_mask[row].bool()].detach().cpu().tolist()
    payload = ",".join(str(int(token_id)) for token_id in token_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def _valid_response_text(tokenizer: Any, batch: Any, row: int) -> str:
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        response_length = batch.batch["responses"].shape[-1]
        response_mask = batch.batch["attention_mask"][:, -response_length:]
    token_ids = batch.batch["responses"][row][response_mask[row].bool()]
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def _action_signature(text: str) -> str | None:
    match = _ACTION_PATTERN.search(text)
    if match is None:
        return None
    content = " ".join(match.group(1).split())
    if not content:
        return None
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        return content.lower()
    return json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _repeated_action_steps(
    signatures: Sequence[str | None],
    rows: Sequence[int],
    step_ids: np.ndarray,
    min_run_length: int = 3,
) -> list[int]:
    repeated = []
    run_length = 1
    for position in range(1, len(signatures)):
        signature = signatures[position]
        if signature is not None and signature == signatures[position - 1]:
            run_length += 1
            if run_length >= min_run_length:
                repeated.append(int(step_ids[rows[position]]))
        else:
            run_length = 1
    return repeated


def _padding_mask(batch: Any) -> np.ndarray:
    raw_padding = batch.non_tensor_batch.get(
        "exact_padding",
        batch.non_tensor_batch.get("rollout_padding", np.zeros(len(batch), dtype=bool)),
    )
    return np.asarray(raw_padding, dtype=bool)


def _step_ids(batch: Any) -> np.ndarray:
    raw_steps = batch.non_tensor_batch.get(
        "agent_step_id",
        batch.non_tensor_batch.get("exact_step_id"),
    )
    if raw_steps is None:
        raise KeyError("agentic monitoring requires agent_step_id")
    return np.asarray(raw_steps, dtype=np.int64)


def build_trajectory_diagnostics(
    batch: Any,
    traces: Sequence[Mapping[str, Any]],
    max_steps: int | None = None,
    residual_warning_ratio: float = 0.95,
    tokenizer: Any | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic, no-model hypotheses for every non-padding trajectory."""

    padding = _padding_mask(batch)
    step_ids = _step_ids(batch)
    trajectory_rows: dict[str, list[int]] = {}
    for row in np.flatnonzero(~padding):
        trajectory_rows.setdefault(str(batch.non_tensor_batch["traj_uid"][row]), []).append(int(row))
    trace_by_id = {str(trace["trajectory_id"]): trace for trace in traces}
    factor_probe_available = all(key in batch.non_tensor_batch for key in ("exact_factor_pre", "exact_factor_post", "exact_effect_schema"))
    diagnostics = []
    for trajectory_id, rows in trajectory_rows.items():
        rows.sort(key=lambda row: int(step_ids[row]))
        rewards = np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float64)
        valid = np.asarray(
            batch.non_tensor_batch.get("is_action_valid", np.ones(len(batch), dtype=bool)),
            dtype=bool,
        )[rows]
        episode_length = int(float(batch.non_tensor_batch.get("episode_lengths", np.full(len(batch), len(rows)))[rows[-1]]))
        tool_call_count = float(batch.non_tensor_batch.get("tool_callings", np.zeros(len(batch)))[rows[-1]])
        done = bool(batch.non_tensor_batch.get("episode_done", np.zeros(len(batch), dtype=bool))[rows[-1]])
        fallback_steps = []
        factor_change_steps = []
        factor_decrease_steps = []
        if factor_probe_available:
            fallback_steps = [int(step_ids[row]) for row in rows if _schema_uses_fallback(batch.non_tensor_batch["exact_effect_schema"][row])]
            for row in rows:
                before = _snapshot_values(batch.non_tensor_batch["exact_factor_pre"][row])
                after = _snapshot_values(batch.non_tensor_batch["exact_factor_post"][row])
                delta = after - before
                step_id = int(step_ids[row])
                if np.any(np.abs(delta) > 1e-12):
                    factor_change_steps.append(step_id)
                if np.any(delta < -1e-12):
                    factor_decrease_steps.append(step_id)

        response_signatures = [_valid_response_signature(batch, row) for row in rows]
        repeated_response_steps = [
            int(step_ids[rows[position]])
            for position in range(1, len(rows))
            if response_signatures[position] == response_signatures[position - 1]
        ]
        action_signatures = (
            [_action_signature(_valid_response_text(tokenizer, batch, row)) for row in rows]
            if tokenizer is not None
            else []
        )
        action_loop_steps = (
            _repeated_action_steps(action_signatures, rows, step_ids)
            if action_signatures
            else []
        )
        repeated_action_steps = sorted(set(repeated_response_steps) | set(action_loop_steps))
        raw_outcomes = batch.non_tensor_batch.get(
            "trajectory_outcomes",
            np.asarray([{} for _ in range(len(batch))], dtype=object),
        )[rows[-1]]
        success_values = {str(key): float(value) for key, value in dict(raw_outcomes).items()}
        primary_success = success_values.get("success_rate")
        if primary_success is not None and primary_success > 0:
            termination = "success"
        elif max_steps is not None and episode_length >= int(max_steps):
            termination = "max_steps"
        elif done:
            termination = "environment_terminal_failure"
        else:
            termination = "collector_incomplete"

        trace = trace_by_id.get(trajectory_id, {})
        residual_ratio = float(trace["residual_ratio"]) if "residual_ratio" in trace else None
        tags = []
        if not bool(np.all(valid)):
            tags.append("invalid_action")
        if termination != "success":
            tags.append(termination)
        if factor_probe_available and not factor_change_steps:
            tags.append("no_factor_progress")
        if repeated_action_steps:
            tags.append("repeated_action")
        if fallback_steps:
            tags.append("opaque_schema_fallback")
        if residual_ratio is not None and residual_ratio >= residual_warning_ratio:
            tags.append("high_residual_ratio")

        diagnostics.append(
            {
                "trajectory_id": trajectory_id,
                "row_indices": rows,
                "step_count": len(rows),
                "episode_length": episode_length,
                "episode_return": float(rewards[rows[-1]]),
                "tool_call_count": tool_call_count,
                "valid_action_ratio": float(np.mean(valid)),
                "invalid_step_ids": [int(step_ids[row]) for row, is_valid in zip(rows, valid) if not is_valid],
                "factor_probe_available": factor_probe_available,
                "factor_change_step_ids": factor_change_steps,
                "factor_decrease_step_ids": factor_decrease_steps,
                "repeated_action_step_ids": repeated_action_steps,
                "repeated_response_step_ids": repeated_response_steps,
                "schema_fallback_step_ids": fallback_steps,
                "success": success_values,
                "termination": termination,
                "diagnostic_tags": tags,
                "residual_ratio": residual_ratio,
                "cone_density": float(trace["cone_density"]) if "cone_density" in trace else None,
                "conservation_error": (float(trace["conservation_error"]) if "conservation_error" in trace else None),
            }
        )
    return diagnostics


def summarize_trajectory_diagnostics(
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Aggregate per-trajectory diagnostics into scalar logger metrics."""

    if not diagnostics:
        return {}
    tags = [set(item.get("diagnostic_tags", ())) for item in diagnostics]
    total_steps = sum(int(item["step_count"]) for item in diagnostics)
    invalid_steps = sum(len(item.get("invalid_step_ids", ())) for item in diagnostics)
    common = {
        "invalid_step_rate": float(invalid_steps / max(total_steps, 1)),
        "invalid_trajectory_rate": float(np.mean(["invalid_action" in item for item in tags])),
        "max_steps_rate": float(np.mean(["max_steps" in item for item in tags])),
        "repeated_action_rate": float(np.mean(["repeated_action" in item for item in tags])),
        "terminal_failure_rate": float(np.mean(["environment_terminal_failure" in item for item in tags])),
    }
    result = {f"agent_diag/{key}": value for key, value in common.items()}
    # Preserve existing dashboards while common agentic comparisons migrate to
    # the estimator-independent namespace.
    result.update({f"exact_diag/{key}": value for key, value in common.items()})
    if any(bool(item.get("factor_probe_available")) for item in diagnostics):
        exact_only = {
            "no_factor_progress_rate": float(np.mean(["no_factor_progress" in item for item in tags])),
            "factor_change_step_rate": float(sum(len(item.get("factor_change_step_ids", ())) for item in diagnostics) / max(total_steps, 1)),
        }
        result.update({f"exact_diag/{key}": value for key, value in exact_only.items()})
    primary_success = [float(item["success"]["success_rate"]) for item in diagnostics if "success_rate" in item.get("success", {})]
    if primary_success:
        success_rate = float(np.mean(primary_success))
        result["agent_diag/success_rate"] = success_rate
        result["exact_diag/success_rate"] = success_rate
    return result


class AgentRunObserver:
    """Write common agentic metrics plus optional EXACT credit artifacts."""

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        conservation_tolerance: float = 1e-8,
        residual_warning_ratio: float = 0.95,
        ppo_kl_warning: float = 0.1,
        clipfrac_warning: float = 0.3,
        grad_norm_warning: float = 100.0,
        response_clip_warning_ratio: float = 0.1,
        invalid_step_warning_ratio: float = 0.2,
        initial_env_steps: int = 0,
        initial_generated_tokens: int = 0,
        initial_active_gpu_hours: float = 0.0,
        initial_step: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.conservation_tolerance = float(conservation_tolerance)
        self.residual_warning_ratio = float(residual_warning_ratio)
        self.ppo_kl_warning = float(ppo_kl_warning)
        self.clipfrac_warning = float(clipfrac_warning)
        self.grad_norm_warning = float(grad_norm_warning)
        self.response_clip_warning_ratio = float(response_clip_warning_ratio)
        self.invalid_step_warning_ratio = float(invalid_step_warning_ratio)
        self.cumulative_env_steps = int(initial_env_steps)
        self.cumulative_generated_tokens = int(initial_generated_tokens)
        self.cumulative_active_gpu_hours = float(initial_active_gpu_hours)
        self._write_heartbeat(
            status="initializing",
            step=initial_step,
            metrics={},
            warnings=[],
        )

    def _metric_warnings(self, metrics: Mapping[str, Any]) -> list[str]:
        warnings = []
        checks = (
            ("actor/ppo_kl", lambda value: abs(value) >= self.ppo_kl_warning, "high_ppo_kl"),
            ("actor/pg_clipfrac", lambda value: value >= self.clipfrac_warning, "high_policy_clipfrac"),
            ("actor/grad_norm", lambda value: value >= self.grad_norm_warning, "high_gradient_norm"),
            (
                "response_length/clip_ratio",
                lambda value: value >= self.response_clip_warning_ratio,
                "response_length_clipping",
            ),
            (
                "agent_diag/invalid_step_rate",
                lambda value: value >= self.invalid_step_warning_ratio,
                "high_invalid_step_rate",
            ),
        )
        for key, predicate, warning in checks:
            if key in metrics and predicate(float(metrics[key])):
                warnings.append(warning)
        if float(metrics.get("exact/residual_ratio_mean", 0.0)) >= self.residual_warning_ratio:
            warnings.append("high_residual_ratio")
        if float(metrics.get("exact/schema_fallback_rate", 0.0)) > 0:
            warnings.append("opaque_schema_fallback")
        if float(metrics.get("exact/appworld_factor_opaque_rate", 0.0)) > 0:
            warnings.append("partial_appworld_factor_graph")
        if "exact/appworld_version_supported" in metrics and float(metrics["exact/appworld_version_supported"]) < 1:
            warnings.append("unsupported_appworld_graph_version")
        if "exact/appworld_source_supported" in metrics and float(metrics["exact/appworld_source_supported"]) < 1:
            warnings.append("unsupported_appworld_graph_source")
        if float(metrics.get("exact/appworld_factor_compile_fallback_rate", 0.0)) > 0:
            warnings.append("appworld_factor_compile_failure")
        return warnings

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
        failure: Mapping[str, Any] | None = None,
    ) -> None:
        payload = {
            "status": status,
            "step": int(step),
            "updated_unix": time.time(),
            "cumulative_env_steps": self.cumulative_env_steps,
            "cumulative_generated_tokens": self.cumulative_generated_tokens,
            "cumulative_active_gpu_hours": self.cumulative_active_gpu_hours,
            "warnings": list(warnings),
            "metrics": dict(metrics),
        }
        if failure is not None:
            payload["failure"] = dict(failure)
        self._atomic_json(self.output_dir / "heartbeat.json", payload)

    def observe_credit(
        self,
        step: int,
        metrics: Mapping[str, float],
        traces: Sequence[Mapping[str, Any]],
        rollout_records: Sequence[Mapping[str, Any]],
        trajectory_diagnostics: Sequence[Mapping[str, Any]] = (),
    ) -> list[str]:
        metrics = self._validated_metrics(metrics, step=step)
        conservation_error = float(metrics["exact/conservation_error_max"])
        warnings = self._metric_warnings(metrics)

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
        for diagnostic in trajectory_diagnostics:
            self._append_jsonl(
                self.output_dir / "trajectory_diagnostics.jsonl",
                {"step": int(step), "recorded_unix": time.time(), **dict(diagnostic)},
            )

        self._write_heartbeat(status="credit_checked", step=step, metrics=metrics, warnings=warnings)
        if conservation_error > self.conservation_tolerance:
            reason = f"EXACT conservation error {conservation_error:.3e} exceeded {self.conservation_tolerance:.3e}"
            self._write_heartbeat(
                status="safety_stopped",
                step=step,
                metrics=metrics,
                warnings=[*warnings, reason],
            )
            raise ExactSafetyStop(reason)
        return warnings

    def observe_rollouts(
        self,
        step: int,
        metrics: Mapping[str, Any],
        rollout_records: Sequence[Mapping[str, Any]],
        trajectory_diagnostics: Sequence[Mapping[str, Any]],
    ) -> list[str]:
        """Persist estimator-independent trajectory evidence for a baseline."""

        metrics = self._validated_metrics(metrics, step=step)
        warnings = self._metric_warnings(metrics)
        for record in rollout_records:
            self._append_jsonl(
                self.output_dir / "rollout_samples.jsonl",
                {"step": int(step), "recorded_unix": time.time(), **dict(record)},
            )
        for diagnostic in trajectory_diagnostics:
            self._append_jsonl(
                self.output_dir / "trajectory_diagnostics.jsonl",
                {"step": int(step), "recorded_unix": time.time(), **dict(diagnostic)},
            )
        self._write_heartbeat(
            status="rollout_checked",
            step=step,
            metrics=metrics,
            warnings=warnings,
        )
        return warnings

    def observe_validation(self, step: int, metrics: Mapping[str, Any]) -> None:
        """Persist validation separately so step-zero baselines remain visible."""

        metrics = self._validated_metrics(metrics, step=step)
        self._append_jsonl(
            self.output_dir / "validation_metrics.jsonl",
            {"step": int(step), "recorded_unix": time.time(), "metrics": dict(metrics)},
        )
        self._write_heartbeat(
            status="validated",
            step=step,
            metrics=metrics,
            warnings=self._metric_warnings(metrics),
        )

    def complete_step(self, step: int, metrics: Mapping[str, Any], warnings: Sequence[str]) -> None:
        metrics = self._validated_metrics(metrics, step=step)
        if "training/cumulative_env_steps" in metrics:
            self.cumulative_env_steps = int(float(metrics["training/cumulative_env_steps"]))
        else:
            self.cumulative_env_steps += int(float(metrics.get("exact/env_step_count", 0)))
        if "training/cumulative_generated_tokens" in metrics:
            self.cumulative_generated_tokens = int(float(metrics["training/cumulative_generated_tokens"]))
        else:
            self.cumulative_generated_tokens += int(float(metrics.get("exact/generated_token_count", 0)))
        if "training/cumulative_active_gpu_hours" in metrics:
            self.cumulative_active_gpu_hours = float(metrics["training/cumulative_active_gpu_hours"])
        payload = {"step": int(step), "recorded_unix": time.time(), "metrics": dict(metrics)}
        self._append_jsonl(self.output_dir / "metrics.jsonl", payload)
        final_warnings = list(dict.fromkeys([*warnings, *self._metric_warnings(metrics)]))
        if final_warnings:
            self._append_jsonl(
                self.output_dir / "alerts.jsonl",
                {
                    "step": int(step),
                    "recorded_unix": time.time(),
                    "warnings": final_warnings,
                    "metrics": dict(metrics),
                },
            )
        self._write_heartbeat(status="running", step=step, metrics=metrics, warnings=final_warnings)

    def mark_completed(self, step: int, metrics: Mapping[str, Any]) -> None:
        metrics = self._validated_metrics(metrics, step=step)
        self._write_heartbeat(
            status="completed",
            step=step,
            metrics=metrics,
            warnings=self._metric_warnings(metrics),
        )

    def mark_budget_reached(self, step: int, metrics: Mapping[str, Any]) -> None:
        metrics = self._validated_metrics(metrics, step=step)
        self._write_heartbeat(
            status="budget_reached",
            step=step,
            metrics=metrics,
            warnings=self._metric_warnings(metrics),
        )

    def mark_failed(self, step: int, error: BaseException) -> None:
        """Persist a bounded, redacted trainer failure without hiding the exception."""

        failure = {
            "type": type(error).__name__,
            "message": redact_rollout_text(str(error), max_chars=2_000),
        }
        self._write_heartbeat(
            status="failed",
            step=step,
            metrics={},
            warnings=["trainer_exception"],
            failure=failure,
        )


def build_rollout_records(
    tokenizer: Any,
    batch: Any,
    max_records: int = 8,
    trajectory_diagnostics: Sequence[Mapping[str, Any]] = (),
    redact_text: bool = True,
    max_text_chars: int = 20_000,
) -> list[dict[str, Any]]:
    """Select deterministic diagnostic strata and decode one representative row each."""

    padding = _padding_mask(batch)
    step_ids = _step_ids(batch)
    candidates = np.flatnonzero(~padding)
    if candidates.size == 0:
        return []
    rewards = np.asarray(batch.non_tensor_batch["episode_rewards"], dtype=np.float64)
    advantage_strength = batch.batch["advantages"].detach().abs().sum(dim=-1).cpu().numpy()

    diagnostics = {str(item["trajectory_id"]): dict(item) for item in trajectory_diagnostics}
    trajectory_rows: dict[str, list[int]] = {}
    for row in candidates:
        trajectory_rows.setdefault(str(batch.non_tensor_batch["traj_uid"][row]), []).append(int(row))
    representative = {}
    for trajectory_id, rows in trajectory_rows.items():
        rows.sort(key=lambda row: int(step_ids[row]))
        diagnostic = diagnostics.get(trajectory_id, {})
        invalid_steps = set(diagnostic.get("invalid_step_ids", ()))
        representative[trajectory_id] = next(
            (row for row in rows if int(step_ids[row]) in invalid_steps),
            rows[-1],
        )

    trajectory_ids = list(trajectory_rows)
    priority_ids = [
        min(trajectory_ids, key=lambda item: rewards[trajectory_rows[item][-1]]),
        max(trajectory_ids, key=lambda item: rewards[trajectory_rows[item][-1]]),
        max(trajectory_ids, key=lambda item: max(advantage_strength[trajectory_rows[item]])),
    ]
    for tag in (
        "invalid_action",
        "max_steps",
        "no_factor_progress",
        "repeated_action",
        "high_residual_ratio",
    ):
        matching = [item for item in trajectory_ids if tag in diagnostics.get(item, {}).get("diagnostic_tags", ())]
        if matching:
            priority_ids.append(matching[0])
    priority_ids.extend(trajectory_ids)
    selected_ids = []
    for trajectory_id in priority_ids:
        if trajectory_id not in selected_ids:
            selected_ids.append(trajectory_id)
        if len(selected_ids) >= max_records:
            break

    records = []
    for trajectory_id in selected_ids:
        row = representative[trajectory_id]
        prompt = tokenizer.decode(batch.batch["prompts"][row], skip_special_tokens=True)
        response = tokenizer.decode(batch.batch["responses"][row], skip_special_tokens=True)
        if redact_text:
            prompt = redact_rollout_text(prompt, max_chars=max_text_chars)
            response = redact_rollout_text(response, max_chars=max_text_chars)
        records.append(
            {
                "row": row,
                "trajectory_id": trajectory_id,
                "step_id": int(step_ids[row]),
                "episode_return": float(rewards[row]),
                "is_action_valid": bool(batch.non_tensor_batch.get("is_action_valid", np.ones(len(batch), dtype=bool))[row]),
                "credit_token_sum": float(batch.batch["advantages"][row].sum().item()),
                "diagnostic_tags": diagnostics.get(trajectory_id, {}).get("diagnostic_tags", []),
                "termination": diagnostics.get(trajectory_id, {}).get("termination", "unknown"),
                "prompt": prompt,
                "response": response,
            }
        )
    return records


# Backward-compatible import for existing EXACT-only callers.
ExactObserver = AgentRunObserver
