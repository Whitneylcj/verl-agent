"""Thin trainer integration for EXACT credit and agent-run diagnostics."""

from __future__ import annotations

from pprint import pprint
from typing import Any, Mapping, Sequence

import numpy as np

from recipe.exact.advantage import fit_delayed_alpha_from_traces
from recipe.exact.core_exact import DelayedAlphaController
from recipe.exact.monitor import (
    AgentRunObserver,
    ExactSafetyStop,
    build_rollout_records,
    build_trajectory_diagnostics,
    redact_rollout_text,
    summarize_appworld_validation_actions,
    summarize_trajectory_diagnostics,
    summarize_validation_action_validity,
    summarize_validation_factor_progress,
)


def _estimator_name(config: Any) -> str:
    value = config.algorithm.adv_estimator
    return str(getattr(value, "value", value)).lower()


class ExactTrainerHooks:
    """Own EXACT control-variate state and optional durable monitoring."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.is_exact = _estimator_name(config) == "exact"
        self.monitor_config = config.algorithm.exact.monitor
        self.observer: AgentRunObserver | None = None
        self.alpha_controller: DelayedAlphaController | None = None
        if self.is_exact and config.algorithm.exact.mode == "graph_cv":
            environment_bucket = str(config.env.env_name).split("/")[0].lower()
            bucket_names = ("appworld.selector", "appworld.arguments") if environment_bucket == "appworld" else (environment_bucket,)
            cv_config = config.algorithm.exact.cv
            self.alpha_controller = DelayedAlphaController(
                bucket_names,
                ridge=cv_config.ridge,
                max_abs_alpha=cv_config.max_abs_alpha,
            )

    @property
    def active_alpha(self) -> Mapping[str, float] | None:
        return self.alpha_controller.active if self.alpha_controller is not None else None

    @property
    def requires_cv_checkpoint(self) -> bool:
        return self.alpha_controller is not None

    def cv_state_dict(self) -> dict[str, Any] | None:
        if self.alpha_controller is None:
            return None
        return dict(self.alpha_controller.state_dict())

    def load_cv_state_dict(self, state: Mapping[str, Any]) -> None:
        if self.alpha_controller is None:
            raise RuntimeError("received EXACT Graph-CV state without an active controller")
        self.alpha_controller.load_state_dict(state)

    def start_run(
        self,
        *,
        env_steps: int,
        generated_tokens: int,
        active_gpu_hours: float,
        step: int,
    ) -> None:
        if not self.monitor_config.enabled:
            return
        self.observer = AgentRunObserver(
            output_dir=self.monitor_config.output_dir,
            conservation_tolerance=self.config.algorithm.exact.conservation_tolerance,
            opaque_target_warning_ratio=self.monitor_config.opaque_target_warning_ratio,
            ppo_kl_warning=self.monitor_config.ppo_kl_warning,
            clipfrac_warning=self.monitor_config.clipfrac_warning,
            grad_norm_warning=self.monitor_config.grad_norm_warning,
            response_clip_warning_ratio=self.monitor_config.response_clip_warning_ratio,
            invalid_step_warning_ratio=self.monitor_config.invalid_step_warning_ratio,
            initial_env_steps=env_steps,
            initial_generated_tokens=generated_tokens,
            initial_active_gpu_hours=active_gpu_hours,
            initial_step=step,
        )

    def mark_failed(self, *, step: int, error: BaseException) -> None:
        if self.observer is None or isinstance(error, ExactSafetyStop):
            return
        try:
            self.observer.mark_failed(step=step, error=error)
        except Exception as monitor_error:
            pprint(f"Failed to persist trainer exception: {monitor_error}")

    def after_advantage(
        self,
        *,
        batch: Any,
        metrics: dict[str, Any],
        tokenizer: Any,
        max_steps: int,
        step: int,
    ) -> tuple[list[Mapping[str, Any]], list[str]]:
        metrics.update(batch.meta_info.pop("exact_metrics", {}))
        traces = batch.meta_info.pop("exact_traces", [])
        if self.alpha_controller is not None:
            pending_alpha = fit_delayed_alpha_from_traces(
                self.alpha_controller,
                traces,
                min_samples=self.config.algorithm.exact.cv.min_samples,
            )
            if pending_alpha is not None:
                for bucket, alpha in self.alpha_controller.advance().items():
                    metrics[f"exact/cv_alpha_next/{bucket}"] = float(alpha)

        if self.observer is None:
            return traces, []
        trajectory_diagnostics = build_trajectory_diagnostics(
            batch,
            traces,
            max_steps=max_steps,
            opaque_target_warning_ratio=self.monitor_config.opaque_target_warning_ratio,
            tokenizer=tokenizer,
        )
        metrics.update(summarize_trajectory_diagnostics(trajectory_diagnostics))
        rollout_records = build_rollout_records(
            tokenizer,
            batch,
            max_records=self.monitor_config.rollout_sample_count,
            trajectory_diagnostics=trajectory_diagnostics,
            redact_text=self.monitor_config.redact_text,
            max_text_chars=self.monitor_config.max_text_chars,
        )
        if self.is_exact:
            warnings = self.observer.observe_credit(
                step=step,
                metrics=metrics,
                traces=traces,
                rollout_records=rollout_records,
                trajectory_diagnostics=trajectory_diagnostics,
            )
        else:
            warnings = self.observer.observe_rollouts(
                step=step,
                metrics=metrics,
                rollout_records=rollout_records,
                trajectory_diagnostics=trajectory_diagnostics,
            )
        return traces, warnings

    def validation_metrics(
        self,
        *,
        action_validity: Mapping[str, Sequence[np.ndarray]],
        factor_snapshots: Mapping[str, Sequence[np.ndarray]],
        trajectory_ids: Sequence[Any],
        action_texts: Sequence[str],
        env_name: str,
    ) -> dict[str, float]:
        metrics = summarize_validation_action_validity(action_validity)
        metrics.update(
            summarize_validation_factor_progress(
                factor_snapshots["exact_factor_pre"],
                factor_snapshots["exact_factor_post"],
                trajectory_ids,
            )
        )
        execution_chunks = action_validity["is_action_execution_valid"]
        if "appworld" in env_name.lower() and execution_chunks:
            metrics.update(
                summarize_appworld_validation_actions(
                    action_texts,
                    trajectory_ids,
                    np.concatenate(execution_chunks, axis=0),
                )
            )
        return metrics

    def redact_texts(self, values: Sequence[Any]) -> list[str]:
        if not self.monitor_config.redact_text:
            return [str(value) for value in values]
        return [redact_rollout_text(value, max_chars=self.monitor_config.max_text_chars) for value in values]

    def observe_validation(self, *, step: int, metrics: Mapping[str, Any]) -> None:
        if self.observer is not None:
            self.observer.observe_validation(step=step, metrics=metrics)

    def observe_update(self, *, step: int, metrics: Mapping[str, Any], warnings: Sequence[str]) -> list[str]:
        if self.observer is None:
            return list(warnings)
        return self.observer.observe_update(step=step, metrics=metrics, warnings=warnings)

    def complete_step(self, *, step: int, metrics: Mapping[str, Any], warnings: Sequence[str]) -> None:
        if self.observer is not None:
            self.observer.complete_step(step=step, metrics=metrics, warnings=warnings)

    def mark_completed(self, *, step: int, metrics: Mapping[str, Any]) -> None:
        if self.observer is not None:
            self.observer.mark_completed(step=step, metrics=metrics)

    def mark_budget_reached(self, *, step: int, metrics: Mapping[str, Any]) -> None:
        if self.observer is not None:
            self.observer.mark_budget_reached(step=step, metrics=metrics)
