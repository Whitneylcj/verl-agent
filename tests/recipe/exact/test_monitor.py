import gzip
import json

import numpy as np
import torch

from recipe.exact.monitor import (
    ExactObserver,
    ExactSafetyStop,
    build_rollout_records,
    build_trajectory_diagnostics,
    redact_rollout_text,
    summarize_trajectory_diagnostics,
)


class _Batch:
    def __init__(self, tensors, non_tensors):
        self.batch = tensors
        self.non_tensor_batch = non_tensors

    def __len__(self):
        return len(self.batch["responses"])


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(value)) for value in token_ids)


def _snapshot(values):
    return {"factor_ids": ("progress",), "values": values}


def _diagnostic_batch():
    return _Batch(
        {
            "prompts": torch.tensor([[101, 102], [103, 104], [105, 106]]),
            "responses": torch.tensor([[1, 2], [1, 2], [3, 0]]),
            "response_mask": torch.tensor([[1, 1], [1, 1], [1, 0]]),
            "advantages": torch.tensor([[1.0, 1.0], [0.5, 0.5], [-1.0, 0.0]]),
        },
        {
            "traj_uid": np.array(["t1", "t1", "t2"], dtype=object),
            "exact_padding": np.array([False, False, False]),
            "exact_step_id": np.array([1, 2, 1]),
            "episode_rewards": np.array([0.0, 0.0, 1.0]),
            "episode_lengths": np.array([2, 2, 1]),
            "tool_callings": np.array([2, 2, 1]),
            "is_action_valid": np.array([True, False, True]),
            "episode_done": np.array([False, False, True]),
            "trajectory_outcomes": np.array(
                [{"success_rate": 0.0}, {"success_rate": 0.0}, {"success_rate": 1.0}],
                dtype=object,
            ),
            "exact_factor_pre": np.array(
                [_snapshot((0.0,)), _snapshot((0.0,)), _snapshot((0.0,))],
                dtype=object,
            ),
            "exact_factor_post": np.array(
                [_snapshot((0.0,)), _snapshot((0.0,)), _snapshot((1.0,))],
                dtype=object,
            ),
            "exact_effect_schema": np.array(
                [
                    {"opaque": False, "descendant_factor_ids": ("progress",)},
                    {"opaque": False, "descendant_factor_ids": ("progress",)},
                    {"opaque": False, "descendant_factor_ids": ("progress",)},
                ],
                dtype=object,
            ),
        },
    )


def _metrics(conservation=0.0):
    return {
        "exact/conservation_error_max": conservation,
        "exact/residual_ratio_mean": 0.2,
        "exact/schema_fallback_rate": 0.0,
        "exact/env_step_count": 3,
        "exact/generated_token_count": 11,
    }


def test_observer_writes_durable_step_artifacts(tmp_path):
    observer = ExactObserver(tmp_path)
    warnings = observer.observe_credit(
        step=1,
        metrics=_metrics(),
        traces=[{"trajectory_id": "t1"}],
        rollout_records=[{"response": "move up"}],
    )
    observer.complete_step(1, _metrics(), warnings)

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["status"] == "running"
    assert heartbeat["cumulative_env_steps"] == 3
    assert heartbeat["cumulative_generated_tokens"] == 11
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1
    with gzip.open(tmp_path / "credit_traces.jsonl.gz", "rt") as handle:
        assert json.loads(handle.readline())["traces"][0]["trajectory_id"] == "t1"


def test_observer_stops_before_update_on_conservation_failure(tmp_path):
    observer = ExactObserver(tmp_path, conservation_tolerance=1e-8)
    try:
        observer.observe_credit(2, _metrics(conservation=1e-4), [], [])
    except ExactSafetyStop:
        pass
    else:
        raise AssertionError("observer did not stop on a hard invariant failure")
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["status"] == "safety_stopped"


def test_observer_restores_budget_counters(tmp_path):
    observer = ExactObserver(
        tmp_path,
        initial_env_steps=17,
        initial_generated_tokens=101,
        initial_active_gpu_hours=1.25,
        initial_step=4,
    )
    observer.complete_step(5, _metrics(), [])

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["step"] == 5
    assert heartbeat["cumulative_env_steps"] == 20
    assert heartbeat["cumulative_generated_tokens"] == 112
    assert heartbeat["cumulative_active_gpu_hours"] == 1.25


def test_observer_safety_stops_on_nonfinite_metric(tmp_path):
    observer = ExactObserver(tmp_path)
    metrics = _metrics()
    metrics["exact/residual_ratio_mean"] = float("nan")

    try:
        observer.observe_credit(7, metrics, [], [])
    except ExactSafetyStop:
        pass
    else:
        raise AssertionError("observer did not stop on a non-finite metric")

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["status"] == "safety_stopped"
    assert heartbeat["step"] == 7


def test_trajectory_diagnostics_link_failure_signals_to_rollout_text():
    batch = _diagnostic_batch()
    traces = [
        {"trajectory_id": "t1", "residual_ratio": 0.99},
        {"trajectory_id": "t2", "residual_ratio": 0.1},
    ]
    diagnostics = build_trajectory_diagnostics(batch, traces, max_steps=2)
    by_id = {item["trajectory_id"]: item for item in diagnostics}
    assert by_id["t1"]["termination"] == "max_steps"
    assert set(by_id["t1"]["diagnostic_tags"]) >= {
        "invalid_action",
        "max_steps",
        "no_factor_progress",
        "repeated_action",
        "high_residual_ratio",
    }
    assert by_id["t2"]["termination"] == "success"

    metrics = summarize_trajectory_diagnostics(diagnostics)
    assert metrics["agent_diag/success_rate"] == 0.5
    assert metrics["agent_diag/invalid_step_rate"] == 1 / 3
    assert metrics["exact_diag/success_rate"] == 0.5
    assert metrics["exact_diag/invalid_step_rate"] == 1 / 3

    records = build_rollout_records(
        _Tokenizer(),
        batch,
        trajectory_diagnostics=diagnostics,
    )
    record_by_id = {item["trajectory_id"]: item for item in records}
    assert record_by_id["t1"]["step_id"] == 2
    assert "invalid_action" in record_by_id["t1"]["diagnostic_tags"]


def test_baseline_diagnostics_do_not_invent_exact_factor_signals():
    batch = _diagnostic_batch()
    batch.non_tensor_batch["agent_step_id"] = batch.non_tensor_batch.pop("exact_step_id")
    batch.non_tensor_batch["rollout_padding"] = batch.non_tensor_batch.pop("exact_padding")
    for key in ("exact_factor_pre", "exact_factor_post", "exact_effect_schema"):
        batch.non_tensor_batch.pop(key)

    diagnostics = build_trajectory_diagnostics(batch, [], max_steps=2)
    by_id = {item["trajectory_id"]: item for item in diagnostics}
    assert by_id["t1"]["factor_probe_available"] is False
    assert "no_factor_progress" not in by_id["t1"]["diagnostic_tags"]
    assert by_id["t1"]["residual_ratio"] is None

    metrics = summarize_trajectory_diagnostics(diagnostics)
    assert metrics["agent_diag/invalid_step_rate"] == 1 / 3
    assert "exact_diag/no_factor_progress_rate" not in metrics

    records = build_rollout_records(
        _Tokenizer(),
        batch,
        trajectory_diagnostics=diagnostics,
    )
    assert {record["step_id"] for record in records} <= {1, 2}


def test_rollout_redaction_bounds_text_and_removes_common_secrets():
    raw = "email jane@example.com phone +1 (555) 123-4567 password=abc123 " + "x" * 200
    redacted = redact_rollout_text(raw, max_chars=100)
    assert "jane@example.com" not in redacted
    assert "555" not in redacted
    assert "abc123" not in redacted
    assert "[TRUNCATED]" in redacted


def test_observer_emits_soft_alerts_after_optimizer_metrics(tmp_path):
    observer = ExactObserver(tmp_path, ppo_kl_warning=0.1)
    observer.complete_step(1, {**_metrics(), "actor/ppo_kl": 0.2}, [])

    alert = json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[0])
    assert "high_ppo_kl" in alert["warnings"]


def test_observer_alerts_on_appworld_graph_fallbacks(tmp_path):
    observer = ExactObserver(tmp_path)
    observer.complete_step(
        1,
        {
            **_metrics(),
            "exact/appworld_factor_opaque_rate": 0.25,
            "exact/appworld_version_supported": 0.0,
            "exact/appworld_source_supported": 0.0,
            "exact/appworld_factor_compile_fallback_rate": 1.0,
        },
        [],
    )

    alert = json.loads((tmp_path / "alerts.jsonl").read_text().splitlines()[0])
    assert set(alert["warnings"]) >= {
        "partial_appworld_factor_graph",
        "unsupported_appworld_graph_version",
        "unsupported_appworld_graph_source",
        "appworld_factor_compile_failure",
    }


def test_observer_writes_common_baseline_rollouts_and_syncs_budgets(tmp_path):
    observer = ExactObserver(tmp_path, invalid_step_warning_ratio=0.2)
    warnings = observer.observe_rollouts(
        step=1,
        metrics={"agent_diag/invalid_step_rate": 0.5},
        rollout_records=[{"trajectory_id": "t1", "response": "bad action"}],
        trajectory_diagnostics=[{"trajectory_id": "t1", "diagnostic_tags": ["invalid_action"]}],
    )
    assert warnings == ["high_invalid_step_rate"]

    observer.complete_step(
        1,
        {
            "training/cumulative_env_steps": 7,
            "training/cumulative_generated_tokens": 53,
            "training/cumulative_active_gpu_hours": 0.75,
        },
        warnings,
    )
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["cumulative_env_steps"] == 7
    assert heartbeat["cumulative_generated_tokens"] == 53
    assert heartbeat["cumulative_active_gpu_hours"] == 0.75
    assert (tmp_path / "rollout_samples.jsonl").exists()
    assert (tmp_path / "trajectory_diagnostics.jsonl").exists()


def test_observer_persists_step_zero_validation(tmp_path):
    observer = ExactObserver(tmp_path)
    observer.observe_validation(0, {"val/success_rate": np.float64(0.25)})

    record = json.loads((tmp_path / "validation_metrics.jsonl").read_text().splitlines()[0])
    assert record["step"] == 0
    assert record["metrics"]["val/success_rate"] == 0.25
    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["status"] == "validated"
