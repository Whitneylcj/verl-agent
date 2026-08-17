import json

from recipe.exact.inspect_run import build_run_report


def _append(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_inspect_run_reports_metric_trends_alerts_and_rollouts(tmp_path):
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    (monitor / "heartbeat.json").write_text(json.dumps({"status": "running", "step": 2, "warnings": ["high_ppo_kl"]}))
    _append(
        monitor / "metrics.jsonl",
        {
            "step": 1,
            "metrics": {
                "training/global_step": 1,
                "actor/ppo_kl": 0.01,
                "agent_diag/invalid_step_rate": 0.5,
            },
        },
    )
    _append(
        monitor / "metrics.jsonl",
        {"step": 2, "metrics": {"training/global_step": 2, "actor/ppo_kl": 0.2}},
    )
    _append(
        monitor / "trajectory_diagnostics.jsonl",
        {"step": 2, "trajectory_id": "t1", "diagnostic_tags": ["invalid_action", "max_steps"]},
    )
    _append(
        monitor / "alerts.jsonl",
        {"step": 2, "warnings": ["high_ppo_kl"]},
    )
    _append(
        monitor / "validation_metrics.jsonl",
        {"step": 2, "metrics": {"val/success_rate": 0.25}},
    )
    _append(
        monitor / "rollout_samples.jsonl",
        {"step": 2, "trajectory_id": "t1", "prompt": "task", "response": "bad action"},
    )

    report = build_run_report(tmp_path, window=2, rollout_count=1)
    assert report["latest_step"] == 2
    assert report["trends"]["actor/ppo_kl"]["window_delta"] == 0.19
    assert report["trends"]["agent_diag/invalid_step_rate"]["latest"] == 0.5
    assert report["diagnostic_tag_counts"] == {"invalid_action": 1, "max_steps": 1}
    assert report["recent_alerts"][0]["warnings"] == ["high_ppo_kl"]
    assert report["latest_validation"]["metrics"]["val/success_rate"] == 0.25
    assert report["rollout_samples"][0]["response"] == "bad action"


def test_inspect_run_handles_initialized_but_empty_monitor(tmp_path):
    (tmp_path / "heartbeat.json").write_text(json.dumps({"status": "initializing", "step": 0, "warnings": []}))
    report = build_run_report(tmp_path)
    assert report["latest_step"] is None
    assert report["trends"] == {}
