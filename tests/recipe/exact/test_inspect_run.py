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
                "agent_diag/repeated_action_rate": 0.25,
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
    diagnosis_codes = {diagnosis["code"] for diagnosis in report["diagnoses"]}
    assert {"high_invalid_action_rate", "policy_action_loop", "large_policy_update"} <= diagnosis_codes


def test_inspect_run_handles_initialized_but_empty_monitor(tmp_path):
    (tmp_path / "heartbeat.json").write_text(json.dumps({"status": "initializing", "step": 0, "warnings": []}))
    report = build_run_report(tmp_path)
    assert report["run_state"] == "initializing"
    assert report["latest_step"] is None
    assert report["trends"] == {}
    assert report["diagnoses"] == []


def test_inspect_run_diagnoses_appworld_graph_fallbacks(tmp_path):
    _append(
        tmp_path / "metrics.jsonl",
        {
            "step": 1,
            "metrics": {
                "exact/appworld_factor_opaque_rate": 0.2,
                "exact/appworld_version_supported": 0.0,
                "exact/appworld_source_supported": 0.0,
                "exact/appworld_factor_compile_fallback_rate": 1.0,
            },
        },
    )
    report = build_run_report(tmp_path)
    codes = {diagnosis["code"] for diagnosis in report["diagnoses"]}
    assert {
        "partial_appworld_factor_graph",
        "unsupported_appworld_graph_version",
        "unsupported_appworld_graph_source",
        "appworld_factor_compile_failure",
    } <= codes


def test_inspect_run_reports_manifest_before_heartbeat(tmp_path):
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    (tmp_path / "run_manifest.json").write_text(json.dumps({"experiment": {"name": "pilot"}}))

    report = build_run_report(tmp_path)
    assert report["run_state"] == "launched_no_heartbeat"
    assert report["manifest"]["experiment"]["name"] == "pilot"
    assert report["resolved_config_present"] is False
    assert report["diagnoses"] == [
        {
            "code": "startup_incomplete",
            "severity": "high",
            "evidence": {"run_state": "launched_no_heartbeat"},
            "next_checks": [
                "inspect the persisted console log",
                "verify model/runtime initialization before changing training hyperparameters",
            ],
        }
    ]
