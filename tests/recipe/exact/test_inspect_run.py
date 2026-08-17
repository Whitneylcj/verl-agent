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
                "agent_diag/syntax_invalid_step_rate": 0.25,
                "agent_diag/execution_error_step_rate": 0.4,
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
    assert {
        "high_invalid_action_rate",
        "high_action_syntax_error_rate",
        "appworld_api_execution_errors",
        "policy_action_loop",
        "large_policy_update",
    } <= diagnosis_codes


def test_inspect_run_handles_initialized_but_empty_monitor(tmp_path):
    (tmp_path / "heartbeat.json").write_text(json.dumps({"status": "initializing", "step": 0, "warnings": []}))
    report = build_run_report(tmp_path)
    assert report["run_state"] == "initializing"
    assert report["latest_step"] is None
    assert report["trends"] == {}
    assert report["diagnoses"] == []


def test_inspect_run_diagnoses_validation_only_action_failures(tmp_path):
    _append(
        tmp_path / "validation_metrics.jsonl",
        {
            "step": 0,
            "metrics": {
                "val/agent_diag/invalid_step_rate": 0.75,
                "val/agent_diag/syntax_invalid_step_rate": 0.1,
                "val/agent_diag/execution_error_step_rate": 0.7,
                "val/agent_diag/appworld_application_api_step_rate": 0.25,
                "val/agent_diag/appworld_successful_application_api_step_rate": 0.0,
                "val/agent_diag/appworld_completion_before_application_success_trajectory_rate": 0.5,
            },
        },
    )

    report = build_run_report(tmp_path)
    diagnoses = {item["code"]: item for item in report["diagnoses"]}
    assert diagnoses["high_invalid_action_rate"]["evidence"] == {"val/agent_diag/invalid_step_rate": 0.75}
    assert diagnoses["appworld_api_execution_errors"]["evidence"] == {"val/agent_diag/execution_error_step_rate": 0.7}
    assert diagnoses["appworld_no_successful_application_calls"]["evidence"] == {
        "val/agent_diag/appworld_application_api_step_rate": 0.25,
        "val/agent_diag/appworld_successful_application_api_step_rate": 0.0,
    }
    assert diagnoses["appworld_completion_before_application_success"]["evidence"] == {"val/agent_diag/appworld_completion_before_application_success_trajectory_rate": 0.5}
    assert "high_action_syntax_error_rate" not in diagnoses


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


def test_inspect_run_allows_recent_startup_without_heartbeat(tmp_path):
    monitor = tmp_path / "monitor"
    monitor.mkdir()
    (tmp_path / "run_manifest.json").write_text(json.dumps({"created_unix": 90.0, "experiment": {"name": "pilot"}}))

    report = build_run_report(
        tmp_path,
        stale_after_seconds=30.0,
        now_unix=100.0,
    )

    assert report["run_state"] == "launched_no_heartbeat"
    assert report["run_age_seconds"] == 10.0
    assert report["diagnoses"] == []


def test_inspect_run_diagnoses_redacted_trainer_failure(tmp_path):
    (tmp_path / "heartbeat.json").write_text(
        json.dumps(
            {
                "status": "failed",
                "step": 3,
                "warnings": ["trainer_exception"],
                "failure": {"type": "RuntimeError", "message": "worker exited"},
                "updated_unix": 90.0,
            }
        )
    )

    report = build_run_report(tmp_path, now_unix=100.0)
    assert report["heartbeat_age_seconds"] == 10.0
    assert report["diagnoses"][0]["code"] == "trainer_failed"
    assert report["diagnoses"][0]["evidence"]["failure_type"] == "RuntimeError"


def test_inspect_run_diagnoses_stale_nonterminal_heartbeat(tmp_path):
    (tmp_path / "heartbeat.json").write_text(
        json.dumps(
            {
                "status": "running",
                "step": 2,
                "warnings": [],
                "updated_unix": 100.0,
            }
        )
    )

    report = build_run_report(
        tmp_path,
        stale_after_seconds=30.0,
        now_unix=145.0,
    )
    assert report["heartbeat_age_seconds"] == 45.0
    assert report["diagnoses"][0]["code"] == "heartbeat_stale"
