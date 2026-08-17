import gzip
import json

from recipe.exact.monitor import ExactObserver, ExactSafetyStop


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
        initial_step=4,
    )
    observer.complete_step(5, _metrics(), [])

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["step"] == 5
    assert heartbeat["cumulative_env_steps"] == 20
    assert heartbeat["cumulative_generated_tokens"] == 112


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
