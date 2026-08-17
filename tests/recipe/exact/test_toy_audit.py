import json

import recipe.exact.toy_audit as toy_audit
from recipe.exact.toy_audit import run_toy_audit, verify_toy_audit


def test_stage_zero_toy_audit_passes_all_paper_gates():
    report = run_toy_audit(max_distractors=4, monte_carlo_samples=50_000)

    assert report["status"] == "pass"
    assert all(report["checks"].values())
    assert report["maximum_conservation_error"] < 1e-10
    assert report["graph_corruption"]["false_edge_curve"][-1]["max_abs_bias"] < 1e-10
    assert report["graph_corruption"]["deleted_edge_curve"][-1]["max_abs_bias"] > 1e-3
    assert report["dynamic_mask_counterexample"]["estimated_gradient"] == 0.24

    scaling = report["temporal_distractor_scaling"]
    assert scaling[-1]["outcome"]["trace_variance"] > scaling[0]["outcome"]["trace_variance"]
    assert scaling[-1]["exact_g"]["trace_variance"] == scaling[0]["exact_g"]["trace_variance"]


def test_saved_audit_is_bound_to_current_clean_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(
        toy_audit,
        "_git_state",
        lambda: {"commit": "current", "tracked_clean": True, "tracked_status": []},
    )
    report = run_toy_audit(max_distractors=2, monte_carlo_samples=50_000)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    verification = verify_toy_audit(path)
    assert verification["status"] == "pass"
    assert verification["errors"] == []

    report["git"]["commit"] = "stale"
    path.write_text(json.dumps(report), encoding="utf-8")
    verification = verify_toy_audit(path)
    assert verification["status"] == "no_go"
    assert "toy audit commit does not match the current checkout" in verification["errors"]
