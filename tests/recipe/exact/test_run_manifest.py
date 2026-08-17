import hashlib
from pathlib import Path

from recipe.exact.run_manifest import build_manifest, redact_override, write_manifest


def _experiment(model_path="Qwen/Qwen2.5-1.5B-Instruct"):
    return {
        "name": "exact_sokoban_seed0",
        "environment": "sokoban",
        "algorithm": "exact",
        "exact_mode": "graph",
        "model_path": model_path,
        "seed": 0,
        "loss_agg_mode": "seq-mean-token-sum",
    }


def test_manifest_records_commit_overrides_and_rejects_mixed_resume(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    manifest = build_manifest(
        repo_root=repo_root,
        experiment=_experiment(),
        overrides=["trainer.max_env_steps=60", "service.password=do-not-save"],
        include_hardware=False,
    )
    assert len(manifest["git"]["commit"]) == 40
    assert manifest["hydra_overrides"][1] == "service.password=[REDACTED]"

    manifest_path = write_manifest(tmp_path / "run", manifest, require_clean=False)
    assert manifest_path.is_file()
    assert write_manifest(tmp_path / "run", manifest, require_clean=False, resume=True) == manifest_path

    changed = build_manifest(
        repo_root=repo_root,
        experiment=_experiment(model_path="Qwen/Qwen3-1.7B"),
        overrides=["trainer.max_env_steps=60", "service.password=do-not-save"],
        include_hardware=False,
    )
    try:
        write_manifest(tmp_path / "run", changed, require_clean=False, resume=True)
    except RuntimeError as error:
        assert "model_path" in str(error)
    else:
        raise AssertionError("manifest accepted a resume with a different model")


def test_manifest_refuses_nonempty_unidentified_directory(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "old.log").write_text("old")
    manifest = {"git": {"tracked_dirty": False}}

    try:
        write_manifest(output_dir, manifest)
    except FileExistsError as error:
        assert "not empty" in str(error)
    else:
        raise AssertionError("manifest accepted a nonempty unidentified run directory")


def test_override_redaction_only_hides_sensitive_keys():
    assert redact_override("model.path=Qwen/Qwen2.5") == "model.path=Qwen/Qwen2.5"
    assert redact_override("api_token=abc") == "api_token=[REDACTED]"


def test_manifest_hashes_the_verified_toy_audit(tmp_path, monkeypatch):
    audit_path = tmp_path / "toy-audit.json"
    payload = b'{"schema_version":"exact-toy-audit/v1","status":"pass","git":{"commit":"abc"}}\n'
    audit_path.write_bytes(payload)
    monkeypatch.setenv("EXACT_TOY_AUDIT_PATH", str(audit_path))

    manifest = build_manifest(
        repo_root=Path(__file__).resolve().parents[3],
        experiment=_experiment(),
        overrides=[],
        include_hardware=False,
    )

    assert manifest["preflight"]["toy_audit"] == {
        "path": str(audit_path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_version": "exact-toy-audit/v1",
        "status": "pass",
        "commit": "abc",
    }


def test_manifest_hashes_the_verified_appworld_schema_audit(tmp_path, monkeypatch):
    audit_path = tmp_path / "appworld-schema-audit.json"
    payload = b'{"schema_version":"exact.appworld.schema-audit.v1","passed":true,"git":{"commit":"def"},"appworld_version":"0.2.0.dev0","appworld_source_revision":"source","task_count":732,"factor_count":3660,"opaque_factor_rate":0.01}\n'
    audit_path.write_bytes(payload)
    monkeypatch.setenv("APPWORLD_SCHEMA_AUDIT_PATH", str(audit_path))

    manifest = build_manifest(
        repo_root=Path(__file__).resolve().parents[3],
        experiment=_experiment(),
        overrides=[],
        include_hardware=False,
    )

    assert manifest["preflight"]["appworld_schema_audit"] == {
        "path": str(audit_path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_version": "exact.appworld.schema-audit.v1",
        "passed": True,
        "commit": "def",
        "appworld_version": "0.2.0.dev0",
        "appworld_source_revision": "source",
        "task_count": 732,
        "factor_count": 3660,
        "opaque_factor_rate": 0.01,
    }


def test_manifest_hashes_the_verified_appworld_real_probe(tmp_path, monkeypatch):
    probe_path = tmp_path / "appworld-real-probe.json"
    payload = b'{"schema_version":"exact.appworld.real-probe.v1","status":"pass","git":{"commit":"ghi"},"appworld_version":"0.2.0.dev0","appworld_source_revision":"source","task_id":"task_1","action_valid":true,"resolution_fallback":false}\n'
    probe_path.write_bytes(payload)
    monkeypatch.setenv("APPWORLD_EXACT_G_PROBE_PATH", str(probe_path))

    manifest = build_manifest(
        repo_root=Path(__file__).resolve().parents[3],
        experiment=_experiment(),
        overrides=[],
        include_hardware=False,
    )

    assert manifest["preflight"]["appworld_real_probe"] == {
        "path": str(probe_path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema_version": "exact.appworld.real-probe.v1",
        "status": "pass",
        "commit": "ghi",
        "appworld_version": "0.2.0.dev0",
        "appworld_source_revision": "source",
        "task_id": "task_1",
        "action_valid": True,
        "resolution_fallback": False,
    }
