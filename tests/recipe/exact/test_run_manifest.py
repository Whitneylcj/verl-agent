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
