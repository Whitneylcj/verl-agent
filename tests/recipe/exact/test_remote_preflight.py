from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from agent_system.webshop_runtime import resolve_webshop_env_kwargs
from recipe.exact.check_appworld_services import load_port_manifest, require_reachable_services
from recipe.exact.prepare_model import _expected_hashes, discover_weight_files, validate_weight_hashes
from recipe.exact.prepare_prompts import build_prompt_rows
from recipe.exact.prepare_webshop import product_document, validate_source_files, write_documents

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_EXACT = REPO_ROOT / "examples" / "exact_trainer" / "run_exact.sh"
LAUNCH_MANAGED = REPO_ROOT / "examples" / "exact_trainer" / "launch_managed.sh"
PREPARE_MODEL = REPO_ROOT / "examples" / "exact_trainer" / "prepare_model.sh"
LAUNCH_PILOT = REPO_ROOT / "examples" / "exact_trainer" / "launch_pilot_stage.sh"


def _run_dir(environment: str, **extra_env: str) -> str:
    env = os.environ.copy()
    env.update({"RUN_DIR_ONLY": "1", "MODEL_PATH": "fixture/model", **extra_env})
    result = subprocess.run(
        ["bash", str(RUN_EXACT), environment],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).name


def _preflight(environment: str, *overrides: str, **extra_env: str) -> str:
    env = os.environ.copy()
    env.update({"PREFLIGHT_ONLY": "1", "MODEL_PATH": "fixture/model", **extra_env})
    result = subprocess.run(
        ["bash", str(RUN_EXACT), environment, *overrides],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize("environment", ("sokoban", "alfworld", "webshop"))
def test_stage_two_environments_default_to_exact_t(environment):
    assert _run_dir(environment).startswith("exact_temporal_")


def test_appworld_defaults_to_exact_g():
    assert _run_dir("appworld").startswith("exact_graph_")


def test_exact_mode_override_is_preserved():
    assert _run_dir("sokoban", EXACT_MODE="graph_cv").startswith("exact_graph_cv_")


def test_prompt_profile_is_explicit_and_environment_specific():
    pytest.importorskip("hydra")
    assert "prompt_profile: benchmark" in _preflight("alfworld")
    appworld_config = _preflight("appworld")
    assert "prompt_profile: appworld_exact_json" in appworld_config
    assert "action_mode: json_api" in appworld_config


def test_matched_baseline_can_use_the_same_explicit_profile():
    pytest.importorskip("hydra")
    config = _preflight("sokoban", ADV_ESTIMATOR="grpo", PROMPT_PROFILE="qwen_small_guided")
    assert "adv_estimator: grpo" in config
    assert "prompt_profile: qwen_small_guided" in config


def test_launcher_resolved_limits_and_paths():
    pytest.importorskip("hydra")
    appworld = _preflight("appworld")
    assert "max_prompt_length: 8192" in appworld
    sokoban = _preflight("sokoban", MAX_RESPONSE_LENGTH="384", TRAIN_SIZE="2", VALIDATION_SIZE="3")
    assert "max_response_length: 384" in sokoban
    assert "train2_val3/text/train.parquet" in sokoban
    assert "train2_val3/text/test.parquet" in sokoban


def test_preflight_parses_additional_hydra_overrides():
    pytest.importorskip("hydra")
    config = _preflight(
        "sokoban",
        "trainer.total_training_steps=7",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
    )
    assert "total_training_steps: 7" in config
    assert "ppo_mini_batch_size: 4" in config


@pytest.mark.parametrize("script", (RUN_EXACT, LAUNCH_MANAGED, LAUNCH_PILOT, PREPARE_MODEL))
def test_remote_launchers_pass_shell_syntax(script):
    subprocess.run(["bash", "-n", str(script)], cwd=REPO_ROOT, check=True)


def test_qwen3_sampling_overrides_reach_hydra_config():
    pytest.importorskip("hydra")
    config = _preflight("sokoban", MODEL_PATH="Qwen/Qwen3-1.7B")
    for expected in ("enable_thinking: false", "temperature: 0.7", "top_p: 0.8", "top_k: 20"):
        assert expected in config


def test_model_weight_discovery_and_hash_validation(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights")
    expected = hashlib.sha256(b"weights").hexdigest()
    assert discover_weight_files(tmp_path) == [weight]
    assert validate_weight_hashes([weight], {weight.name: expected}) == [{"name": weight.name, "bytes": 7, "sha256": expected}]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_weight_hashes([weight], {weight.name: "0" * 64})


def test_indexed_weights_require_every_shard(tmp_path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"layer.0": first.name, "layer.1": second.name}}))
    with pytest.raises(FileNotFoundError, match=second.name):
        discover_weight_files(tmp_path)
    second.write_bytes(b"second")
    assert discover_weight_files(tmp_path) == [first, second]


def test_expected_hashes_require_name_sha_pairs():
    assert _expected_hashes([f"model.safetensors={'a' * 64}"]) == {"model.safetensors": "a" * 64}
    with pytest.raises(ValueError, match="NAME=SHA256"):
        _expected_hashes(["missing-separator"])


def test_prompt_rows_have_agent_schema_and_exact_size():
    rows = build_prompt_rows(3, "train")
    assert len(rows) == 3
    assert rows[0] == {
        "data_source": "text",
        "prompt": [{"role": "user", "content": ""}],
        "ability": "agent",
        "extra_info": {"split": "train", "index": 0},
    }
    assert rows[-1]["extra_info"] == {"split": "train", "index": 2}


def _product():
    return {
        "asin": "A-1",
        "Title": "Blue Mug",
        "Description": "Ceramic cup",
        "BulletPoints": ["Dishwasher safe"],
        "options": {"size": ["small", "large"]},
    }


def test_webshop_documents_are_searchable_and_serializable(tmp_path):
    document = product_document(_product())
    assert document["id"] == "A-1"
    assert "blue mug" in document["contents"]
    assert "size: small, large" in document["contents"]
    destination = tmp_path / "resources_1k" / "documents.jsonl"
    assert write_documents([_product()], destination) == 1
    assert json.loads(destination.read_text())["id"] == "A-1"


def test_webshop_source_files_require_aligned_unique_products(tmp_path):
    products = [{"asin": f"A-{index}"} for index in range(1000)]
    attributes = {product["asin"]: {"attributes": []} for product in products}
    (tmp_path / "items_shuffle_1000.json").write_text(json.dumps(products))
    (tmp_path / "items_ins_v2_1000.json").write_text(json.dumps(attributes))
    (tmp_path / "items_human_ins.json").write_text(json.dumps({"A-0": [{"instruction": "find it"}]}))
    assert validate_source_files(tmp_path) == (1000, 1000, 1)
    products[-1]["asin"] = products[0]["asin"]
    (tmp_path / "items_shuffle_1000.json").write_text(json.dumps(products))
    with pytest.raises(ValueError, match="duplicate ASINs"):
        validate_source_files(tmp_path)


def test_webshop_runtime_selects_small_and_full_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBSHOP_DATA_ROOT", str(tmp_path))
    small = resolve_webshop_env_kwargs(use_small=True, human_goals=False)
    assert small["num_products"] == 1000
    assert Path(small["file_path"]).name == "items_shuffle_1000.json"
    full = resolve_webshop_env_kwargs(use_small=False, human_goals=True)
    assert full["num_products"] is None
    assert full["human_goals"] is True
    assert Path(full["file_path"]).name == "items_shuffle.json"


def test_appworld_service_manifest_and_reachability(tmp_path, monkeypatch):
    duplicate = tmp_path / "duplicate.ports"
    duplicate.write_text("8200\n8200\n")
    with pytest.raises(ValueError, match="duplicates"):
        load_port_manifest(duplicate)

    connected = []

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_create_connection(address, timeout):
        connected.append((address, timeout))
        return _Connection()

    monkeypatch.setattr("recipe.exact.check_appworld_services.socket.create_connection", fake_create_connection)
    manifest = tmp_path / "appworld.ports"
    manifest.write_text("8200\n")
    assert require_reachable_services(manifest, required=1, timeout=0.1) == [8200]
    assert connected == [(("127.0.0.1", 8200), 0.1)]
    with pytest.raises(RuntimeError, match="needs 2 services"):
        require_reachable_services(manifest, required=2, timeout=0.1)
