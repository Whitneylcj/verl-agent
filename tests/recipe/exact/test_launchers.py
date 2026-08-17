from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_EXACT = REPO_ROOT / "examples" / "exact_trainer" / "run_exact.sh"
LAUNCH_MANAGED = REPO_ROOT / "examples" / "exact_trainer" / "launch_managed.sh"


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


def _preflight(environment: str, **extra_env: str) -> str:
    env = os.environ.copy()
    env.update({"PREFLIGHT_ONLY": "1", "MODEL_PATH": "fixture/model", **extra_env})
    result = subprocess.run(
        ["bash", str(RUN_EXACT), environment],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.mark.parametrize("environment", ("sokoban", "alfworld", "webshop"))
def test_stage_two_environments_default_to_exact_t(environment: str) -> None:
    assert _run_dir(environment).startswith("exact_temporal_")


def test_appworld_defaults_to_exact_g() -> None:
    assert _run_dir("appworld").startswith("exact_graph_")


def test_exact_mode_override_is_preserved() -> None:
    assert _run_dir("sokoban", EXACT_MODE="graph_cv").startswith("exact_graph_cv_")


def test_prepared_data_paths_are_bound_to_requested_sizes() -> None:
    pytest.importorskip("hydra")
    config = _preflight("sokoban", TRAIN_SIZE="2", VALIDATION_SIZE="3")
    assert "train_files: /root/autodl-tmp/data/verl-agent/train2_val3/text/train.parquet" in config
    assert "val_files: /root/autodl-tmp/data/verl-agent/train2_val3/text/test.parquet" in config


def test_nvidia_pilot_defaults_to_flash_attention() -> None:
    launcher = RUN_EXACT.read_text()
    assert "VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}" in launcher


def test_managed_launcher_detaches_without_waiting_for_training() -> None:
    launcher = LAUNCH_MANAGED.read_text()
    assert 'screen -dmS "${session_name}"' in launcher
    assert 'screen -DmS "${session_name}"' not in launcher


def test_appworld_checks_required_service_capacity_before_training() -> None:
    launcher = RUN_EXACT.read_text()
    assert "required_appworld_services=$((train_size * group_size + validation_size))" in launcher
    assert "python3 -m recipe.exact.check_appworld_services" in launcher
