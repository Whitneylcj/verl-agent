from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_EXACT = REPO_ROOT / "examples" / "exact_trainer" / "run_exact.sh"


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


@pytest.mark.parametrize("environment", ("sokoban", "alfworld", "webshop"))
def test_stage_two_environments_default_to_exact_t(environment: str) -> None:
    assert _run_dir(environment).startswith("exact_temporal_")


def test_appworld_defaults_to_exact_g() -> None:
    assert _run_dir("appworld").startswith("exact_graph_")


def test_exact_mode_override_is_preserved() -> None:
    assert _run_dir("sokoban", EXACT_MODE="graph_cv").startswith("exact_graph_cv_")
