"""Dependency-light unit tests of real environment methods, without Ray/CUDA.

Load the method ASTs to avoid importing the optional GPU/environment stack.
Remote tests still need to validate the imported modules and real TextWorld.
"""

import ast
import os
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[3]
ENV_PATH = "agent_system/environments/env_package/alfworld/"


def load_class(path, name, namespace, methods=None):
    source = ast.parse((ROOT / path).read_text())
    node = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == name)
    if methods is not None:
        node.body = [node for node in node.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    node.bases = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("signal", ["planner", "predicates"])
def test_requested_infos_keep_outcomes_and_avoid_unused_planning(enabled, signal):
    registered = {}

    def register_games(files, infos, **kwargs):
        registered.update(infos=infos, **kwargs)
        return "test-env"

    textworld = SimpleNamespace(
        EnvInfos=SimpleNamespace,
        gym=SimpleNamespace(register_games=register_games, make=lambda value: value),
    )
    cls = load_class(
        ENV_PATH + "alfworld/agents/environment/alfred_tw_env.py",
        "AlfredTWEnv",
        {
            "textworld": textworld,
            "AlfredDemangler": SimpleNamespace,
            "AlfredInfos": object,
            "AlfredExactInstrumentation": SimpleNamespace,
        },
        methods={"init_env"},
    )
    env = cls()
    env.config = {
        "env": {"domain_randomization": False, "expert_type": "handcoded"},
        "general": {"training_method": "dagger"},
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
        "exact": {"enabled": enabled, "signal": signal, "horizon": 30},
    }
    env.train_eval, env.use_expert, env.game_files = "train", False, ["game"]
    assert env.init_env(1) == "test-env"
    infos = registered["infos"]
    assert infos.won and infos.lost and infos.admissible_commands
    assert infos.policy_commands == infos.intermediate_reward == infos.facts == (enabled and signal == "planner")
    assert ("exact" in infos.extras) == (signal == "predicates")
    assert registered["max_episode_steps"] == 50


def test_baseline_worker_preserves_transition_without_credit_fields():
    reset_result = (["initial"], {"won": [False], "lost": [False]})
    step_result = (["terminal"], [1.0], [True], {"won": [True], "lost": [False]})
    batch = SimpleNamespace(seed=lambda seed: None, reset=lambda: reset_result, step=lambda actions: step_result)
    base = SimpleNamespace(init_env=lambda batch_size: batch)
    cls = load_class(ENV_PATH + "envs.py", "AlfworldWorker", {"np": np})
    worker = cls({"exact": {"enabled": False}}, 17, base)
    assert worker.reset() == reset_result
    assert worker.step("look") == step_result
    with pytest.raises(RuntimeError, match="probe requested before reset"):
        worker.exact_credit_snapshot()


def test_exact_planner_worker_still_requires_credit_fields():
    batch = SimpleNamespace(seed=lambda seed: None, reset=lambda: (["initial"], {"won": [False]}))
    base = SimpleNamespace(init_env=lambda batch_size: batch)
    cls = load_class(ENV_PATH + "envs.py", "AlfworldWorker", {"np": np})
    worker = cls({"exact": {"enabled": True}}, 17, base)
    with pytest.raises(RuntimeError, match="info field: intermediate_reward"):
        worker.reset()


@pytest.mark.parametrize(
    ("algorithm", "signal", "enabled"),
    [("grpo", "planner", False), ("gigpo", "planner", False), ("exact", "planner", True), ("AdvantageEstimator.EXACT", "planner", True), ("grpo", "predicates", True)],
)
def test_manager_selects_instrumentation_for_training_and_validation(monkeypatch, algorithm, signal, enabled):
    path = "agent_system/environments/env_manager.py"
    tree = ast.parse((ROOT / path).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "make_envs")

    class Config(dict):
        __getattr__ = dict.__getitem__

    config = Config(
        algorithm=Config(adv_estimator=algorithm),
        data=Config(train_batch_size=16, val_batch_size=8),
        env=Config(
            env_name="alfworld/AlfredTWEnv",
            rollout=Config(n=8),
            resources_per_worker={},
            seed=0,
            max_steps=30,
            alfworld=Config(exact_signal=signal, eval_dataset="eval_in_distribution"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "agent_system.environments.env_package.alfworld",
        SimpleNamespace(alfworld_projection=lambda: None, build_alfworld_envs=lambda *args, **kwargs: kwargs["env_kwargs"]),
    )
    namespace = {
        "os": os,
        "__file__": str(ROOT / path),
        "partial": partial,
        "OmegaConf": SimpleNamespace(to_container=lambda value, **kwargs: value),
        "AlfWorldEnvironmentManager": lambda envs, *args: envs,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), path, "exec"), namespace)
    for kwargs in namespace["make_envs"](config):
        assert kwargs["exact_enabled"] is enabled
        assert kwargs["exact_signal"] == signal
