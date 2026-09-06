import json
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import AlfWorldEnvironmentManager
from recipe.exact.probe_alfworld_exact_g import TASKS, select_games
from tests.recipe.exact.alfworld_fixtures import CharacterTokenizer, native_state, response, session


class FakePredicateEnvs:
    exact_signal = "predicates"
    get_admissible_commands = [[]]

    def __init__(self):
        self.current = session()
        self.calls = 0

    def info(self, decision=None):
        return {
            "extra.exact": {
                "snapshot": {**self.current.record, "secret_fixture": "TRAINING_ONLY_SENTINEL"},
                "actions": self.current.public_actions,
                "protections": self.current.public_protections(),
                "syntax_valid": decision.syntax_valid if decision else True,
                "execution_valid": decision.execution_valid if decision else True,
            }
        }

    def reset(self):
        return ["Your task is to: put a hot apple in a fridge."], None, [self.info()]

    def step(self, actions):
        assert len(actions) == 1
        self.calls += 1
        decision = self.current.prepare(actions[0])
        after = decision.expected_facts if decision.execution_valid else self.current.state["_facts"]
        self.current.finish(decision, native_state(self.current.model, after))
        return ["Observation after action."], None, [0], [False], [self.info(decision)]


def test_predicate_manager_preserves_raw_json_history_and_private_metadata():
    envs = FakePredicateEnvs()

    def forbidden_projection(*args):
        raise AssertionError("JSON action must not go through legacy text projection")

    manager = AlfWorldEnvironmentManager(envs, forbidden_projection, SimpleNamespace(env=SimpleNamespace(history_length=2)))
    obs, _ = manager.reset({})
    assert "TRAINING_ONLY_SENTINEL" not in obs["text"][0]
    actions = [response("take", "apple1", "table"), response("put", "apple1", "table", commit=True), "malformed"]
    for text in actions:
        registry = manager.exact_effect_schemas([envs.current.record])
        tokens = torch.tensor([list(map(ord, text))])
        schemas = manager.resolve_exact_effect_schemas(registry, [text], tokens, torch.ones_like(tokens), CharacterTokenizer())
        assert sum(span["token_end"] - span["token_start"] for span in schemas[0]["spans"]) == len(text)
        obs, _, _, infos = manager.step([text])
        assert "TRAINING_ONLY_SENTINEL" not in obs["text"][0]
        assert "source_hash" not in obs["text"][0]
        assert "future_factor_ids" not in obs["text"][0]
    assert envs.calls == len(actions)
    assert not infos[0]["is_action_syntax_valid"]
    assert '"predicate": "inreceptacle"' in obs["text"][0]
    assert actions[1] in obs["text"][0]


def test_probe_selects_two_distinct_unsliced_games_of_each_official_type(tmp_path):
    for task in TASKS:
        for index in range(3):
            path = tmp_path / task / str(index)
            path.mkdir(parents=True)
            (path / "game.tw-pddl").write_text("{}")
            (path / "traj_data.json").write_text(json.dumps({"task_type": task, "pddl_params": {"object_sliced": index == 0}}))
    selected = select_games(tmp_path, 2)
    assert all(len(paths) == 2 for paths in selected.values())
    assert all(path.parent.name != "0" for paths in selected.values() for path in paths)
    with pytest.raises(ValueError, match="distinct supported games"):
        select_games(tmp_path, 3)
