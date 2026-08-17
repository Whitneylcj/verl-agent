from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import AlfWorldEnvironmentManager


class _FakeAlfWorldEnvs:
    get_admissible_commands = [["look", "go to fridge 1"]]

    def step(self, actions):
        assert actions == ["go to fridge 1"]
        self.get_admissible_commands = [["look", "open fridge 1"]]
        return ["You arrive at fridge 1."], None, [0.0], [False], [{}]


def test_alfworld_manager_separates_response_syntax_from_execution():
    envs = _FakeAlfWorldEnvs()

    def projection(_responses, _action_pools):
        return ["go to fridge 1"], [0]

    config = SimpleNamespace(env=SimpleNamespace(history_length=0))
    manager = AlfWorldEnvironmentManager(envs, projection, config)
    manager.memory.reset(batch_size=1)
    manager.pre_text_obs = ["At the start."]
    manager.gamefile = [None]

    _, _, _, infos = manager.step(["malformed response"])

    assert not bool(infos[0]["is_action_syntax_valid"])
    assert bool(infos[0]["is_action_execution_valid"])
    assert not bool(infos[0]["is_action_valid"])


def test_alfworld_manager_marks_well_formed_inadmissible_action_as_execution_error():
    envs = _FakeAlfWorldEnvs()

    def projection(_responses, _action_pools):
        return ["open cabinet 9"], [0]

    def step(actions):
        assert actions == ["open cabinet 9"]
        return ["Nothing happens."], None, [0.0], [False], [{}]

    envs.step = step
    config = SimpleNamespace(env=SimpleNamespace(history_length=0))
    manager = AlfWorldEnvironmentManager(envs, projection, config)
    manager.memory.reset(batch_size=1)
    manager.pre_text_obs = ["At the start."]
    manager.gamefile = [None]
    response = "<thinking>Try the cabinet.</thinking><action>open cabinet 9</action>"

    _, _, _, infos = manager.step([response])

    assert bool(infos[0]["is_action_syntax_valid"])
    assert not bool(infos[0]["is_action_execution_valid"])
    assert not bool(infos[0]["is_action_valid"])
