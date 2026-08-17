from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import SokobanEnvironmentManager


class _FakeSokobanEnvs:
    mode = "tiny_rgb_array"

    def step(self, actions):
        assert actions == [4, 4]
        return ["same board", "changed board"], [0.0, 0.0], [False, False], [{}, {}]


def test_sokoban_manager_separates_syntax_from_noop_execution():
    envs = _FakeSokobanEnvs()

    def projection(_actions):
        return [4, 4], [1, 1]

    config = SimpleNamespace(env=SimpleNamespace(history_length=0))
    manager = SokobanEnvironmentManager(envs, projection, config)
    manager.memory.reset(batch_size=2)
    manager.pre_text_obs = ["same board", "old board"]

    _, _, _, infos = manager.step(["first", "second"])

    assert bool(infos[0]["is_action_syntax_valid"])
    assert not bool(infos[0]["is_action_execution_valid"])
    assert not bool(infos[0]["is_action_valid"])
    assert bool(infos[1]["is_action_syntax_valid"])
    assert bool(infos[1]["is_action_execution_valid"])
    assert bool(infos[1]["is_action_valid"])
