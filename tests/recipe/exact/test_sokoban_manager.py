from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import SokobanEnvironmentManager, _sokoban_legal_actions


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


def test_sokoban_manager_warns_against_repeating_unchanged_action():
    envs = _FakeSokobanEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=2))
    manager = SokobanEnvironmentManager(envs, lambda actions: ([4], [1]), config)
    manager.memory.reset(batch_size=1)
    manager.memory.store({"text_obs": ["same board"], "action": ["Right"]})

    prompts = manager.build_text_obs([{}], ["same board"])

    assert "# Mandatory Loop Break" in prompts[0]
    assert "previous right action left the board unchanged" in prompts[0]
    assert "Do not choose right again this step" in prompts[0]


def test_sokoban_legal_actions_are_inferred_without_stepping_environment():
    board = """
    # # # # #
    # _ O _ #
    # _ X _ #
    # _ P _ #
    # # # # #
    """

    legal_actions, legal_pushes = _sokoban_legal_actions(board)

    assert legal_actions == ["up", "left", "right"]
    assert legal_pushes == ["up"]


def test_sokoban_prompt_exposes_only_current_observation_constraints():
    envs = _FakeSokobanEnvs()
    config = SimpleNamespace(env=SimpleNamespace(history_length=0))
    manager = SokobanEnvironmentManager(envs, lambda actions: ([4], [1]), config)
    board = """
    # # # # #
    # _ O _ #
    # _ X _ #
    # _ P _ #
    # # # # #
    """

    prompt = manager.build_text_obs([{}], [board], init=True)[0]

    assert "Choose only an action that changes the board: [up, left, right]" in prompt
    assert "Legal box pushes available now: [up]" in prompt
    assert "your action MUST be one of: [up]" in prompt
    assert prompt.rfind("# Current Legal Actions") > prompt.rfind("Line 2 must be exactly one of")
