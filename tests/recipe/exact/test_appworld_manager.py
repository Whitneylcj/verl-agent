from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import AppWorldEnvironmentManager


class _FakeAppWorldEnvs:
    def __init__(self):
        self.projected_actions = None

    def step(self, actions):
        self.projected_actions = list(actions)
        return ["document result"], [0.0], [False], [{"won": False}]


def test_json_api_history_keeps_model_action_not_compiled_python():
    envs = _FakeAppWorldEnvs()

    def mutating_projection(actions):
        actions[0] = "print(apis.api_docs.show_app_descriptions(**{}))"
        return actions, [1]

    config = SimpleNamespace(
        env=SimpleNamespace(
            history_length=2,
            appworld=SimpleNamespace(action_mode="json_api"),
        )
    )
    manager = AppWorldEnvironmentManager(envs, mutating_projection, config)
    manager.memory.reset(batch_size=1)
    manager.supervisors = [
        {
            "first_name": "Ada",
            "last_name": "Lovelace",
            "email": "ada@example.com",
            "phone_number": "5550100",
        }
    ]
    manager.tasks = ["Inspect the available apps."]
    manager.pre_text_obs = manager.tasks.copy()
    model_action = '{"app":"api_docs","api":"show_app_descriptions","arguments":{}}'

    observations, _, _, infos = manager.step([model_action])

    assert envs.projected_actions == ["print(apis.api_docs.show_app_descriptions(**{}))"]
    assert manager.memory[0][0]["action"] == model_action
    assert f"Action 1:\n{model_action}" in observations["text"][0]
    assert bool(infos[0]["is_action_valid"])
