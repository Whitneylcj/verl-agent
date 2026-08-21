from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import AppWorldEnvironmentManager
from agent_system.environments.env_package.appworld.envs import appworld_execution_succeeded


class _FakeAppWorldEnvs:
    def __init__(self):
        self.projected_actions = None

    def step(self, actions):
        self.projected_actions = list(actions)
        return ["document result"], [0.0], [False], [{"won": False, "is_action_execution_valid": False}]


def _config(profile="appworld_exact_json", action_mode="json_api"):
    return SimpleNamespace(
        env=SimpleNamespace(
            env_name="AppWorld",
            prompt_profile=profile,
            history_length=2,
            appworld=SimpleNamespace(action_mode=action_mode),
        )
    )


def _manager(profile="appworld_exact_json", action_mode="json_api"):
    envs = _FakeAppWorldEnvs()

    def mutating_projection(actions):
        actions[0] = "print(apis.api_docs.show_app_descriptions(**{}))"
        return actions, [1]

    manager = AppWorldEnvironmentManager(envs, mutating_projection, _config(profile, action_mode))
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
    manager.allowed_apps = [["spotify"]]
    manager.pre_text_obs = manager.tasks.copy()
    return manager, envs


def test_appworld_execution_error_detector_uses_runtime_prefix():
    assert appworld_execution_succeeded("{'playlists': []}")
    assert not appworld_execution_succeeded("\nExecution failed. Traceback:\nException: bad API")


def test_json_api_history_keeps_model_action_not_compiled_python():
    manager, envs = _manager()
    model_action = '{"app":"api_docs","api":"show_app_descriptions","arguments":{}}'

    observations, _, _, infos = manager.step([model_action])

    assert envs.projected_actions == ["print(apis.api_docs.show_app_descriptions(**{}))"]
    assert manager.memory[0][0]["action"] == model_action
    assert f"Action 1:\n{model_action}" in observations["text"][0]
    assert "NEXT-ACTION CONSTRAINT" not in observations["text"][0]
    assert bool(infos[0]["is_action_syntax_valid"])
    assert not bool(infos[0]["is_action_execution_valid"])
    assert not bool(infos[0]["is_action_valid"])
    assert infos[0]["tool_calling"] == 1.0


@pytest.mark.parametrize(
    ("profile", "action_mode"),
    (("benchmark", "json_api"), ("appworld_exact_json", "python")),
)
def test_appworld_prompt_profile_rejects_mismatched_action_mode(profile, action_mode):
    manager, _ = _manager(profile, action_mode)

    with pytest.raises(ValueError, match="requires env.appworld.action_mode"):
        manager.build_text_obs(["result"], init=True)
