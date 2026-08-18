from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import AppWorldEnvironmentManager
from agent_system.environments.env_package.appworld.envs import appworld_execution_succeeded
from agent_system.environments.prompts.appworld import appworld_json_auth_guidance


class _FakeAppWorldEnvs:
    def __init__(self):
        self.projected_actions = None

    def step(self, actions):
        self.projected_actions = list(actions)
        return ["document result"], [0.0], [False], [{"won": False, "is_action_execution_valid": False}]


def test_appworld_execution_error_detector_uses_runtime_prefix():
    assert appworld_execution_succeeded("{'playlists': []}")
    assert not appworld_execution_succeeded("\nExecution failed. Traceback:\nException: bad API")


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
    manager.allowed_apps = [["spotify"]]
    manager.pre_text_obs = manager.tasks.copy()
    model_action = '{"app":"api_docs","api":"show_app_descriptions","arguments":{}}'

    observations, _, _, infos = manager.step([model_action])

    assert envs.projected_actions == ["print(apis.api_docs.show_app_descriptions(**{}))"]
    assert manager.memory[0][0]["action"] == model_action
    assert f"Action 1:\n{model_action}" in observations["text"][0]
    required_action = '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}'
    assert required_action in observations["text"][0]
    assert observations["text"][0].rstrip().endswith(required_action)
    assert bool(infos[0]["is_action_syntax_valid"])
    assert not bool(infos[0]["is_action_execution_valid"])
    assert not bool(infos[0]["is_action_valid"])
    assert infos[0]["tool_calling"] == 1.0


def test_json_api_auth_guidance_advances_only_after_required_calls():
    show_supervisor_apis = '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}'
    show_password_doc = '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"show_account_passwords"}}'
    fetch_passwords = '{"app":"supervisor","api":"show_account_passwords","arguments":{}}'

    assert "show_api_descriptions" in appworld_json_auth_guidance([])
    assert "show_api_descriptions" in appworld_json_auth_guidance(["not-json"])
    wrong_app = '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"spotify"}}'
    assert "show_api_descriptions" in appworld_json_auth_guidance([wrong_app])
    assert "show_api_doc" in appworld_json_auth_guidance([show_supervisor_apis])
    assert "show_account_passwords" in appworld_json_auth_guidance([show_supervisor_apis, show_password_doc])
    assert "access_token" in appworld_json_auth_guidance([show_supervisor_apis, show_password_doc, fetch_passwords])


def test_json_api_auth_guidance_persists_task_app_access_token():
    show_supervisor_apis = '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}'
    show_password_doc = '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"show_account_passwords"}}'
    fetch_passwords = '{"app":"supervisor","api":"show_account_passwords","arguments":{}}'
    bootstrap_actions = [show_supervisor_apis, show_password_doc, fetch_passwords]
    bootstrap_results = ["[]", "{}", '[{"account_name":"spotify","password":"secret"}]']

    guidance = appworld_json_auth_guidance(
        bootstrap_actions,
        prior_results=bootstrap_results,
        task_apps=["spotify"],
        supervisor_email="user@example.com",
    )
    assert '"app_name":"spotify","api_name":"login"' in guidance

    show_login_doc = '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"login"}}'
    guidance = appworld_json_auth_guidance(
        [*bootstrap_actions, show_login_doc],
        prior_results=[*bootstrap_results, "{}"],
        task_apps=["spotify"],
        supervisor_email="user@example.com",
    )
    assert '"username":"user@example.com","password":"secret"' in guidance

    login = '{"app":"spotify","api":"login","arguments":{"username":"user@example.com","password":"secret"}}'
    guidance = appworld_json_auth_guidance(
        [*bootstrap_actions, show_login_doc, login],
        prior_results=[*bootstrap_results, "{}", '{"access_token":"token-123"}'],
        task_apps=["spotify"],
        supervisor_email="user@example.com",
    )
    assert '"api":"show_api_descriptions"' in guidance

    show_spotify_apis = '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"spotify"}}'
    guidance = appworld_json_auth_guidance(
        [*bootstrap_actions, show_login_doc, login, show_spotify_apis],
        prior_results=[*bootstrap_results, "{}", '{"access_token":"token-123"}', "[]"],
        task_apps=["spotify"],
        supervisor_email="user@example.com",
    )
    assert "spotify access_token=token-123" in guidance


def test_json_api_auth_guidance_prioritizes_apps_named_in_task():
    actions = [
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"show_account_passwords"}}',
        '{"app":"supervisor","api":"show_account_passwords","arguments":{}}',
    ]
    results = [
        "[]",
        "{}",
        '[{"account_name":"amazon","password":"a"},{"account_name":"spotify","password":"s"}]',
    ]

    guidance = appworld_json_auth_guidance(
        actions,
        prior_results=results,
        task_apps=["amazon", "phone", "file_system", "spotify"],
        task_description="How many songs across my Spotify libraries were released before this year?",
        supervisor_email="user@example.com",
    )

    assert '"app_name":"spotify","api_name":"login"' in guidance
    assert '"app_name":"amazon"' not in guidance


def test_json_api_auth_guidance_uses_phone_number_when_login_schema_requires_it():
    actions = [
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"show_account_passwords"}}',
        '{"app":"supervisor","api":"show_account_passwords","arguments":{}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"phone","api_name":"login"}}',
    ]
    results = [
        "[]",
        "{}",
        '[{"account_name":"phone","password":"phone-secret"}]',
        '{"parameters":[{"name":"username","description":"Your account phone_number."}]}',
    ]

    guidance = appworld_json_auth_guidance(
        actions,
        prior_results=results,
        task_apps=["phone"],
        supervisor_email="user@example.com",
        supervisor_phone_number="5550100",
    )

    assert '"username":"5550100","password":"phone-secret"' in guidance
