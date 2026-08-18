import json
from types import SimpleNamespace

import pytest

pytest.importorskip("omegaconf")

from agent_system.environments.env_manager import AppWorldEnvironmentManager
from agent_system.environments.env_package.appworld.envs import appworld_execution_succeeded
from agent_system.environments.prompts.appworld import _appworld_relevant_api_names, appworld_json_auth_guidance


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

    clock_action = '{"app":"phone","api":"get_current_date_and_time","arguments":{}}'
    assert clock_action in guidance
    assert '"app_name":"amazon"' not in guidance

    guidance = appworld_json_auth_guidance(
        [*actions, clock_action],
        prior_results=[*results, '{"date":"2026-08-18","time":"09:00:00"}'],
        task_apps=["amazon", "phone", "file_system", "spotify"],
        task_description="How many songs across my Spotify libraries were released before this year?",
        supervisor_email="user@example.com",
    )
    assert '"app_name":"spotify","api_name":"login"' in guidance
    assert '"app_name":"amazon"' not in guidance


def test_appworld_api_ranking_prefers_read_only_library_endpoints():
    descriptions = [
        {"name": "play_music", "description": "Play a song, album, or playlist."},
        {"name": "remove_song_from_library", "description": "Remove a song from your library."},
        {"name": "show_song", "description": "Get details of a specific song."},
        {"name": "show_song_library", "description": "Show songs in your song library."},
        {"name": "show_album_library", "description": "Show albums in your album library."},
        {"name": "search_albums", "description": "Search albums with a query."},
    ]
    task = "How many songs across my song and album libraries were released before this year?"

    ranked = _appworld_relevant_api_names(descriptions, task, limit=6)

    assert ranked[:2] == ["show_song_library", "show_album_library"]
    assert "show_song" in ranked
    assert "search_albums" in ranked
    assert "play_music" not in ranked
    assert "remove_song_from_library" not in ranked


def test_json_api_auth_guidance_inspects_top_task_api_schemas():
    actions = [
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"show_account_passwords"}}',
        '{"app":"supervisor","api":"show_account_passwords","arguments":{}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"login"}}',
        '{"app":"spotify","api":"login","arguments":{"username":"user@example.com","password":"secret"}}',
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"spotify"}}',
    ]
    results = [
        "[]",
        "{}",
        '[{"account_name":"spotify","password":"secret"}]',
        "{}",
        '{"access_token":"token-123"}',
        json.dumps(
            [
                {"name": "show_song_library", "description": "Show songs in your song library."},
                {"name": "show_album_library", "description": "Show albums in your album library."},
                {"name": "show_profile", "description": "Show the profile."},
            ]
        ),
    ]
    task = "How many songs across my Spotify song and album libraries were released before this year?"
    library_doc = json.dumps(
        {
            "parameters": [
                {"name": "access_token", "required": True},
                {"name": "page_index", "required": False},
            ]
        }
    )

    guidance = appworld_json_auth_guidance(
        actions,
        prior_results=results,
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert '"api_name":"show_song_library"' in guidance

    song_doc = '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"show_song_library"}}'
    guidance = appworld_json_auth_guidance(
        [*actions, song_doc],
        prior_results=[*results, library_doc],
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert '"app":"spotify","api":"show_song_library","arguments":{"access_token":"token-123"}' in guidance

    song_call = '{"app":"spotify","api":"show_song_library","arguments":{"access_token":"token-123"}}'
    guidance = appworld_json_auth_guidance(
        [*actions, song_doc, song_call],
        prior_results=[*results, library_doc, "[]"],
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert '"api_name":"show_album_library"' in guidance

    album_doc = '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"show_album_library"}}'
    guidance = appworld_json_auth_guidance(
        [*actions, song_doc, song_call, album_doc],
        prior_results=[*results, library_doc, "[]", library_doc],
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert '"app":"spotify","api":"show_album_library","arguments":{"access_token":"token-123"}' in guidance

    album_call = '{"app":"spotify","api":"show_album_library","arguments":{"access_token":"token-123"}}'
    guidance = appworld_json_auth_guidance(
        [*actions, song_doc, song_call, album_doc, album_call],
        prior_results=[*results, library_doc, "[]", library_doc, "[]"],
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert '"app_name":"supervisor","api_name":"complete_task"' in guidance

    complete_doc = '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"complete_task"}}'
    guidance = appworld_json_auth_guidance(
        [*actions, song_doc, song_call, album_doc, album_call, complete_doc],
        prior_results=[*results, library_doc, "[]", library_doc, "[]", "{}"],
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert "Authentication bootstrap is complete" in guidance
    assert "Never invent a derived API name" in guidance
    assert "spotify: show_song_library, show_album_library" in guidance
    assert "call supervisor.complete_task now" in guidance


def test_json_api_auth_guidance_anchors_relative_dates_with_public_phone_clock():
    actions = [
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"show_account_passwords"}}',
        '{"app":"supervisor","api":"show_account_passwords","arguments":{}}',
    ]
    results = ["[]", "{}", '[{"account_name":"spotify","password":"secret"}]']
    task = "Count Spotify songs released before this year."

    guidance = appworld_json_auth_guidance(
        actions,
        prior_results=results,
        task_apps=["spotify", "phone"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    clock_action = '{"app":"phone","api":"get_current_date_and_time","arguments":{}}'
    assert clock_action in guidance

    guidance = appworld_json_auth_guidance(
        [*actions, clock_action],
        prior_results=[*results, '{"date":"2026-08-18","time":"09:00:00"}'],
        task_apps=["spotify", "phone"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert '"app_name":"spotify","api_name":"login"' in guidance


def test_json_api_auth_guidance_fetches_missing_library_detail_dates():
    api_descriptions = json.dumps(
        [
            {"name": "show_song_library", "description": "Show songs in your song library."},
            {"name": "show_album_library", "description": "Show albums in your album library."},
            {"name": "show_song", "description": "Get details of a specific song."},
        ]
    )
    library_doc = json.dumps({"parameters": [{"name": "access_token", "required": True}]})
    detail_doc = json.dumps({"parameters": [{"name": "song_id", "required": True}]})
    actions = [
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"supervisor"}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"supervisor","api_name":"show_account_passwords"}}',
        '{"app":"supervisor","api":"show_account_passwords","arguments":{}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"login"}}',
        '{"app":"spotify","api":"login","arguments":{"username":"user@example.com","password":"secret"}}',
        '{"app":"api_docs","api":"show_api_descriptions","arguments":{"app_name":"spotify"}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"show_song_library"}}',
        '{"app":"spotify","api":"show_song_library","arguments":{"access_token":"token-123"}}',
        '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"show_album_library"}}',
        '{"app":"spotify","api":"show_album_library","arguments":{"access_token":"token-123"}}',
    ]
    results = [
        "[]",
        "{}",
        '[{"account_name":"spotify","password":"secret"}]',
        "{}",
        '{"access_token":"token-123"}',
        api_descriptions,
        library_doc,
        '[{"song_id":11,"album_id":1},{"song_id":12,"album_id":2}]',
        library_doc,
        '[{"album_id":2,"release_date":"2020-01-01","song_ids":[12,13]}]',
    ]
    task = "Count distinct songs across my song and album libraries released before this year."

    guidance = appworld_json_auth_guidance(
        actions,
        prior_results=results,
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    detail_doc_action = '{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","api_name":"show_song"}}'
    assert detail_doc_action in guidance

    guidance = appworld_json_auth_guidance(
        [*actions, detail_doc_action],
        prior_results=[*results, detail_doc],
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    first_detail_action = '{"app":"spotify","api":"show_song","arguments":{"song_id":11}}'
    assert first_detail_action in guidance

    guidance = appworld_json_auth_guidance(
        [*actions, detail_doc_action, first_detail_action],
        prior_results=[*results, detail_doc, '{"song_id":11,"release_date":"2019-01-01"}'],
        task_apps=["spotify"],
        task_description=task,
        supervisor_email="user@example.com",
    )
    assert '{"app":"spotify","api":"show_song","arguments":{"song_id":12}}' in guidance


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
