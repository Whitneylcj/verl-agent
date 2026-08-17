import importlib.util
from pathlib import Path

import pytest

from recipe.exact.appworld_adapter import compile_json_api_action, parse_json_api_action

_PROJECTION_PATH = Path(__file__).parents[3] / "agent_system/environments/env_package/appworld/projection.py"
_PROJECTION_SPEC = importlib.util.spec_from_file_location("exact_appworld_projection", _PROJECTION_PATH)
assert _PROJECTION_SPEC is not None and _PROJECTION_SPEC.loader is not None
_PROJECTION_MODULE = importlib.util.module_from_spec(_PROJECTION_SPEC)
_PROJECTION_SPEC.loader.exec_module(_PROJECTION_MODULE)
appworld_projection = _PROJECTION_MODULE.appworld_projection


def test_json_api_projection_compiles_one_fixed_shape_call():
    action = compile_json_api_action('<think>inspect docs</think><action>{"app":"api_docs","api":"show_api_doc","arguments":{"app_name":"spotify","include_private":false}}</action>')
    assert action == "print(apis.api_docs.show_api_doc(**{'app_name': 'spotify', 'include_private': False}))"


def test_json_api_parser_exposes_prefix_safe_argument_boundary():
    action = '<think>x</think><action>{"app":"spotify","api":"play_song","arguments":{"song_id":7}}</action>'
    parsed = parse_json_api_action(action)
    assert (parsed.app, parsed.api, parsed.arguments) == (
        "spotify",
        "play_song",
        {"song_id": 7},
    )
    assert action[parsed.arguments_char_start :].startswith('"arguments"')


@pytest.mark.parametrize(
    "action",
    [
        '{"app":"spotify","api":"play_song","arguments":{"song_id":7}}',
        '```json\n{"app":"spotify","api":"play_song","arguments":{"song_id":7}}\n```',
        '<action>{"app":"spotify","api":"play_song","arguments":{"song_id":7}}</action>',
        '{"think":"inspect docs","action":{"app":"spotify","api":"play_song","arguments":{"song_id":7}}}',
    ],
)
def test_json_api_parser_accepts_safe_small_model_wrappers(action):
    parsed = parse_json_api_action(action)
    assert (parsed.app, parsed.api, parsed.arguments) == (
        "spotify",
        "play_song",
        {"song_id": 7},
    )
    assert action[parsed.arguments_char_start :].startswith('"arguments"')


def test_json_api_projection_does_not_replace_model_history_in_place():
    model_actions = ['{"app":"api_docs","api":"show_app_descriptions","arguments":{}}']
    projected, valids = appworld_projection(model_actions, action_mode="json_api")
    assert valids == [1]
    assert projected == ["print(apis.api_docs.show_app_descriptions(**{}))"]
    assert model_actions == ['{"app":"api_docs","api":"show_app_descriptions","arguments":{}}']


def test_json_api_projection_rejects_code_and_multiple_calls():
    invalid_actions = [
        '<think>x</think><action>{"app":"api_docs","api":"show_app_descriptions","arguments":{}}</action><action>{"app":"supervisor","api":"complete_task","arguments":{}}</action>',
        "<think>x</think><code>print(apis.supervisor.complete_task())</code>",
        '<think>x</think><action>{"app":"__class__","api":"mro","arguments":{}}</action>',
        '<think>x</think><action>{"api":"show_profile","app":"supervisor","arguments":{}}</action>',
        '<think>x</think><action>{"app":"supervisor","api":"show_profile","api":"complete_task","arguments":{}}</action>',
        '<think>x</think><action>{"app":"supervisor","api":"show_profile","arguments":{}}</action>junk',
        'prefix {"app":"supervisor","api":"show_profile","arguments":{}}',
        '```python\n{"app":"supervisor","api":"show_profile","arguments":{}}\n```',
        '{"think":"x","action":{"app":"supervisor","api":"show_profile","arguments":{}},"extra":1}',
        '{"app":"app_name","api":"show_profile","arguments":{}}',
        '{"app":"supervisor","api":"api_name","arguments":{}}',
    ]
    for invalid_action in invalid_actions:
        try:
            compile_json_api_action(invalid_action)
        except ValueError:
            pass
        else:
            raise AssertionError("structured adapter accepted an unsafe action")
