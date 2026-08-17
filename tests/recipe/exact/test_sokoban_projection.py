import importlib.util
from pathlib import Path

_PROJECTION_PATH = Path(__file__).parents[3] / "agent_system/environments/env_package/sokoban/projection.py"
_PROJECTION_SPEC = importlib.util.spec_from_file_location("exact_sokoban_projection", _PROJECTION_PATH)
assert _PROJECTION_SPEC is not None and _PROJECTION_SPEC.loader is not None
_PROJECTION_MODULE = importlib.util.module_from_spec(_PROJECTION_SPEC)
_PROJECTION_SPEC.loader.exec_module(_PROJECTION_MODULE)
sokoban_projection = _PROJECTION_MODULE.sokoban_projection


def test_sokoban_projection_accepts_one_exact_action_without_think_tags():
    actions = ["<tool_call>Move toward the box.<action>right</action></tool_call>"]

    projected, valid = sokoban_projection(actions)

    assert projected == [4]
    assert valid == [1]


def test_sokoban_projection_rejects_missing_ambiguous_or_fuzzy_actions():
    actions = [
        "right",
        "<action>upright</action>",
        "<action>left</action><action>right</action>",
    ]

    projected, valid = sokoban_projection(actions)

    assert projected == [0, 0, 0]
    assert valid == [0, 0, 0]
