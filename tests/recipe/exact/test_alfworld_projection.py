import importlib.util
from pathlib import Path


_PROJECTION_PATH = (
    Path(__file__).parents[3]
    / "agent_system/environments/env_package/alfworld/projection.py"
)
_SPEC = importlib.util.spec_from_file_location("alfworld_projection_under_test", _PROJECTION_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
alfworld_projection = _MODULE.alfworld_projection


def test_alfworld_projection_requires_an_admissible_action():
    actions, valids = alfworld_projection(
        ["<think>Go to the visible fridge.</think><action>GO TO FRIDGE 1</action>"],
        [["look", "go to fridge 1"]],
    )

    assert actions == ["go to fridge 1"]
    assert valids == [1]


def test_alfworld_projection_rejects_well_formed_hallucinated_action():
    actions, valids = alfworld_projection(
        ["<think>The mug was elsewhere.</think><action>take mug 1 from countertop 4</action>"],
        [["look", "open cabinet 1"]],
    )

    assert actions == ["take mug 1 from countertop 4"]
    assert valids == [0]


def test_alfworld_projection_still_rejects_missing_reasoning_tags():
    actions, valids = alfworld_projection(
        ["<action>look</action>"],
        [["look"]],
    )

    assert actions == ["look"]
    assert valids == [0]
