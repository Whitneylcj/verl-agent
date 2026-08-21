from types import SimpleNamespace

import pytest

from agent_system.environments.prompts.alfworld import (
    ALFWORLD_QWEN_SMALL_GUIDED_TEMPLATE,
    ALFWORLD_TEMPLATE,
)
from agent_system.environments.prompts.appworld import (
    APPWORLD_JSON_API_TEMPLATE,
    APPWORLD_QWEN_SMALL_GUIDED_JSON_API_TEMPLATE,
)
from agent_system.environments.prompts.profiles import resolve_prompt_profile
from agent_system.environments.prompts.sokoban import (
    SOKOBAN_QWEN_SMALL_GUIDED_TEMPLATE,
    SOKOBAN_TEMPLATE,
)


def _config(profile=None, env_name="Sokoban"):
    env = SimpleNamespace(env_name=env_name)
    if profile is not None:
        env.prompt_profile = profile
    return SimpleNamespace(env=env)


def test_benchmark_is_default_and_rejects_environment_specific_profile():
    assert resolve_prompt_profile(_config()) == "benchmark"
    with pytest.raises(ValueError, match="only valid for AppWorld"):
        resolve_prompt_profile(_config("appworld_exact_json"))


def test_benchmark_prompts_exclude_model_specific_guidance():
    assert "Push Geometry" not in SOKOBAN_TEMPLATE
    assert "Mandatory Loop Break" not in SOKOBAN_TEMPLATE
    assert "exactly two non-empty lines" not in ALFWORLD_TEMPLATE
    assert "Push Geometry" in SOKOBAN_QWEN_SMALL_GUIDED_TEMPLATE
    assert "exactly two non-empty lines" in ALFWORLD_QWEN_SMALL_GUIDED_TEMPLATE


def test_appworld_exact_json_is_minimal_and_guidance_is_opt_in():
    assert "one documented API call per turn" in APPWORLD_JSON_API_TEMPLATE
    assert "intermediate_reward" not in APPWORLD_JSON_API_TEMPLATE
    assert "exact_factor" not in APPWORLD_JSON_API_TEMPLATE
    assert "NEXT-ACTION CONSTRAINT" not in APPWORLD_JSON_API_TEMPLATE
    assert "NEXT-ACTION CONSTRAINT" in APPWORLD_QWEN_SMALL_GUIDED_JSON_API_TEMPLATE
