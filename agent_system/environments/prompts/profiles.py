"""Explicit prompt profiles for benchmark and model-specific adaptations."""

from __future__ import annotations

from typing import Any

PROMPT_PROFILES = frozenset({"benchmark", "appworld_exact_json", "qwen_small_guided"})
__all__ = ["PROMPT_PROFILES", "resolve_prompt_profile"]


def resolve_prompt_profile(config: Any) -> str:
    env_config = config.env
    if hasattr(env_config, "get"):
        profile = str(env_config.get("prompt_profile", "benchmark"))
    else:
        profile = str(getattr(env_config, "prompt_profile", "benchmark"))
    if profile not in PROMPT_PROFILES:
        supported = ", ".join(sorted(PROMPT_PROFILES))
        raise ValueError(f"unsupported env.prompt_profile={profile!r}; choose one of: {supported}")
    env_name = str(getattr(env_config, "env_name", "")).lower()
    if profile == "appworld_exact_json" and "appworld" not in env_name:
        raise ValueError("appworld_exact_json is only valid for AppWorld")
    return profile
