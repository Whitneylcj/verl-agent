"""Explicit batch/update metadata contract shared by actor objectives."""

from __future__ import annotations

import math
from typing import Any, Mapping


def resolve_update_contract(meta_info: Mapping[str, Any]) -> tuple[bool, float]:
    update_batch_mode = str(meta_info.get("update_batch_mode", "ppo_minibatch"))
    if update_batch_mode not in {"ppo_minibatch", "full_rollout"}:
        raise ValueError(f"Unsupported update_batch_mode: {update_batch_mode!r}")
    regularizer_scale = float(meta_info.get("regularizer_scale", 1.0))
    if not math.isfinite(regularizer_scale) or regularizer_scale <= 0:
        raise ValueError(f"regularizer_scale must be finite and positive, got {regularizer_scale!r}")
    return update_batch_mode == "full_rollout", regularizer_scale
