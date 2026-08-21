"""Small, behavior-testable helpers for actor-to-rollout weight staging."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def build_rollout_components(enabled: bool, builder: Callable[[], tuple[Any, Any]]) -> tuple[Any, Any] | None:
    """Build rollout state only for workers that own the rollout role."""

    return builder() if enabled else None


def finalize_actor_update(
    *,
    is_rollout: bool,
    rollout_sharding_manager: Any,
    offload_param: bool,
    offload_optimizer: bool,
    actor_module: Any,
    actor_optimizer: Any,
    device: Any,
    offload_model: Callable[[Any], None],
    offload_optim: Callable[..., None],
) -> None:
    """Stage updated rollout weights before enqueueing actor offload copies."""

    if is_rollout:
        device.empty_cache()
        rollout_sharding_manager.stage_updated_weights()
    if offload_param:
        offload_model(actor_module)
    if offload_optimizer:
        offload_optim(optimizer=actor_optimizer)
    if offload_param or offload_optimizer:
        device.synchronize()
        device.empty_cache()
