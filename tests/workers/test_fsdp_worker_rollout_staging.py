import importlib.util
from pathlib import Path
from types import SimpleNamespace

MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "workers" / "rollout_staging.py"
SPEC = importlib.util.spec_from_file_location("rollout_staging_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_rollout_components = MODULE.build_rollout_components
finalize_actor_update = MODULE.finalize_actor_update


class _Device:
    def __init__(self, events):
        self.events = events

    def empty_cache(self):
        self.events.append("empty_cache")

    def synchronize(self):
        self.events.append("synchronize")


def test_rollout_components_are_built_only_for_rollout_role():
    calls = []

    def build():
        calls.append("build")
        return "rollout", "manager"

    assert build_rollout_components(False, build) is None
    assert calls == []
    assert build_rollout_components(True, build) == ("rollout", "manager")
    assert calls == ["build"]


def test_update_stages_weights_before_actor_offload_and_synchronization():
    events = []
    manager = SimpleNamespace(stage_updated_weights=lambda: events.append("stage_weights"))

    finalize_actor_update(
        is_rollout=True,
        rollout_sharding_manager=manager,
        offload_param=True,
        offload_optimizer=True,
        actor_module="actor",
        actor_optimizer="optimizer",
        device=_Device(events),
        offload_model=lambda module: events.append(f"offload_model:{module}"),
        offload_optim=lambda *, optimizer: events.append(f"offload_optimizer:{optimizer}"),
    )

    assert events == [
        "empty_cache",
        "stage_weights",
        "offload_model:actor",
        "offload_optimizer:optimizer",
        "synchronize",
        "empty_cache",
    ]


def test_update_without_offload_does_not_synchronize():
    events = []
    manager = SimpleNamespace(stage_updated_weights=lambda: events.append("stage_weights"))

    finalize_actor_update(
        is_rollout=True,
        rollout_sharding_manager=manager,
        offload_param=False,
        offload_optimizer=False,
        actor_module=None,
        actor_optimizer=None,
        device=_Device(events),
        offload_model=lambda module: events.append("unexpected_model_offload"),
        offload_optim=lambda *, optimizer: events.append("unexpected_optimizer_offload"),
    )

    assert events == ["empty_cache", "stage_weights"]
