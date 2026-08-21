from types import SimpleNamespace

import pytest

from recipe.exact.integration import ExactEnvironmentAdapter


def test_adapter_delegates_snapshots_to_vector_environment():
    snapshots = [SimpleNamespace(checkpoint_id=0)]
    manager = SimpleNamespace(envs=SimpleNamespace(exact_credit_snapshots=lambda: snapshots))

    assert ExactEnvironmentAdapter(manager).snapshots() == snapshots


def test_adapter_uses_vector_schema_and_manager_specific_resolution():
    manager = SimpleNamespace(
        envs=SimpleNamespace(exact_effect_schemas=lambda: ["schema:1"]),
        resolve_exact_effect_schemas=lambda schemas, **kwargs: [f"resolved:{schemas[0]}"],
    )
    adapter = ExactEnvironmentAdapter(manager)

    schemas = adapter.effect_schemas(["before"])
    assert schemas == ["schema:1"]
    assert adapter.resolve_effect_schemas(schemas, ["action"], None, None, None) == ["resolved:schema:1"]


def test_adapter_fails_closed_when_snapshot_provider_is_missing():
    with pytest.raises(NotImplementedError, match="EXACT credit probe"):
        ExactEnvironmentAdapter(SimpleNamespace(envs=SimpleNamespace())).snapshots()
