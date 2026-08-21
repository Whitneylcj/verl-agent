import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "workers" / "actor" / "update_contract.py"
SPEC = importlib.util.spec_from_file_location("actor_update_contract_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
resolve_update_contract = MODULE.resolve_update_contract


def test_default_actor_contract_uses_ppo_minibatches_without_rescaling():
    assert resolve_update_contract({}) == (False, 1.0)


def test_exact_actor_contract_requests_one_full_rollout_update():
    assert resolve_update_contract({"update_batch_mode": "full_rollout", "regularizer_scale": 4.0}) == (True, 4.0)


@pytest.mark.parametrize(
    "meta_info",
    (
        {"update_batch_mode": "unknown"},
        {"regularizer_scale": 0.0},
        {"regularizer_scale": float("nan")},
    ),
)
def test_actor_contract_fails_closed(meta_info):
    with pytest.raises(ValueError):
        resolve_update_contract(meta_info)
