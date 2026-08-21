import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[3] / "verl" / "trainer" / "ppo" / "checkpoint_utils.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_utils_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
save_checkpoint_if_needed = MODULE.save_checkpoint_if_needed


def test_periodic_and_budget_save_on_same_step_execute_once():
    calls = []
    saved = save_checkpoint_if_needed(
        should_save=True,
        already_saved=False,
        save=lambda: calls.append("periodic"),
    )
    saved = save_checkpoint_if_needed(
        should_save=True,
        already_saved=saved,
        save=lambda: calls.append("budget"),
    )

    assert saved
    assert calls == ["periodic"]
