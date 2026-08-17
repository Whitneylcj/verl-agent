from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("ray")

from recipe.exact.monitor import ExactObserver
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_fit_marks_trainer_exception_and_reraises(tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_steps = 4
    observer = ExactObserver(tmp_path)

    def fail(self):
        self._exact_run_observer = observer
        raise RuntimeError("synthetic trainer failure")

    trainer._fit_impl = types.MethodType(fail, trainer)
    with pytest.raises(RuntimeError, match="synthetic trainer failure"):
        trainer.fit()

    heartbeat = json.loads((tmp_path / "heartbeat.json").read_text())
    assert heartbeat["status"] == "failed"
    assert heartbeat["step"] == 4
    assert heartbeat["failure"]["type"] == "RuntimeError"
