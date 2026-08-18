import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("ray")

from agent_system.environments.env_package.alfworld.envs import AlfworldWorker


class _FakeBatchEnv:
    def __init__(self):
        self.seed_calls = []

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        return ["observation"], {"extra.gamefile": [None]}


class _FakeBaseEnv:
    def __init__(self):
        self.batch_env = _FakeBatchEnv()

    def init_env(self, batch_size):
        assert batch_size == 1
        return self.batch_env


def _worker(deterministic_reset):
    base_env = _FakeBaseEnv()
    worker = AlfworldWorker({}, 17, base_env, deterministic_reset=deterministic_reset)
    worker._refresh_exact_snapshot = lambda: None
    return worker, base_env.batch_env


def test_validation_worker_reseeds_before_every_reset():
    worker, batch_env = _worker(deterministic_reset=True)

    worker.reset()
    worker.reset()

    assert batch_env.seed_calls == [17, 17, 17]


def test_training_worker_advances_without_reseeding():
    worker, batch_env = _worker(deterministic_reset=False)

    worker.reset()
    worker.reset()

    assert batch_env.seed_calls == [17]
