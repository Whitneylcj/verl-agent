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


def test_worker_remembers_target_discovery_after_leaving_observation():
    worker, _ = _worker(deterministic_reset=False)
    worker._exact_task_params = {
        "task_type": "pick_heat_then_place_in_recep",
        "object_target": "Mug",
        "parent_target": "CoffeeMachine",
    }
    worker._exact_info = {
        "observation_text": "The fridge is open. In it, you see a mug 1.",
        "facts": [],
    }
    worker._refresh_exact_snapshot()
    worker._exact_info = {
        "observation_text": "You arrive at coffeemachine 1.",
        "facts": [],
    }
    worker._refresh_exact_snapshot()

    factor_index = worker._exact_snapshot["factor_ids"].index("object_discovered")
    assert worker._exact_snapshot["values"][factor_index] == 1.0
