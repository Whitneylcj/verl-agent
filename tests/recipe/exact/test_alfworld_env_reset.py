import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("ray")

from agent_system.environments.env_package.alfworld.envs import AlfworldWorker


class _FakeBatchEnv:
    def __init__(self):
        self.seed_calls = []
        self.intermediate_rewards = []
        self.reset_intermediate_reward = None

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        return ["observation"], {
            "extra.gamefile": [None],
            "intermediate_reward": [self.reset_intermediate_reward],
        }

    def step(self, actions):
        assert actions == ["look"]
        reward = self.intermediate_rewards.pop(0)
        return ["observation"], [0.0], [False], {"intermediate_reward": [reward]}


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


def test_worker_accumulates_official_textworld_intermediate_reward():
    worker, batch_env = _worker(deterministic_reset=False)
    batch_env.intermediate_rewards = [1.0, 0.0, -1.0]
    worker.reset()

    assert worker.exact_credit_snapshot()["values"] == (0.0,)
    previous_value = worker.exact_credit_snapshot()["values"][0]
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (1.0,)
    assert worker.exact_credit_snapshot()["values"][0] - previous_value == 1.0
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (1.0,)
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (0.0,)
    worker.reset()
    assert worker.exact_credit_snapshot()["values"] == (0.0,)


def test_worker_fails_closed_on_invalid_intermediate_reward():
    worker, batch_env = _worker(deterministic_reset=False)
    batch_env.intermediate_rewards = [0.5]
    worker.reset()

    with pytest.raises(RuntimeError, match=r"\{-1, 0, 1\}"):
        worker.step("look")


def test_worker_accepts_explicit_zero_at_reset():
    worker, batch_env = _worker(deterministic_reset=False)
    batch_env.reset_intermediate_reward = 0.0

    worker.reset()

    assert worker.exact_credit_snapshot()["values"] == (0.0,)


def test_worker_fails_closed_when_intermediate_reward_is_missing_after_step():
    worker, batch_env = _worker(deterministic_reset=False)
    batch_env.intermediate_rewards = [None]
    worker.reset()

    with pytest.raises(RuntimeError, match="unavailable after step"):
        worker.step("look")
