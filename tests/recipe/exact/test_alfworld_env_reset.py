import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("ray")

from agent_system.environments.env_package.alfworld.envs import AlfworldWorker


class _FakeBatchEnv:
    def __init__(self):
        self.seed_calls = []
        self.step_rewards = []

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        return ["observation"], {
            "intermediate_reward": [0.0],
            "won": [False],
        }

    def step(self, actions):
        assert actions == ["look"]
        intermediate_reward, won = self.step_rewards.pop(0)
        return (
            ["observation"],
            [float(won)],
            [bool(won)],
            {
                "intermediate_reward": [intermediate_reward],
                "won": [won],
            },
        )


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


def test_worker_accumulates_official_intermediate_reward_and_resets():
    worker, batch_env = _worker(deterministic_reset=False)
    batch_env.step_rewards = [(1.0, False), (0.0, False), (-1.0, False)]
    worker.reset()

    initial = worker.exact_credit_snapshot()
    assert initial["factor_ids"] == ("textworld_intermediate_reward_cumulative",)
    assert initial["values"] == (0.0,)
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (1.0,)
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (1.0,)
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (0.0,)
    worker.reset()
    assert worker.exact_credit_snapshot()["values"] == (0.0,)


def test_worker_rejects_invalid_official_intermediate_reward():
    worker, batch_env = _worker(deterministic_reset=False)
    batch_env.step_rewards = [(0.5, False)]
    worker.reset()

    with pytest.raises(RuntimeError, match="must be one of"):
        worker.step("look")


def test_worker_requires_official_intermediate_reward_after_step():
    worker, batch_env = _worker(deterministic_reset=False)
    worker.reset()

    def step_without_facts(actions):
        assert actions == ["look"]
        return ["observation"], [0.0], [False], {"won": [False]}

    batch_env.step = step_without_facts
    with pytest.raises(RuntimeError, match="info field: intermediate_reward"):
        worker.step("look")
