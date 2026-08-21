import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("ray")

from agent_system.environments.env_package.alfworld.envs import AlfworldWorker


class _FakeBatchEnv:
    def __init__(self):
        self.seed_calls = []
        self.step_rewards = []
        self.policy_length = 3

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        self.policy_length = 3
        return ["observation"], {
            "intermediate_reward": [None],
            "won": [False],
            "lost": [False],
            "policy_commands": [[f"action-{index}" for index in range(self.policy_length)]],
        }

    def step(self, actions):
        assert actions == ["look"]
        intermediate_reward, won = self.step_rewards.pop(0)
        self.policy_length = max(0, self.policy_length - 1)
        return (
            ["observation"],
            [float(won)],
            [bool(won)],
            {
                "intermediate_reward": [intermediate_reward],
                "won": [won],
                "lost": [False],
                "policy_commands": [[f"action-{index}" for index in range(self.policy_length)]],
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


def test_worker_accepts_explicit_zero_intermediate_reward_on_reset():
    worker, batch_env = _worker(deterministic_reset=False)
    original_reset = batch_env.reset

    def reset_with_explicit_zero():
        observation, infos = original_reset()
        infos["intermediate_reward"] = [0.0]
        return observation, infos

    batch_env.reset = reset_with_explicit_zero
    worker.reset()

    assert worker.exact_credit_snapshot()["values"] == (0.0,)


def test_worker_rejects_nonzero_intermediate_reward_on_reset():
    worker, batch_env = _worker(deterministic_reset=False)
    original_reset = batch_env.reset

    def reset_with_nonzero_reward():
        observation, infos = original_reset()
        infos["intermediate_reward"] = [1.0]
        return observation, infos

    batch_env.reset = reset_with_nonzero_reward

    with pytest.raises(RuntimeError, match="reset must expose"):
        worker.reset()


def test_worker_derives_pddl_reward_from_official_policy_progress():
    worker, batch_env = _worker(deterministic_reset=False)
    batch_env.step_rewards = [(None, False), (None, False), (None, True)]
    worker.reset()

    worker.step("look")
    assert worker.exact_last_intermediate_reward() == 1.0
    assert worker.exact_credit_snapshot()["values"] == (1.0,)
    worker.step("look")
    assert worker.exact_last_intermediate_reward() == 1.0
    worker.step("look")
    assert worker.exact_last_intermediate_reward() == 1.0
    assert worker.exact_credit_snapshot()["values"] == (3.0,)


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
        return ["observation"], [0.0], [False], {
            "won": [False],
            "lost": [False],
            "policy_commands": [[]],
        }

    batch_env.step = step_without_facts
    with pytest.raises(RuntimeError, match="info field: intermediate_reward"):
        worker.step("look")
