import json

import pytest

pytest.importorskip("omegaconf")
pytest.importorskip("ray")

from agent_system.environments.env_package.alfworld.envs import AlfworldWorker


class _Variable:
    def __init__(self, name):
        self.name = name


class _Fact:
    def __init__(self, name, *arguments):
        self.name = name
        self.arguments = tuple(_Variable(argument) for argument in arguments)


def _simple_facts(placed=False):
    facts = [
        _Fact("objectType", "apple_1", "AppleType"),
        _Fact("receptacleType", "bowl_1", "BowlType"),
    ]
    if placed:
        facts.append(_Fact("inReceptacle", "apple_1", "bowl_1"))
    return facts


class _FakeBatchEnv:
    def __init__(self, gamefile):
        self.gamefile = str(gamefile)
        self.seed_calls = []
        self.step_facts = []

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        return ["observation"], {
            "extra.gamefile": [self.gamefile],
            "facts": [_simple_facts()],
            "won": [False],
        }

    def step(self, actions):
        assert actions == ["look"]
        facts, won = self.step_facts.pop(0)
        return (
            ["observation"],
            [float(won)],
            [bool(won)],
            {
                "facts": [facts],
                "won": [won],
            },
        )


class _FakeBaseEnv:
    def __init__(self, gamefile):
        self.batch_env = _FakeBatchEnv(gamefile)

    def init_env(self, batch_size):
        assert batch_size == 1
        return self.batch_env


def _worker(tmp_path, deterministic_reset):
    gamefile = tmp_path / "game.tw-pddl"
    gamefile.write_text("", encoding="utf-8")
    (tmp_path / "traj_data.json").write_text(
        json.dumps(
            {
                "task_type": "pick_and_place_simple",
                "pddl_params": {
                    "object_target": "Apple",
                    "parent_target": "Bowl",
                    "object_sliced": False,
                },
            }
        ),
        encoding="utf-8",
    )
    base_env = _FakeBaseEnv(gamefile)
    worker = AlfworldWorker({}, 17, base_env, deterministic_reset=deterministic_reset)
    return worker, base_env.batch_env


def test_validation_worker_reseeds_before_every_reset(tmp_path):
    worker, batch_env = _worker(tmp_path, deterministic_reset=True)

    worker.reset()
    worker.reset()

    assert batch_env.seed_calls == [17, 17, 17]


def test_training_worker_advances_without_reseeding(tmp_path):
    worker, batch_env = _worker(tmp_path, deterministic_reset=False)

    worker.reset()
    worker.reset()

    assert batch_env.seed_calls == [17]


def test_worker_tracks_official_pddl_facts_and_terminal_return(tmp_path):
    worker, batch_env = _worker(tmp_path, deterministic_reset=False)
    batch_env.step_facts = [(_simple_facts(), False), (_simple_facts(placed=True), True)]
    worker.reset()

    initial = worker.exact_credit_snapshot()
    assert initial["factor_ids"] == ("terminal_success", "target_placed")
    assert initial["values"] == (0.0, 0.0)
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (0.0, 0.0)
    worker.step("look")
    assert worker.exact_credit_snapshot()["values"] == (1.0, 1.0)


def test_worker_fails_closed_when_factors_disagree_with_won(tmp_path):
    worker, batch_env = _worker(tmp_path, deterministic_reset=False)
    batch_env.step_facts = [(_simple_facts(placed=True), False)]
    worker.reset()

    with pytest.raises(RuntimeError, match="disagree with the official PDDL won"):
        worker.step("look")


def test_worker_requires_structured_facts_after_step(tmp_path):
    worker, batch_env = _worker(tmp_path, deterministic_reset=False)
    worker.reset()

    def step_without_facts(actions):
        assert actions == ["look"]
        return ["observation"], [0.0], [False], {"won": [False]}

    batch_env.step = step_without_facts
    with pytest.raises(RuntimeError, match="info field: facts"):
        worker.step("look")
