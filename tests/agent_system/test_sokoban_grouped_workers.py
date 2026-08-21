from types import SimpleNamespace

import pytest

from agent_system.environments.env_package.sokoban import envs as sokoban_envs


class _RemoteMethod:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def remote(self, *args):
        self.calls.append(args)
        return self.function(*args)


def _fake_worker(worker_id):
    return SimpleNamespace(
        step_many=_RemoteMethod(lambda actions: [(f"obs-{worker_id}-{action}", float(action), action == 0, {"action": action}) for action in actions]),
        reset_many=_RemoteMethod(lambda seeds: [(f"reset-{worker_id}-{seed}", {"seed": seed}) for seed in seeds]),
        render_one=_RemoteMethod(lambda mode, replica_idx: f"render-{worker_id}-{replica_idx}-{mode}"),
        render_many=_RemoteMethod(lambda mode: [f"render-{worker_id}-{replica_idx}-{mode}" for replica_idx in range(2)]),
        exact_credit_snapshots=_RemoteMethod(lambda: [{"worker": worker_id, "replica": replica_idx} for replica_idx in range(2)]),
    )


@pytest.fixture
def grouped_env(monkeypatch):
    monkeypatch.setattr(sokoban_envs.ray, "get", lambda values: values)
    monkeypatch.setattr(sokoban_envs.ray, "kill", lambda worker: None)
    env = object.__new__(sokoban_envs.SokobanMultiProcessEnv)
    env.group_n = 2
    env.env_num = 2
    env.num_processes = 4
    env.workers = [_fake_worker(0), _fake_worker(1)]
    return env


def test_grouped_workers_preserve_flat_trajectory_order(grouped_env):
    observations, rewards, dones, infos = grouped_env.step([1, 2, 3, 0])

    assert grouped_env.workers[0].step_many.calls == [([1, 2],)]
    assert grouped_env.workers[1].step_many.calls == [([3, 0],)]
    assert observations == ["obs-0-1", "obs-0-2", "obs-1-3", "obs-1-0"]
    assert rewards == [1.0, 2.0, 3.0, 0.0]
    assert dones == [False, False, False, True]
    assert [info["action"] for info in infos] == [1, 2, 3, 0]


def test_grouped_workers_preserve_reset_render_and_snapshot_order(grouped_env):
    grouped_env._seeds_for_reset = lambda: [11, 11, 22, 22]

    observations, infos = grouped_env.reset()

    assert grouped_env.workers[0].reset_many.calls == [([11, 11],)]
    assert grouped_env.workers[1].reset_many.calls == [([22, 22],)]
    assert observations == ["reset-0-11", "reset-0-11", "reset-1-22", "reset-1-22"]
    assert [info["seed"] for info in infos] == [11, 11, 22, 22]
    assert grouped_env.render(mode="state") == [
        "render-0-0-state",
        "render-0-1-state",
        "render-1-0-state",
        "render-1-1-state",
    ]
    assert grouped_env.render(mode="list", env_idx=3) == "render-1-1-list"
    assert grouped_env.exact_credit_snapshots() == [
        {"worker": 0, "replica": 0},
        {"worker": 0, "replica": 1},
        {"worker": 1, "replica": 0},
        {"worker": 1, "replica": 1},
    ]


def test_grouped_workers_reject_wrong_flat_batch_size(grouped_env):
    with pytest.raises(ValueError, match="expected 4 values, got 3"):
        grouped_env.step([1, 2, 3])

    with pytest.raises(IndexError, match="out of range"):
        grouped_env.render(env_idx=4)
