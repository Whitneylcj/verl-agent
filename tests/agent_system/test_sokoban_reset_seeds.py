import numpy as np

from agent_system.environments.env_package.sokoban.envs import SokobanMultiProcessEnv


def _seed_only_env(*, is_train: bool, seed: int, env_num: int = 3, group_n: int = 2):
    env = object.__new__(SokobanMultiProcessEnv)
    env.is_train = is_train
    env.env_num = env_num
    env.group_n = group_n
    env._reset_rng = np.random.RandomState(seed)
    env._validation_seeds = None
    if not is_train:
        env._validation_seeds = env._reset_rng.randint(2**16, 2**32 - 1, size=env_num)
    return env


def test_validation_reuses_fixed_grouped_room_seeds():
    env = _seed_only_env(is_train=False, seed=1000)

    first = np.asarray(env._seeds_for_reset())
    second = np.asarray(env._seeds_for_reset())

    np.testing.assert_array_equal(first, second)
    assert all(first[index] == first[index + 1] for index in range(0, len(first), 2))


def test_training_draws_fresh_grouped_room_seeds():
    env = _seed_only_env(is_train=True, seed=0)

    first = np.asarray(env._seeds_for_reset())
    second = np.asarray(env._seeds_for_reset())

    assert not np.array_equal(first, second)
    assert all(first[index] == first[index + 1] for index in range(0, len(first), 2))
