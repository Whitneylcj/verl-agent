# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import numpy as np
import ray

from agent_system.environments.env_package.sokoban.sokoban import SokobanEnv


class SokobanWorker:
    """
    Ray remote actor that owns one prompt group of independent environments.

    Keeping the ``group_n`` replicas in one actor preserves the flat
    trajectory ordering expected by GRPO while avoiding one Python/Ray process
    per rollout.  The latter exhausts container PID/thread limits for the
    upstream 32 prompts x 8 rollouts configuration.
    """

    def __init__(self, mode, env_kwargs, group_n=1):
        """Initialize independent Sokoban replicas for one prompt group."""
        self.envs = [SokobanEnv(mode, **env_kwargs) for _ in range(group_n)]

    def step_many(self, actions):
        """Execute one action in each replica, preserving replica order."""
        if len(actions) != len(self.envs):
            raise ValueError(f"expected {len(self.envs)} Sokoban actions, got {len(actions)}")
        return [env.step(action) for env, action in zip(self.envs, actions)]

    def reset_many(self, seeds_for_reset):
        """Reset each replica with the corresponding (usually shared) seed."""
        if len(seeds_for_reset) != len(self.envs):
            raise ValueError(f"expected {len(self.envs)} Sokoban seeds, got {len(seeds_for_reset)}")
        return [env.reset(seed=seed) for env, seed in zip(self.envs, seeds_for_reset)]

    def render_one(self, mode_for_render, replica_idx):
        """Render one replica in this prompt group."""
        return self.envs[replica_idx].render(mode=mode_for_render)

    def render_many(self, mode_for_render):
        """Render every replica in this prompt group."""
        return [env.render(mode=mode_for_render) for env in self.envs]

    def exact_credit_snapshots(self):
        from recipe.exact.env_probes import sokoban_factor_snapshot

        snapshots = []
        for env in self.envs:
            factor_state = env.exact_reward_factor_state()
            snapshots.append(
                sokoban_factor_snapshot(
                    factor_state["event_counts"],
                    factor_state["reward_weights"],
                )
            )
        return snapshots


class SokobanMultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the Sokoban environment.
    Each Ray actor creates one group of independent SokobanEnv instances.
    The main process communicates with Ray actors to collect step/reset results.
    """

    def __init__(self, seed=0, env_num=1, group_n=1, mode="rgb_array", resources_per_worker=None, is_train=True, env_kwargs=None):
        """
        - env_num: Number of different environments
        - group_n: Number of same environments in each group (for GRPO and GiGPO)
        - env_kwargs: Dictionary of parameters for initializing SokobanEnv
        - seed: Random seed for reproducibility
        """
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.mode = mode
        # Keep reset sampling local to this environment.  The previous global
        # RNG made train/validation wrappers perturb one another and also gave
        # every validation call a different task set, so before/after metrics
        # were not paired on the same held-out rooms.
        self._reset_rng = np.random.RandomState(seed)
        self._validation_seeds = None
        if not self.is_train:
            self._validation_seeds = self._reset_rng.randint(
                2**16,
                2**32 - 1,
                size=self.env_num,
            )

        if env_kwargs is None:
            env_kwargs = {}
        if resources_per_worker is None:
            resources_per_worker = {"num_cpus": 0.1}

        # One actor per distinct prompt.  Replicas within the prompt group are
        # independent environments, but sharing their process prevents the
        # official group size from multiplying the Ray worker count.
        env_worker = ray.remote(**resources_per_worker)(SokobanWorker)
        self.workers = []
        for _ in range(self.env_num):
            worker = env_worker.remote(self.mode, env_kwargs, self.group_n)
            self.workers.append(worker)

    def _grouped(self, values):
        if len(values) != self.num_processes:
            raise ValueError(f"expected {self.num_processes} values, got {len(values)}")
        return [values[start : start + self.group_n] for start in range(0, self.num_processes, self.group_n)]

    @staticmethod
    def _flatten(grouped_values):
        return [value for group in grouped_values for value in group]

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[int], length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
            Each is a list of length self.num_processes
        """
        action_groups = self._grouped(actions)
        results = ray.get([worker.step_many.remote(group) for worker, group in zip(self.workers, action_groups)])
        results = self._flatten(results)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def _seeds_for_reset(self):
        # Training gets fresh rooms, while validation replays one fixed suite
        # so metrics from different checkpoints are directly comparable.
        if self.is_train:
            seeds = self._reset_rng.randint(0, 2**16 - 1, size=self.env_num)
        else:
            seeds = self._validation_seeds.copy()

        # repeat the seeds for each group
        return np.repeat(seeds, self.group_n).tolist()

    def reset(self):
        """
        Perform reset in parallel.
        :return: obs_list and info_list, the initial observations for each environment
        """
        seeds = self._seeds_for_reset()

        seed_groups = self._grouped(seeds)
        results = ray.get([worker.reset_many.remote(group) for worker, group in zip(self.workers, seed_groups)])
        results = self._flatten(results)
        obs_list = []
        info_list = []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)
        return obs_list, info_list

    def render(self, mode="rgb_array", env_idx=None):
        """
        Request rendering from Ray actor environments.
        Can specify env_idx to get render result from a specific environment,
        otherwise returns a list from all environments.
        """
        if env_idx is not None:
            if not 0 <= env_idx < self.num_processes:
                raise IndexError(f"Sokoban environment index out of range: {env_idx}")
            worker_idx, replica_idx = divmod(env_idx, self.group_n)
            future = self.workers[worker_idx].render_one.remote(mode, replica_idx)
            return ray.get(future)
        else:
            results = ray.get([worker.render_many.remote(mode) for worker in self.workers])
            return self._flatten(results)

    def exact_credit_snapshots(self):
        grouped_snapshots = ray.get([worker.exact_credit_snapshots.remote() for worker in self.workers])
        return self._flatten(grouped_snapshots)

    def close(self):
        """
        Close all Ray actors
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_sokoban_envs(seed=0, env_num=1, group_n=1, mode="rgb_array", resources_per_worker=None, is_train=True, env_kwargs=None):
    return SokobanMultiProcessEnv(seed, env_num, group_n, mode, resources_per_worker, is_train, env_kwargs=env_kwargs)
