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
    Ray remote actor that replaces the worker function.
    Each actor holds its own independent instance of SokobanEnv.
    """

    def __init__(self, mode, env_kwargs):
        """Initialize the Sokoban environment in this worker"""
        self.env = SokobanEnv(mode, **env_kwargs)

    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self, seed_for_reset):
        """Reset the environment with given seed"""
        obs, info = self.env.reset(seed=seed_for_reset)
        return obs, info

    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered

    def exact_credit_snapshot(self):
        from recipe.exact.env_probes import sokoban_factor_snapshot

        factor_state = self.env.exact_reward_factor_state()
        return sokoban_factor_snapshot(
            factor_state["event_counts"],
            factor_state["reward_weights"],
        )


class SokobanMultiProcessEnv(gym.Env):
    """
    Ray-based wrapper for the Sokoban environment.
    Each Ray actor creates an independent SokobanEnv instance.
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

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(SokobanWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(self.mode, env_kwargs)
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list[int], length must match self.num_processes
        :return:
            obs_list, reward_list, done_list, info_list
            Each is a list of length self.num_processes
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
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

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
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
            future = self.workers[env_idx].render.remote(mode)
            return ray.get(future)
        else:
            futures = []
            for worker in self.workers:
                future = worker.render.remote(mode)
                futures.append(future)
            results = ray.get(futures)
            return results

    def exact_credit_snapshots(self):
        return ray.get([worker.exact_credit_snapshot.remote() for worker in self.workers])

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
