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

import os

import gymnasium as gym
import numpy as np
import ray
import torch
import torchvision.transforms as T
import yaml

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import get_environment

ALF_ACTION_LIST=["pass", "goto", "pick", "put", "open", "close", "toggle", "heat", "clean", "cool", "slice", "inventory", "examine", "look"]
# ALF_ITEM_LIST =

def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config

def get_obs_image(env):
    transform = T.Compose([T.ToTensor()])
    current_frames = env.get_frames()
    image_tensors = [transform(i).cuda() for i in current_frames]
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i]*= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:,:,[2,1,0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors

def compute_reward(info, multi_modal=False):
    if multi_modal:
        reward = 10.0 * float(info['won']) + float(info['goal_condition_success_rate'])
    else:
        reward = 10.0 * float(info['won'])
    return reward

class AlfworldWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds one environment instance.
    """
    
    def __init__(self, config, seed, base_env, deterministic_reset=False):
        self.env = base_env.init_env(batch_size=1)  # Each worker holds only one sub-environment
        self.seed = seed
        self.deterministic_reset = deterministic_reset
        self.env.seed(seed)
        self._exact_cumulative_intermediate_reward = 0.0
        self._exact_previous_policy_length = None
        self._exact_last_intermediate_reward = None
        self._exact_snapshot = None
        self._exact_signal = config.get("exact", {}).get("signal", "planner")

    @staticmethod
    def _single_batch_info(infos, key):
        if key not in infos:
            raise RuntimeError(f"ALFWorld EXACT requires TextWorld info field: {key}")
        raw_value = infos[key]
        if isinstance(raw_value, np.ndarray):
            if raw_value.size != 1:
                raise RuntimeError(f"ALFWorld worker expected one {key} value")
            raw_value = raw_value.reshape(-1)[0]
        elif isinstance(raw_value, (list, tuple)):
            if len(raw_value) != 1:
                raise RuntimeError(f"ALFWorld worker expected one {key} value")
            raw_value = raw_value[0]
        return raw_value

    def _refresh_exact_snapshot(self, infos, *, reset=False):
        from recipe.exact.env_probes import alfworld_intermediate_reward_snapshot

        if self._exact_signal == "predicates":
            self._exact_snapshot = self._single_batch_info(infos, "extra.exact")["snapshot"]
            return

        raw_step_reward = self._single_batch_info(infos, "intermediate_reward")
        raw_policy = self._single_batch_info(infos, "policy_commands")
        if not isinstance(raw_policy, (list, tuple)):
            raise RuntimeError("ALFWorld TextWorld policy_commands must be a sequence")
        current_policy_length = len(raw_policy)
        if raw_step_reward is None:
            # TextWorld 1.7's PddlEnv does not populate intermediate_reward.
            # Reproduce TextWorld's official definition from its own replanned
            # winning policy instead of inspecting PDDL facts or goal clauses.
            won = bool(self._single_batch_info(infos, "won"))
            lost = bool(self._single_batch_info(infos, "lost"))
            if reset:
                step_reward = 0.0
            elif won:
                step_reward = 1.0
            elif lost:
                step_reward = -1.0
            elif self._exact_previous_policy_length is None:
                raise RuntimeError("ALFWorld PDDL policy baseline is missing after reset")
            else:
                policy_delta = self._exact_previous_policy_length - current_policy_length
                step_reward = float((policy_delta > 0) - (policy_delta < 0))
        else:
            try:
                step_reward = float(raw_step_reward)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    "ALFWorld TextWorld intermediate_reward must be numeric after step"
                ) from error
        if not np.isfinite(step_reward) or step_reward not in {-1.0, 0.0, 1.0}:
            raise RuntimeError(
                "ALFWorld TextWorld intermediate_reward must be one of -1, 0, or 1"
            )
        if reset:
            if step_reward != 0.0:
                raise RuntimeError(
                    "ALFWorld reset must expose intermediate_reward == 0"
                )
            self._exact_cumulative_intermediate_reward = 0.0
        else:
            self._exact_cumulative_intermediate_reward += step_reward
        self._exact_previous_policy_length = current_policy_length
        self._exact_last_intermediate_reward = step_reward
        self._exact_snapshot = alfworld_intermediate_reward_snapshot(
            self._exact_cumulative_intermediate_reward
        )
    
    def step(self, action):
        """Execute a step in the environment"""
        actions = [action] 
        
        obs, scores, dones, infos = self.env.step(actions)
        self._refresh_exact_snapshot(infos)
        return obs, scores, dones, infos
    
    def reset(self):
        """Reset the environment"""
        if self.deterministic_reset:
            self.env.seed(self.seed)
        obs, infos = self.env.reset()
        self._exact_snapshot = None
        self._refresh_exact_snapshot(infos, reset=True)
        return obs, infos
    
    def getobs(self):
        """Get current observation image"""
        image = get_obs_image(self.env)
        image = image.cpu()  
        return image

    def exact_credit_snapshot(self):
        if self._exact_snapshot is None:
            raise RuntimeError("ALFWorld exact probe requested before reset")
        return self._exact_snapshot

    def exact_last_intermediate_reward(self):
        if self._exact_last_intermediate_reward is None:
            raise RuntimeError("ALFWorld intermediate reward requested before reset")
        return self._exact_last_intermediate_reward

class AlfworldEnvs(gym.Env):
    def __init__(self, alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
        super().__init__()
        
        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
            
        env_kwargs = env_kwargs or {}
        eval_dataset = env_kwargs.get('eval_dataset', 'eval_in_distribution')
        config = load_config_file(alf_config_path)
        self.exact_signal = env_kwargs.get('exact_signal', 'planner')
        if self.exact_signal not in {'planner', 'predicates'}:
            raise ValueError('unsupported ALFWorld EXACT signal')
        config['exact'] = {
            'signal': self.exact_signal,
            'commit_guard': bool(env_kwargs.get('commit_guard', True)),
            'horizon': int(env_kwargs.get('max_steps', 50)),
        }
        env_type = config['env']['type']
        if self.exact_signal == 'predicates' and env_type != 'AlfredTWEnv':
            raise ValueError('ALFWorld predicate certificates require TextWorld')
        base_env = get_environment(env_type)(config, train_eval='train' if is_train else eval_dataset)
        self.multi_modal = (env_type == 'AlfredThorEnv')
        self.num_processes = env_num * group_n
        self.group_n = group_n

        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(
                config,
                seed + (i // self.group_n),
                base_env,
                not is_train,
            )
            self.workers.append(worker)

        self.prev_admissible_commands = [None for _ in range(self.num_processes)]

    def step(self, actions):
        assert len(actions) == self.num_processes, \
            "The num of actions must be equal to the num of processes"

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        text_obs_list = []
        image_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        results = ray.get(futures)
        for i, (obs, scores, dones, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0]

            text_obs_list.append(obs[0])
            dones_list.append(dones[0])
            info_list.append(info)

            self.prev_admissible_commands[i] = info['admissible_commands']
            rewards_list.append(compute_reward(info, self.multi_modal))

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        """
        Send the reset command to all workers at once and collect initial obs/info from each environment.
        """
        text_obs_list = []
        image_obs_list = []
        info_list = []

        # Send reset commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.reset.remote()
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        for i, (obs, info) in enumerate(results):
            for k in info.keys():
                info[k] = info[k][0] 
            text_obs_list.append(obs[0])
            self.prev_admissible_commands[i] = info['admissible_commands']
            info_list.append(info)

        if self.multi_modal:
            image_obs_list = self.getobs()
        else:
            image_obs_list = None

        return text_obs_list, image_obs_list, info_list

    def getobs(self):
        """
        Ask each worker to return its current frame image.
        Usually needed only for multi-modal environments; otherwise can return None.
        """
        futures = []
        for worker in self.workers:
            future = worker.getobs.remote()
            futures.append(future)

        images = ray.get(futures)
        return images

    def exact_credit_snapshots(self):
        return ray.get([worker.exact_credit_snapshot.remote() for worker in self.workers])

    @property
    def get_admissible_commands(self):
        """
        Simply return the prev_admissible_commands stored by the main process.
        You could also design it to fetch after each step or another method.
        """
        return self.prev_admissible_commands

    def close(self):
        """
        Close all workers
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

def build_alfworld_envs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train=True, env_kwargs=None):
    return AlfworldEnvs(alf_config_path, seed, env_num, group_n, resources_per_worker, is_train, env_kwargs)
