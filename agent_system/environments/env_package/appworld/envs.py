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
import time

import numpy as np
import ray

import appworld
from appworld import AppWorld, load_task_ids


def appworld_execution_succeeded(observation):
    """Return whether AppWorld executed the projected API call without error."""

    return not str(observation).lstrip().startswith("Execution failed.")


def load_available_ports(port_file="appworld_ports.ports"):
    """
    Load available port list from file
    """
    port_file = os.environ.get("APPWORLD_PORT_FILE", port_file)
    if not os.path.exists(port_file):
        raise FileNotFoundError(f"Port file {port_file} does not exist. Please run the service startup script first.")

    ports = []
    with open(port_file) as f:
        for line in f:
            line = line.strip()
            if line and line.isdigit():
                ports.append(int(line))

    if not ports:
        raise ValueError(f"No valid ports found in port file {port_file}.")

    return ports


class AppWorldWorker:
    """
    Ray Actor that holds an instance of AppWorld and operates the environment
    based on method calls from the main process.
    """

    def __init__(self, worker_id, max_interactions, port):
        self.env = None
        self.current_step_count = 0
        self.max_interactions = max_interactions
        self.worker_id = worker_id

        self.url = f"http://127.0.0.1:{port}"
        self._exact_factor_ids = None
        self._exact_snapshot = None
        self._exact_read_sets = None
        self._exact_effect_registry = None

    def _build_exact_graph_schema(self, evaluation):
        from appworld.common.inspect import get_name_to_model
        from recipe.exact.appworld_schema import (
            build_appworld_effect_registry,
            compile_appworld_factor_reads,
            detect_appworld_source_revision,
        )
        from recipe.exact.env_probes import appworld_factor_snapshot

        bootstrap_snapshot = appworld_factor_snapshot(evaluation)
        factor_ids = tuple(bootstrap_snapshot["factor_ids"])
        api_docs = self.env.task.api_docs
        app_names = set(api_docs) | set(self.env.task.allowed_apps) | {"admin"}
        app_to_model_names = {}
        for app_name in sorted(app_names):
            try:
                app_to_model_names[app_name] = tuple(sorted(get_name_to_model(app_name)))
            except (ImportError, ModuleNotFoundError):
                app_to_model_names[app_name] = ()
        registry = build_appworld_effect_registry(
            api_docs=api_docs,
            app_to_model_names=app_to_model_names,
            appworld_version=str(appworld.__version__),
            appworld_source_revision=detect_appworld_source_revision(appworld.__file__),
        )
        all_resources = tuple(registry["all_model_resources"])
        compile_error = None
        try:
            ground_truth = self.env.task.ground_truth
            if ground_truth is None or not ground_truth.evaluation_code:
                raise ValueError("task ground truth exposes no evaluation code")
            factor_reads = compile_appworld_factor_reads(
                ground_truth.evaluation_code,
                all_resources,
                known_values={
                    "public_data": ground_truth.public_data,
                    "private_data": ground_truth.private_data,
                    "main_user": self.env.task.supervisor,
                },
            )
            read_sets = {factor_id: tuple(factor_reads.read_sets.get(factor_id, all_resources)) for factor_id in factor_ids}
            opaque_factor_ids = set(factor_reads.opaque_factor_ids) | (set(factor_ids) - set(factor_reads.read_sets))
        except (SyntaxError, TypeError, ValueError) as error:
            read_sets = {factor_id: all_resources for factor_id in factor_ids}
            opaque_factor_ids = set(factor_ids)
            compile_error = f"{type(error).__name__}: {error}"
        registry["factor_count"] = len(factor_ids)
        registry["opaque_factor_count"] = len(opaque_factor_ids)
        registry["factor_schema_compile_fallback"] = compile_error is not None
        registry["factor_schema_compile_error"] = compile_error
        self._exact_read_sets = read_sets
        self._exact_effect_registry = registry

    def _evaluate_exact(self):
        from recipe.exact.env_probes import appworld_factor_snapshot

        evaluation = self.env.evaluate()
        if self._exact_read_sets is None:
            self._build_exact_graph_schema(evaluation)
        snapshot = appworld_factor_snapshot(
            evaluation,
            expected_factor_ids=self._exact_factor_ids,
            read_sets=self._exact_read_sets,
        )
        if self._exact_factor_ids is None:
            self._exact_factor_ids = snapshot["factor_ids"]
        return evaluation, snapshot

    def reset(self, task_id):
        """Reset the environment with a new task."""
        if self.env is not None:
            self.env.close()
            time.sleep(2)

        self.current_step_count = 0
        self._exact_factor_ids = None
        self._exact_read_sets = None
        self._exact_effect_registry = None

        self.env = AppWorld(
            task_id=task_id,
            experiment_name=f"default_{self.worker_id}",
            remote_environment_url=self.url,
        )

        completion_before_probe = self.env.task_completed()
        _, first_snapshot = self._evaluate_exact()
        _, second_snapshot = self._evaluate_exact()
        completion_after_probe = self.env.task_completed()
        if first_snapshot != second_snapshot or completion_before_probe != completion_after_probe:
            raise RuntimeError("AppWorld evaluate() is not side-effect-free on this task")
        self._exact_snapshot = first_snapshot

        obs = self.env.task.instruction
        info = {
            "task_id": task_id,
            "supervisor": dict(self.env.task.supervisor),
            "allowed_apps": list(self.env.task.allowed_apps),
        }
        return obs, info

    def step(self, action):
        """Execute one step in the environment."""
        if self.env is None:
            raise RuntimeError("Environment not reset before step. Please call reset() first.")

        self.current_step_count += 1

        obs = self.env.execute(action)
        execution_valid = appworld_execution_succeeded(obs)

        evaluation, self._exact_snapshot = self._evaluate_exact()

        done = self.env.task_completed() or (self.current_step_count >= self.max_interactions)

        if done:
            is_success = evaluation.success

            reward = 10.0 if is_success else 0.0
            info = {
                "won": is_success,
                "step_count": self.current_step_count,
                "is_action_execution_valid": execution_valid,
            }
        else:
            reward = 0.0
            info = {
                "won": False,
                "step_count": self.current_step_count,
                "is_action_execution_valid": execution_valid,
            }

        return obs, reward, done, info

    def exact_credit_snapshot(self):
        if self._exact_snapshot is None:
            raise RuntimeError("AppWorld exact probe requested before reset")
        return self._exact_snapshot

    def exact_effect_schema(self):
        if self._exact_effect_registry is None:
            raise RuntimeError("AppWorld exact effect schema requested before reset")
        return self._exact_effect_registry

    def close(self):
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class AppWorldEnvs:
    """
    A Ray-based distributed wrapper for AppWorld.
    - Creates multiple Ray actors, each holding a separate AppWorld instance.
    - Implements Gym-style interfaces such as step() / reset() / close().
    """

    def __init__(self, dataset_name, max_interactions, seed, env_num, group_n, start_server_id, resources_per_worker, port_file="appworld_ports.ports"):
        super().__init__()

        self.dataset_name = dataset_name
        self.max_interactions = max_interactions
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.rng = np.random.default_rng(seed)
        self.task_ids = load_task_ids(dataset_name)

        if self.env_num > len(self.task_ids):
            raise ValueError(f"Env_num ({self.env_num}) exceeds available task_ids in '{self.dataset_name}' ({len(self.task_ids)}). Please reducing env_num to {len(self.task_ids)}.")

        all_ports = load_available_ports(port_file)

        self.available_ports = all_ports[start_server_id : start_server_id + self.num_processes]

        # Check if we have enough ports
        if len(self.available_ports) < self.num_processes:
            raise ValueError(f"Need {self.num_processes} ports, but only {len(self.available_ports)} available ports. Please ensure enough service instances are started.")

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        # Create Ray actors (workers)
        env_worker = ray.remote(**resources_per_worker)(AppWorldWorker)
        self.workers = []
        for i in range(self.num_processes):
            port = self.available_ports[i]
            worker = env_worker.remote(worker_id=start_server_id + i, max_interactions=self.max_interactions, port=port)
            self.workers.append(worker)

    def step(self, actions):
        """
        actions: Must be a list with length equal to self.num_processes,
        each sent to the corresponding worker.

        Return format follows Gym's step() convention:
            observations, rewards, dones, infos
        """
        assert len(actions) == self.num_processes, "The length of actions must match the number of processes."

        # Send step commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.step.remote(actions[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        """
        Reset all worker environments simultaneously,
        returning each environment's initial observation and info.
        """
        # randomly select self.env_num task_id from self.task_ids
        task_id = self.rng.choice(self.task_ids, self.env_num, replace=False)
        # repeat task_id group_n times
        task_id = np.repeat(task_id, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(task_id[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)

        obs_list = []
        info_list = []

        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def exact_credit_snapshots(self):
        return ray.get([worker.exact_credit_snapshot.remote() for worker in self.workers])

    def exact_effect_schemas(self):
        return ray.get([worker.exact_effect_schema.remote() for worker in self.workers])

    def close(self):
        """Close all workers."""
        # Send close commands to all workers
        futures = []
        for worker in self.workers:
            future = worker.close.remote()
            futures.append(future)

        # Wait for all workers to close
        ray.get(futures)

        # Shutdown Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def render(self):
        """Implement this if visualization is needed."""
        pass


def build_appworld_envs(
    dataset_name="train",
    max_interactions=50,
    seed=0,
    env_num=1,
    group_n=1,
    start_server_id=0,
    resources_per_worker=None,
    port_file="appworld_ports.ports",
):

    if resources_per_worker is None:
        resources_per_worker = {"num_cpus": 0.1}

    return AppWorldEnvs(
        dataset_name=dataset_name,
        max_interactions=max_interactions,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        start_server_id=start_server_id,
        resources_per_worker=resources_per_worker,
        port_file=port_file,
    )
