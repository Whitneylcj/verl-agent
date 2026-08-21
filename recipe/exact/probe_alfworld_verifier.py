"""Run the EXACT verifier against official ALFWorld TextWorld games.

This is a remote-only integration probe. It follows ALFWorld's hand-coded
expert without printing benchmark instance paths, observations, or actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from agent_system.environments.env_package.alfworld.alfworld.agents.environment import (
    get_environment,
)
from agent_system.environments.env_package.alfworld.envs import (
    AlfworldWorker,
    load_config_file,
)
from recipe.exact.core_exact import IdentityPotential, build_scoped_conserved_atoms
from recipe.exact.credit_spec import FactorSnapshot


def _single(value: Any, name: str) -> Any:
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__") or len(value) != 1:
        raise RuntimeError(f"expected exactly one batched ALFWorld {name} value")
    return value[0]


def _compile_real_trajectory(
    snapshots: list[dict[str, Any]],
    episode_return: float,
) -> float:
    typed_snapshots = tuple(
        FactorSnapshot(
            checkpoint_id=checkpoint_id,
            factor_ids=tuple(snapshot["factor_ids"]),
            values=np.asarray(snapshot["values"], dtype=np.float64),
            read_sets=snapshot["read_sets"],
            schema_version=snapshot["schema_version"],
            channel_roles=snapshot["channel_roles"],
            potential_weights=np.asarray(
                snapshot["potential_weights"],
                dtype=np.float64,
            ),
            source_revision=snapshot["source_revision"],
        )
        for checkpoint_id, snapshot in enumerate(snapshots)
    )
    potential = IdentityPotential(
        factor_ids=typed_snapshots[0].factor_ids,
        weights=typed_snapshots[0].potential_weights,
    )
    conserved = build_scoped_conserved_atoms(
        typed_snapshots,
        episode_return,
        potential,
    )
    if any(value != 0.0 for value in conserved.channel_target_masses.values()):
        raise RuntimeError("ALFWorld process verifier changed the conserved return target")
    if conserved.opaque_target_atom.value != episode_return:
        raise RuntimeError("ALFWorld binary training return must remain opaque")
    return conserved.conservation_error


def _probe_game(base_env: Any, gamefile: str, max_steps: int) -> dict[str, Any]:
    base_env.game_files = [gamefile]
    base_env.num_games = 1
    base_env.use_expert = True
    worker = AlfworldWorker({}, seed=0, base_env=base_env, deterministic_reset=True)
    try:
        _, infos = worker.reset()
        snapshots = [worker.exact_credit_snapshot()]
        factor_ids = snapshots[0]["factor_ids"]
        if factor_ids != ("textworld_intermediate_reward_cumulative",):
            raise RuntimeError("ALFWorld probe exposed an unexpected factor schema")
        total_reward = 0.0
        done = False
        intermediate_rewards = []
        previous_policy_length = len(
            _single(infos["policy_commands"], "policy commands")
        )

        for step_count in range(1, max_steps + 1):
            plan = _single(infos["extra.expert_plan"], "expert plan")
            if not plan:
                raise RuntimeError("ALFWorld expert plan ended before official success")
            action = plan[0]
            admissible = _single(infos["admissible_commands"], "admissible commands")
            if action not in admissible:
                raise RuntimeError("ALFWorld expert proposed a non-admissible action")

            _, _, dones, infos = worker.step(action)
            won = bool(_single(infos["won"], "won"))
            done = bool(_single(dones, "done"))
            current_policy_length = len(
                _single(infos["policy_commands"], "policy commands")
            )
            lost = bool(_single(infos["lost"], "lost"))
            if won:
                expected_intermediate_reward = 1.0
            elif lost:
                expected_intermediate_reward = -1.0
            else:
                policy_delta = previous_policy_length - current_policy_length
                expected_intermediate_reward = float(
                    (policy_delta > 0) - (policy_delta < 0)
                )
            intermediate_reward = worker.exact_last_intermediate_reward()
            if intermediate_reward not in {-1.0, 0.0, 1.0}:
                raise RuntimeError("ALFWorld emitted an invalid intermediate_reward")
            if intermediate_reward != expected_intermediate_reward:
                raise RuntimeError(
                    "ALFWorld verifier differs from TextWorld winning-policy progress"
                )
            total_reward += 10.0 * float(won)
            snapshot = worker.exact_credit_snapshot()
            if snapshot["factor_ids"] != factor_ids:
                raise RuntimeError("ALFWorld factor schema changed during expert rollout")
            snapshot_delta = snapshot["values"][0] - snapshots[-1]["values"][0]
            if snapshot_delta != intermediate_reward:
                raise RuntimeError(
                    "ALFWorld checkpoint delta differs from official intermediate_reward"
                )
            intermediate_rewards.append(intermediate_reward)
            previous_policy_length = current_policy_length
            snapshots.append(snapshot)
            if done:
                break
        else:
            step_count = max_steps

        if not done or not bool(_single(infos["won"], "won")):
            raise RuntimeError("ALFWorld expert did not reach official success")
        if total_reward != 10.0:
            raise RuntimeError("ALFWorld training return no longer equals 10 * won")
        process_change_count = sum(value != 0.0 for value in intermediate_rewards)
        if process_change_count == 0:
            raise RuntimeError("ALFWorld process verifier never changed on an expert plan")
        conservation_error = _compile_real_trajectory(snapshots, total_reward)

        return {
            "steps": step_count,
            "factor_ids": factor_ids,
            "process_change_count": process_change_count,
            "intermediate_reward_counts": {
                str(value): intermediate_rewards.count(value)
                for value in (-1.0, 0.0, 1.0)
            },
            "episode_return": total_reward,
            "conservation_error": conservation_error,
        }
    finally:
        worker.env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("agent_system/environments/env_package/alfworld/configs/config_tw.yaml"),
    )
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--games", type=int, default=6)
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.games <= 0:
        raise ValueError("--games must be positive")

    config = load_config_file(args.config)
    config["env"]["domain_randomization"] = False
    env_type = config["env"]["type"]
    base_env = get_environment(env_type)(config, train_eval="train")
    selected = list(base_env.game_files[: args.games])
    if len(selected) < args.games:
        raise RuntimeError(
            f"ALFWorld data contains only {len(selected)} games; requested {args.games}"
        )
    results = [
        _probe_game(base_env, gamefile, args.max_steps)
        for gamefile in selected
    ]
    print(json.dumps({"status": "passed", "tasks": results}, indent=2))


if __name__ == "__main__":
    main()
