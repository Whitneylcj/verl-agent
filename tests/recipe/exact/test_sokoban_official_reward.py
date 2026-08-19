import numpy as np
import pytest

pytest.importorskip("gym_sokoban")
pytest.importorskip("ray")

from agent_system.environments.env_package.sokoban.sokoban.env import SokobanEnv
from recipe.exact.core_exact import IdentityPotential, build_scoped_conserved_atoms
from recipe.exact.credit_spec import FactorSnapshot
from recipe.exact.env_probes import sokoban_factor_snapshot


def _environment(room_fixed, room_state, boxes_on_target):
    env = object.__new__(SokobanEnv)
    env.mode = "tiny_rgb_array"
    env.room_fixed = np.asarray(room_fixed)
    env.room_state = np.asarray(room_state)
    env.player_position = np.argwhere(env.room_state == 5)[0]
    env.num_boxes = 1
    env.boxes_on_target = boxes_on_target
    env.num_env_steps = 0
    env.max_steps = 20
    env.penalty_for_step = -0.1
    env.reward_box_on_target = 1.0
    env.penalty_box_off_target = -1.0
    env.reward_finished = 10.0
    env._exact_reward_event_counts = {factor_id: 0 for factor_id in env.EXACT_REWARD_FACTOR_IDS}
    return env


def _snapshot(env):
    state = env.exact_reward_factor_state()
    return sokoban_factor_snapshot(state["event_counts"], state["reward_weights"])


def _factor_delta(before, after):
    assert before["factor_ids"] == after["factor_ids"]
    assert before["potential_weights"] == after["potential_weights"]
    values_before = np.asarray(before["values"])
    values_after = np.asarray(after["values"])
    weights = np.asarray(after["potential_weights"])
    return float((values_after - values_before) @ weights)


def _typed_snapshot(checkpoint_id, snapshot):
    return FactorSnapshot(
        checkpoint_id=checkpoint_id,
        factor_ids=snapshot["factor_ids"],
        values=np.asarray(snapshot["values"]),
        read_sets=snapshot["read_sets"],
        schema_version=snapshot["schema_version"],
        potential_weights=np.asarray(snapshot["potential_weights"]),
        channel_roles=snapshot["channel_roles"],
        source_revision=snapshot["source_revision"],
    )


@pytest.mark.parametrize(
    ("room_fixed", "room_state", "boxes_on_target", "action", "expected_reward"),
    (
        (
            [[0, 0, 0, 0, 0], [0, 1, 1, 2, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 5, 4, 2, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
            0,
            4,
            10.9,
        ),
        (
            [[0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 5, 3, 1, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
            1,
            4,
            -1.1,
        ),
        (
            [[0, 0, 0, 0, 0], [0, 1, 1, 2, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 5, 1, 2, 0], [0, 1, 4, 1, 0], [0, 0, 0, 0, 0]],
            0,
            4,
            -0.1,
        ),
    ),
)
def test_official_gym_step_reward_equals_factor_delta(
    room_fixed,
    room_state,
    boxes_on_target,
    action,
    expected_reward,
):
    env = _environment(room_fixed, room_state, boxes_on_target)
    before = _snapshot(env)
    _, reward, _, _ = env.step(action)
    after = _snapshot(env)

    assert reward == pytest.approx(expected_reward)
    assert _factor_delta(before, after) == pytest.approx(reward)


def test_projected_invalid_action_uses_the_official_step_penalty_factor():
    env = _environment(
        [[0, 0, 0, 0, 0], [0, 1, 1, 2, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 5, 1, 2, 0], [0, 1, 4, 1, 0], [0, 0, 0, 0, 0]],
        0,
    )
    before = _snapshot(env)
    _, reward, done, info = env.step(env.INVALID_ACTION)
    after = _snapshot(env)

    assert reward == pytest.approx(-0.1)
    assert _factor_delta(before, after) == pytest.approx(reward)
    assert env.num_env_steps == 1
    assert env.reward_last == pytest.approx(reward)
    assert not done
    assert not info["action_is_effective"]


def test_projected_invalid_action_terminates_at_official_horizon():
    env = _environment(
        [[0, 0, 0, 0, 0], [0, 1, 1, 2, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 5, 1, 2, 0], [0, 1, 4, 1, 0], [0, 0, 0, 0, 0]],
        0,
    )
    env.max_steps = 1

    _, reward, done, info = env.step(env.INVALID_ACTION)

    assert reward == pytest.approx(-0.1)
    assert env.num_env_steps == env.max_steps
    assert env._exact_reward_event_counts["step_penalty"] == env.num_env_steps
    assert done
    assert not info["action_is_effective"]
    assert not info["won"]


def test_truncated_episode_has_zero_local_closures_and_opaque_target():
    env = _environment(
        [[0, 0, 0, 0, 0], [0, 1, 1, 2, 0], [0, 1, 1, 1, 0], [0, 0, 0, 0, 0]],
        [[0, 0, 0, 0, 0], [0, 5, 1, 2, 0], [0, 1, 4, 1, 0], [0, 0, 0, 0, 0]],
        0,
    )
    snapshots = [_snapshot(env)]
    rewards = []
    for action in (env.INVALID_ACTION, 4):
        _, reward, _, _ = env.step(action)
        rewards.append(reward)
        snapshots.append(_snapshot(env))

    typed_snapshots = tuple(_typed_snapshot(checkpoint_id, snapshot) for checkpoint_id, snapshot in enumerate(snapshots))
    potential = IdentityPotential(
        typed_snapshots[0].factor_ids,
        typed_snapshots[0].potential_weights,
    )
    conserved = build_scoped_conserved_atoms(
        typed_snapshots,
        episode_return=sum(rewards),
        potential=potential,
        tolerance=1e-12,
    )

    assert conserved.closure_abs_mass == pytest.approx(0.0, abs=1e-12)
    assert conserved.opaque_target_atom.value == pytest.approx(0.0, abs=1e-12)
    assert sum(conserved.channel_target_masses.values()) == pytest.approx(sum(rewards))
