import numpy as np
import torch

from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage


def test_grpo_identical_float32_returns_have_exactly_zero_advantage():
    rewards = torch.zeros(4, 3, dtype=torch.float32)
    rewards[:, -1] = torch.tensor(-1.500000238418579, dtype=torch.float32)
    response_mask = torch.ones_like(rewards)
    prompt_ids = np.array(["a", "a", "b", "b"], dtype=object)
    trajectory_ids = np.array(["a1", "a2", "b1", "b2"], dtype=object)

    advantages, returns = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=prompt_ids,
        traj_index=trajectory_ids,
    )

    torch.testing.assert_close(advantages, torch.zeros_like(advantages), rtol=0, atol=0)
    torch.testing.assert_close(returns, torch.zeros_like(returns), rtol=0, atol=0)


def test_grpo_nonconstant_group_keeps_standardized_outcome_signal():
    rewards = torch.tensor([[0.0, 1.0], [0.0, 3.0]], dtype=torch.float32)
    response_mask = torch.ones_like(rewards)

    advantages, _ = compute_grpo_outcome_advantage(
        token_level_rewards=rewards,
        response_mask=response_mask,
        index=np.array(["group", "group"], dtype=object),
        traj_index=np.array(["t1", "t2"], dtype=object),
    )

    expected = torch.tensor(
        [[-2**-0.5, -2**-0.5], [2**-0.5, 2**-0.5]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(advantages, expected, rtol=1e-5, atol=1e-6)
