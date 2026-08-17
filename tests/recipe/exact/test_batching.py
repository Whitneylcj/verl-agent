from types import SimpleNamespace

import numpy as np
import torch

from agent_system.multi_turn_rollout.utils import adjust_batch
from verl import DataProto


def _config():
    return SimpleNamespace(
        algorithm=SimpleNamespace(adv_estimator="exact", use_kl_in_reward=False),
        actor_rollout_ref=SimpleNamespace(
            rollout=SimpleNamespace(log_prob_micro_batch_size_per_gpu=2),
            ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=2),
            actor=SimpleNamespace(
                use_kl_loss=True,
                ppo_micro_batch_size_per_gpu=2,
                ppo_mini_batch_size=32,
            ),
        ),
        trainer=SimpleNamespace(n_gpus_per_node=1, nnodes=1),
    )


def _batch(size):
    return DataProto.from_dict(
        tensors={"input_ids": torch.arange(size).reshape(size, 1)},
        non_tensors={"traj_uid": np.asarray([f"t-{index}" for index in range(size)], dtype=object)},
    )


def test_exact_batch_does_not_pad_to_ppo_mini_batch_size():
    adjusted = adjust_batch(_config(), _batch(98))

    assert len(adjusted) == 98
    assert not adjusted.non_tensor_batch["exact_padding"].any()
    assert not adjusted.non_tensor_batch["rollout_padding"].any()


def test_exact_batch_only_pads_to_worker_micro_batch_divisor():
    adjusted = adjust_batch(_config(), _batch(99))

    assert len(adjusted) == 100
    assert adjusted.non_tensor_batch["exact_padding"].tolist() == [False] * 99 + [True]
    assert adjusted.non_tensor_batch["rollout_padding"].tolist() == [False] * 99 + [True]


def test_exact_multimodal_batch_also_avoids_ppo_mini_batch_padding():
    batch = _batch(98)
    batch.non_tensor_batch["multi_modal_inputs"] = np.asarray([{} for _ in range(98)], dtype=object)

    adjusted = adjust_batch(_config(), batch)

    assert len(adjusted) == 98
    assert not adjusted.non_tensor_batch["exact_padding"].any()
