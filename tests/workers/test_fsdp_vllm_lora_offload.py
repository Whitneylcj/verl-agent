from collections import OrderedDict

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("vllm")

from verl.utils.debug import performance
from verl.workers.sharding_manager import fsdp_vllm


class _FakeDevice:
    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0

    def mem_get_info(self):
        return 1, 1

    def empty_cache(self):
        return None


def test_cpu_lora_state_offloads_actor_before_vllm_wake(monkeypatch):
    events = []

    class _FakePeftModel:
        peft_config = {"default": object()}

    class _FakeModule:
        _fsdp_wrapped_module = _FakePeftModel()

    class _FakeInferenceEngine:
        def wake_up(self, tags=None):
            events.append(f"wake_{tags[0]}")

    manager = object.__new__(fsdp_vllm.FSDPVLLMShardingManager)
    manager.module = _FakeModule()
    manager.inference_engine = _FakeInferenceEngine()
    manager.offload_param = True
    manager.layered_summon = True
    manager.base_sync_done = True
    manager.full_params = False
    manager.device_mesh = None
    manager.update_params = lambda params, peft_config=None: events.append("update_lora")

    fake_device = _FakeDevice()
    monkeypatch.setattr(performance, "get_torch_device", lambda: fake_device)
    monkeypatch.setattr(fsdp_vllm, "get_torch_device", lambda: fake_device)
    monkeypatch.setattr(fsdp_vllm, "PeftModel", _FakePeftModel)
    monkeypatch.setattr(fsdp_vllm, "fsdp_version", lambda module: 1)
    monkeypatch.setattr(
        fsdp_vllm,
        "layered_summon_lora_params",
        lambda module: OrderedDict(lora=torch.zeros(1, device="cpu")),
    )
    monkeypatch.setattr(
        fsdp_vllm,
        "load_fsdp_model_to_gpu",
        lambda module: events.append("load_actor"),
    )
    monkeypatch.setattr(
        fsdp_vllm,
        "offload_fsdp_model_to_cpu",
        lambda module: events.append("offload_actor"),
    )
    monkeypatch.setattr(fsdp_vllm, "vllm_version", "0.11.0")

    manager.__enter__()

    assert events == [
        "load_actor",
        "offload_actor",
        "wake_weights",
        "update_lora",
        "wake_kv_cache",
    ]
