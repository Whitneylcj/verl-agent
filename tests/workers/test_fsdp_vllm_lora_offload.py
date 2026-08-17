from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("peft")
pytest.importorskip("vllm")

from verl.utils.debug import performance
from verl.workers.sharding_manager import fsdp_vllm
from peft.utils import save_and_load as peft_save


class _FakeDevice:
    def memory_allocated(self):
        return 0

    def memory_reserved(self):
        return 0

    def mem_get_info(self):
        return 1, 1

    def empty_cache(self):
        return None

    def synchronize(self):
        self.events.append("synchronize_offload")


def test_cpu_lora_state_offloads_actor_before_vllm_wake(monkeypatch):
    events = []

    class _FakePeftModel:
        peft_config = {"default": SimpleNamespace(bias="none", use_dora=False)}

        def named_modules(self):
            layer = SimpleNamespace(weight=torch.ones(1, device="cpu"), bias=None)
            return [
                ("", self),
                ("base_model.model.layers.0._fsdp_wrapped_module.q_proj.lora_A.default", layer),
                ("base_model.model.layers.0._fsdp_wrapped_module.q_proj.lora_B.default", layer),
            ]

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
    manager._weights_dirty = True
    manager._staged_lora_params = None
    manager._staged_peft_config = None

    fake_device = _FakeDevice()
    fake_device.events = events
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
    monkeypatch.setattr(peft_save, "get_peft_model_state_dict", lambda model, state_dict=None: state_dict)

    manager.__enter__()

    assert events == [
        "load_actor",
        "offload_actor",
        "synchronize_offload",
        "wake_weights",
        "update_lora",
        "wake_kv_cache",
    ]

    events.clear()
    manager.__enter__()
    assert events == ["wake_weights", "wake_kv_cache"]

    events.clear()
    manager.layered_summon = False
    manager.module.world_size = 1
    monkeypatch.setattr(
        fsdp_vllm,
        "layered_summon_lora_params",
        lambda module: pytest.fail("single-rank staging must not summon FSDP params"),
    )
    manager.stage_updated_weights()
    assert set(manager._staged_lora_params) == {
        "base_model.model.layers.0.q_proj.lora_A.default.weight",
        "base_model.model.layers.0.q_proj.lora_B.default.weight",
    }
    manager.__enter__()
    assert events == ["wake_weights", "update_lora", "wake_kv_cache"]
