from pathlib import Path

from agent_system.webshop_runtime import resolve_webshop_env_kwargs


def test_small_webshop_uses_external_1k_data_and_index(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBSHOP_DATA_ROOT", str(tmp_path))

    kwargs = resolve_webshop_env_kwargs(use_small=True, human_goals=False)

    assert kwargs == {
        "observation_mode": "text",
        "num_products": 1000,
        "human_goals": False,
        "file_path": str(tmp_path / "items_shuffle_1000.json"),
        "attr_path": str(tmp_path / "items_ins_v2_1000.json"),
    }


def test_full_webshop_uses_full_data_and_index(monkeypatch, tmp_path):
    monkeypatch.setenv("WEBSHOP_DATA_ROOT", str(tmp_path))

    kwargs = resolve_webshop_env_kwargs(use_small=False, human_goals=True)

    assert kwargs["num_products"] is None
    assert kwargs["human_goals"] is True
    assert Path(kwargs["file_path"]).name == "items_shuffle.json"
    assert Path(kwargs["attr_path"]).name == "items_ins_v2.json"
