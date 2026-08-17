import json

from recipe.exact.prepare_model import discover_weight_files


def test_discover_single_safetensors_file(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights")
    assert discover_weight_files(tmp_path) == [weight]


def test_discover_indexed_safetensors_requires_every_shard(tmp_path):
    first = tmp_path / "model-00001-of-00002.safetensors"
    second = tmp_path / "model-00002-of-00002.safetensors"
    first.write_bytes(b"first")
    index = {
        "weight_map": {
            "layer.0": first.name,
            "layer.1": second.name,
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    try:
        discover_weight_files(tmp_path)
    except FileNotFoundError as error:
        assert second.name in str(error)
    else:
        raise AssertionError("an incomplete sharded model was accepted")

    second.write_bytes(b"second")
    assert discover_weight_files(tmp_path) == [first, second]
