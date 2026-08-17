import hashlib
import json

import pytest

from recipe.exact.prepare_model import (
    _expected_hashes,
    discover_weight_files,
    validate_weight_hashes,
)


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


def test_validate_weight_hashes_records_and_checks_sha256(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights")
    expected = hashlib.sha256(b"weights").hexdigest()

    assert validate_weight_hashes([weight], {weight.name: expected}) == [{"name": weight.name, "bytes": 7, "sha256": expected}]
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_weight_hashes([weight], {weight.name: "0" * 64})


def test_expected_hashes_require_complete_name_sha_pairs():
    assert _expected_hashes([f"model.safetensors={'a' * 64}"]) == {"model.safetensors": "a" * 64}
    with pytest.raises(ValueError, match="NAME=SHA256"):
        _expected_hashes(["missing-separator"])
