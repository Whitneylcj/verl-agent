import pytest

from recipe.exact.check_appworld_services import load_port_manifest, require_reachable_services


def test_port_manifest_rejects_duplicates_and_malformed_values(tmp_path):
    duplicate = tmp_path / "duplicate.ports"
    duplicate.write_text("8200\n8200\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicates"):
        load_port_manifest(duplicate)

    malformed = tmp_path / "malformed.ports"
    malformed.write_text("not-a-port\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid AppWorld port"):
        load_port_manifest(malformed)


def test_service_preflight_checks_capacity_and_reachability(tmp_path, monkeypatch):
    connected = []

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_create_connection(address, timeout):
        connected.append((address, timeout))
        return _Connection()

    monkeypatch.setattr("recipe.exact.check_appworld_services.socket.create_connection", fake_create_connection)
    port = 8200
    manifest = tmp_path / "appworld.ports"
    manifest.write_text(f"{port}\n", encoding="utf-8")
    assert require_reachable_services(manifest, required=1, timeout=0.1) == [port]
    assert connected == [(("127.0.0.1", 8200), 0.1)]
    with pytest.raises(RuntimeError, match="needs 2 services"):
        require_reachable_services(manifest, required=2, timeout=0.1)
