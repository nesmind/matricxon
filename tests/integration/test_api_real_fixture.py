"""M5: the read-mostly endpoints against a real (symlinked) local GGUF file,
not just the synthetic 0-byte files FakeInstalledModelFactory uses elsewhere.
Skips automatically on a machine without pAIring's local blob store - see
`real_ministral_model` in conftest.py.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel
from tests.integration.conftest import REAL_MINISTRAL_BLOB


def test_tags_lists_the_real_ministral_fixture(
    client: TestClient, real_ministral_model: InstalledModel
) -> None:
    response = client.get("/api/tags")

    assert response.status_code == 200
    [entry] = response.json()["models"]
    assert entry["name"] == "ministral-3:3b"
    assert entry["capabilities"] == ["completion"]
    assert entry["size"] == real_ministral_model.size_bytes
    assert entry["details"]["family"] == "mistral3"
    assert entry["details"]["context_length"] == 262144


def test_show_the_real_ministral_fixture(
    client: TestClient, real_ministral_model: InstalledModel
) -> None:
    response = client.post("/api/show", json={"model": "ministral-3:3b"})

    assert response.status_code == 200
    body = response.json()
    assert body["capabilities"] == ["completion"]
    assert body["details"]["parameter_size"] == "3.8B"


def test_ps_still_reports_nothing_loaded(
    client: TestClient, real_ministral_model: InstalledModel
) -> None:
    response = client.get("/api/ps")

    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_delete_removes_only_the_symlink_not_the_real_shared_blob(
    client: TestClient, real_ministral_model: InstalledModel
) -> None:
    gguf_symlink = Path(real_ministral_model.path)
    assert gguf_symlink.is_symlink()

    response = client.request("DELETE", "/api/delete", json={"model": "ministral-3:3b"})

    assert response.status_code == 200
    assert not gguf_symlink.exists()
    assert client.get("/api/tags").json() == {"models": []}
    assert REAL_MINISTRAL_BLOB.exists(), "the real, shared pAIring blob must survive delete"
