import threading

from fastapi.testclient import TestClient

from app.models.installed_model import InstalledModel
from app.models.manager import ModelManager


def test_chat_unload_call_returns_200_without_hanging(client: TestClient) -> None:
    response = client.post(
        "/api/chat",
        json={"model": "ministral-3:3b", "messages": [], "keep_alive": 0},
    )

    assert response.status_code == 200


def test_chat_skips_generation_when_stop_arrives_while_still_loading(
    client: TestClient,
    model_manager: ModelManager,
    tiny_mistral3_model: InstalledModel,
    monkeypatch,
) -> None:
    """Regression guard for the real bug, reproduced through the actual HTTP layer (not just the
    ModelManager unit tests): a stop request arriving while /api/chat's own get_or_load() call is
    still stuck loading used to be silently lost, since unload() had no handle yet to signal - see
    ModelManager.pop_cancelled_load's own docstring. _load is blocked with a threading.Event
    handshake so this proves the actual race, not a convenient but unrealistic call ordering."""
    load_started = threading.Event()
    proceed = threading.Event()
    real_load = ModelManager._load

    def blocking_load(self, tag, keep_alive_seconds, now):
        load_started.set()
        proceed.wait()
        return real_load(self, tag, keep_alive_seconds, now)

    monkeypatch.setattr(ModelManager, "_load", blocking_load)

    result: dict = {}

    def do_request() -> None:
        result["response"] = client.post(
            "/api/chat",
            json={
                "model": tiny_mistral3_model.tag,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    thread = threading.Thread(target=do_request)
    thread.start()
    try:
        assert load_started.wait(timeout=5)
        assert model_manager.list_loaded() == []  # still loading - no handle exists yet
        model_manager.unload(tiny_mistral3_model.tag)  # the real stop request, arriving mid-load
    finally:
        proceed.set()
        thread.join(timeout=10)

    assert result["response"].status_code == 200
    assert result["response"].json() == {}
    # The model still ended up loaded and cached (the real work wasn't wasted) - just this one
    # request's own generation was skipped.
    assert model_manager.list_loaded()[0].tag == tiny_mistral3_model.tag
