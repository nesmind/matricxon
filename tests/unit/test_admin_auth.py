import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from app.server.admin_auth import (
    NONCE_HEADER,
    SIGNATURE_HEADER,
    TIME_HEADER,
    AdminAuth,
    require_admin,
)
from app.server.node_kind import require_compute


def _signed(secret, method, target, body=b"", stamp="1000", nonce="n1"):
    sig = AdminAuth.sign(secret, method, target, body, stamp, nonce)
    return {TIME_HEADER: stamp, NONCE_HEADER: nonce, SIGNATURE_HEADER: sig}


def _app(auth: AdminAuth) -> TestClient:
    from fastapi import Depends, Request

    app = FastAPI()

    async def guard(request: Request) -> None:
        await auth.check(request)

    @app.post("/x", dependencies=[Depends(guard)])
    async def x(request: Request) -> dict:
        return {"body": (await request.body()).decode()}

    return TestClient(app)


def test_open_without_a_secret():
    client = _app(AdminAuth(lambda: None))
    assert client.post("/x", content=b"hi").status_code == 200


def test_a_signed_request_passes_and_the_body_still_reaches_the_route():
    client = _app(AdminAuth(lambda: "s3", clock=lambda: 1000.0))
    r = client.post("/x", content=b"hi", headers=_signed("s3", "POST", "/x", b"hi"))
    assert r.status_code == 200 and r.json() == {"body": "hi"}


@pytest.mark.parametrize("case", ["unsigned", "wrong secret", "changed body", "expired"])
def test_bad_requests_are_refused(case):
    client = _app(AdminAuth(lambda: "s3", clock=lambda: 1000.0))
    headers = {}
    body = b"hi"
    if case == "wrong secret":
        headers = _signed("other", "POST", "/x", body)
    elif case == "changed body":
        headers = _signed("s3", "POST", "/x", b"original")
    elif case == "expired":
        headers = _signed("s3", "POST", "/x", body, stamp="500")
    assert client.post("/x", content=body, headers=headers).status_code == 401


def test_a_replayed_request_is_refused():
    client = _app(AdminAuth(lambda: "s3", clock=lambda: 1000.0))
    headers = _signed("s3", "POST", "/x", b"hi")
    assert client.post("/x", content=b"hi", headers=headers).status_code == 200
    assert client.post("/x", content=b"hi", headers=headers).status_code == 401


def test_voter_only_nodes_refuse_compute_routes(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "node_kind", "voter")
    with pytest.raises(HTTPException) as err:
        require_compute()
    assert err.value.status_code == 409
    monkeypatch.setattr(settings, "node_kind", "compute")
    require_compute()


def test_require_admin_is_open_by_default():
    assert asyncio.run(_noop()) is None


async def _noop():
    from starlette.requests import Request

    request = Request(
        {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    )
    return await require_admin(request)
