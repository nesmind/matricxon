import logging

import pytest

from app.config import Settings
from app.consensus.factory import ConsensusFactory
from app.server.log_filters import ConsensusTrafficFilter, quiet_cluster_logging


def _record(path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        "",
        0,
        '%s - "%s %s HTTP/%s" %d',
        ("10.0.0.1:1234", "POST", path, "1.1", status),
        None,
    )


def test_delivered_consensus_messages_are_not_logged_but_refused_ones_and_other_requests_are():
    keep = ConsensusTrafficFilter().filter
    assert not keep(_record("/api/consensus/message", 204))
    assert not keep(_record("/api/consensus/message", 503))
    assert keep(_record("/api/consensus/message", 401))
    assert keep(_record("/api/consensus/message", 400))
    assert keep(_record("/api/chat", 204))
    assert keep(_record("/api/consensus/status", 200))


def test_the_filter_is_installed_once_and_traffic_logging_can_be_turned_back_on():
    access = logging.getLogger("uvicorn.access")
    before = list(access.filters)
    try:
        quiet_cluster_logging()
        quiet_cluster_logging()
        assert sum(isinstance(f, ConsensusTrafficFilter) for f in access.filters) == 1
        for f in [f for f in access.filters if isinstance(f, ConsensusTrafficFilter)]:
            access.removeFilter(f)
        quiet_cluster_logging(include_traffic=True)
        assert not any(isinstance(f, ConsensusTrafficFilter) for f in access.filters)
    finally:
        for f in list(access.filters):
            access.removeFilter(f)
        for f in before:
            access.addFilter(f)


def test_the_http_clients_debug_output_is_silenced_while_the_server_traces_at_debug():
    logging.getLogger("httpcore").setLevel(logging.DEBUG)
    quiet_cluster_logging()
    assert logging.getLogger("httpcore").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING


def test_a_real_node_beats_slower_than_the_simulation(tmp_path):
    settings = Settings(
        _env_file=None,
        cache_key="k",
        cluster_role="seed",
        advertise_url="http://me:8420",
        node_state_dir=tmp_path,
    )
    config = ConsensusFactory(settings, "me", tmp_path / "j", tmp_path / "key")._paxos_config()
    assert (config.heartbeat, config.election_timeout, config.retry_interval) == (0.2, 1.0, 0.5)
    custom = Settings(
        _env_file=None,
        consensus_heartbeat_seconds=0.5,
        consensus_election_seconds=3.0,
    )
    config = ConsensusFactory(custom, "me", tmp_path / "j", tmp_path / "key")._paxos_config()
    assert (config.heartbeat, config.election_timeout) == (0.5, 3.0)


def test_a_sender_that_hangs_up_is_not_an_error():
    import asyncio

    from starlette.requests import ClientDisconnect

    from app.routers.consensus_router import receive_message

    class Gone:
        headers: dict = {}

        async def body(self) -> bytes:
            raise ClientDisconnect()

    response = asyncio.run(receive_message(Gone(), object()))  # type: ignore[arg-type]
    assert response.status_code == 204


def test_messages_to_a_node_with_consensus_off_get_one_warning_a_minute(caplog):
    import asyncio
    from types import SimpleNamespace

    from fastapi import HTTPException

    from app.routers import consensus_router
    from app.routers.consensus_router import receive_message

    consensus_router._last_off_warning[0] = -1e9
    request = SimpleNamespace(client=SimpleNamespace(host="172.20.10.14"), headers={})

    async def send() -> None:
        with pytest.raises(HTTPException) as refused:
            consensus_router._runtime(request, None)
        assert refused.value.status_code == 503

    caplog.set_level(logging.WARNING, logger="matricxon.cluster")
    for _ in range(5):
        asyncio.run(send())
    warnings = [r.getMessage() for r in caplog.records if r.name == "matricxon.cluster"]
    assert len(warnings) == 1 and "172.20.10.14 keeps sending cluster messages" in warnings[0]
    assert receive_message
