import logging
import time

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.requests import ClientDisconnect

from app.consensus.codec import CodecError
from app.consensus.runtime import (
    ADDRESS_HEADER,
    ADDRESS_SIGNATURE_HEADER,
    SIGNATURE_HEADER,
    BadSignatureError,
    ConsensusRuntime,
)
from app.dependencies import get_consensus_runtime
from app.schemas.consensus import (
    ConfigChange,
    ConfigResult,
    ConsensusStatus,
    ReachRequest,
    ReachResult,
    VoterChange,
    VoterResult,
)
from app.server.admin_auth import require_admin

router = APIRouter()


_OFF_WARNING_EVERY = 60.0
_last_off_warning = [-_OFF_WARNING_EVERY]


def _runtime(
    request: Request, runtime: ConsensusRuntime | None = Depends(get_consensus_runtime)
) -> ConsensusRuntime:
    if runtime is None:
        _warn_cluster_messages_while_off(request)
        raise HTTPException(status_code=503, detail="consensus is not enabled on this node")
    return runtime


def _warn_cluster_messages_while_off(request: Request) -> None:
    """Another node still sends cluster messages although cluster mode is off here (just turned off,
    or reset): warned once a minute, not per message."""
    now = time.monotonic()
    if now - _last_off_warning[0] >= _OFF_WARNING_EVERY:
        _last_off_warning[0] = now
        sender = request.client.host if request.client else "a node"
        logging.getLogger("matricxon.cluster").warning(
            "cluster: %s keeps sending cluster messages, but cluster mode is off on this node; "
            "ignoring them until it is turned on or that node stops",
            sender,
        )


@router.post("/api/consensus/message", status_code=204)
async def receive_message(
    request: Request, runtime: ConsensusRuntime = Depends(_runtime)
) -> Response:
    """A signed consensus message from another node of the cluster."""
    try:
        body = await request.body()
    except ClientDisconnect:  # the sender timed out (this node was busy): Paxos tolerates the loss
        return Response(status_code=204)
    try:
        # Writes the journal (fsync): off the event loop, or a slow disk stalls the server.
        headers = request.headers
        await run_in_threadpool(
            runtime.handle,
            body,
            headers.get(SIGNATURE_HEADER, ""),
            headers.get(ADDRESS_HEADER, ""),
            headers.get(ADDRESS_SIGNATURE_HEADER, ""),
        )
    except BadSignatureError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    except (CodecError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return Response(status_code=204)


@router.get("/api/consensus/status", response_model=ConsensusStatus)
def get_status(
    runtime: ConsensusRuntime | None = Depends(get_consensus_runtime),
) -> ConsensusStatus:
    if runtime is None:
        return ConsensusStatus(enabled=False)
    return ConsensusStatus(enabled=True, **runtime.status())


@router.put(
    "/api/consensus/config", dependencies=[Depends(require_admin)], response_model=ConfigResult
)
def set_config(change: ConfigChange, runtime: ConsensusRuntime = Depends(_runtime)) -> ConfigResult:
    """Sets one cluster-wide config value through the cluster (needs a majority). Signed when an
    admin secret is set;
    pAIring's admin pages sit in front."""
    _, applied, result = runtime.submit({"op": "set", "key": change.key, "value": change.value})
    return ConfigResult(applied=applied, result=result)


@router.post(
    "/api/consensus/voters", dependencies=[Depends(require_admin)], response_model=VoterResult
)
def add_voter(change: VoterChange, runtime: ConsensusRuntime = Depends(_runtime)) -> VoterResult:
    """Makes a node a voter (a new epoch). Any voter can be asked; it goes through the leader."""
    return VoterResult(ok=runtime.add_voter(change.id, change.address.rstrip("/")))


@router.delete(
    "/api/consensus/voters/{node_id}",
    dependencies=[Depends(require_admin)],
    response_model=VoterResult,
)
def remove_voter(
    node_id: str, leave: bool = False, runtime: ConsensusRuntime = Depends(_runtime)
) -> VoterResult:
    """`leave=true`: removed on purpose, so it is not promoted again while it still follows."""
    return VoterResult(ok=runtime.remove_voter(node_id, leave=leave))


@router.post("/api/consensus/reach", response_model=ReachResult)
def can_reach(change: ReachRequest) -> ReachResult:
    """Whether THIS node can reach `address`: a new node must be reachable at its advertised address
    (pAIring's 127.0.0.1 can mean something else in a container)."""
    address = change.address.rstrip("/")
    if not address.startswith(("http://", "https://")):
        return ReachResult(reachable=False, detail="not an http(s) address")
    try:
        resp = httpx.get(
            address + "/api/cluster/readiness", timeout=httpx.Timeout(4.0, connect=2.0)
        )
    except httpx.HTTPError as error:
        return ReachResult(reachable=False, detail=str(error) or type(error).__name__)
    if resp.status_code != 200:
        return ReachResult(reachable=False, detail=f"HTTP {resp.status_code}")
    return ReachResult(reachable=True, node_id=str(resp.json().get("node_id") or ""))
