import threading
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends

from app.consensus.runtime import ConsensusRuntime
from app.dependencies import get_consensus_runtime, get_model_catalog
from app.models.catalog import ModelCatalog
from app.pull.job import PullJob
from app.schemas.pull import PullRequest
from app.server.admin_auth import require_admin
from app.server.errors import NotLeaseHolderError
from app.server.ndjson import NDJSONResponse
from app.server.node_kind import require_compute
from app.server.thread_stream import stream_in_thread

router = APIRouter()


class PullRequestHandler:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    async def stream_response(self, request: PullRequest) -> AsyncIterator[dict]:
        """`cancel_event` is set in `finally`, covering every way this generator stops early: normal
        completion (harmless), an unhandled exception, and the disconnect that motivated it
        (Starlette cancels this generator once the client is gone; see PullJob for the bug when
        nothing cancelled an abandoned pull: its download kept running on its own thread, unseen,
        until an unrelated retry raced it for the file)."""
        cancel_event = threading.Event()
        job = PullJob(self._catalog.models_dir, cancel_event=cancel_event)
        results = stream_in_thread(lambda: job.run(request.model))
        try:
            while True:
                item = await results.get()
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            cancel_event.set()


def require_pull_lease(runtime: ConsensusRuntime | None) -> None:
    """In a cluster only the node owning the pull lease downloads models (the folder is shared, so
    two nodes pulling would write the same files). Alone, any pull is fine."""
    if runtime is None or runtime.keeper is None or runtime.keeper.holds("model_pull"):
        return
    holder = runtime.leases.holder("model_pull")
    where = (
        f"node {holder} owns the model-pull lease" if holder else "no node owns the pull lease yet"
    )
    raise NotLeaseHolderError(f"this node cannot change models: {where}")


@router.post("/api/pull", dependencies=[Depends(require_admin), Depends(require_compute)])
def post_pull(
    request: PullRequest,
    catalog: ModelCatalog = Depends(get_model_catalog),
    runtime: ConsensusRuntime | None = Depends(get_consensus_runtime),
) -> NDJSONResponse:
    require_pull_lease(runtime)
    return NDJSONResponse(PullRequestHandler(catalog).stream_response(request))
