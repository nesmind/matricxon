import threading
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends

from app.dependencies import get_model_catalog
from app.models.catalog import ModelCatalog
from app.pull.job import PullJob
from app.schemas.pull import PullRequest
from app.server.ndjson import NDJSONResponse
from app.server.thread_stream import stream_in_thread

router = APIRouter()


class PullRequestHandler:
    def __init__(self, catalog: ModelCatalog) -> None:
        self._catalog = catalog

    async def stream_response(self, request: PullRequest) -> AsyncIterator[dict]:
        """`cancel_event` is set in `finally` - covering every way this generator stops early,
        not just a client disconnect: normal completion (harmless, nothing left to cancel by
        then), an unhandled exception, and the disconnect case that motivated this (Starlette
        cancels this generator once it detects the client is gone - see PullJob's own docstring
        for the real bug that surfaced with no cancellation here at all: an abandoned pull's
        download just kept running on its own thread, invisible, until an unrelated retry
        raced it for the same file)."""
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


@router.post("/api/pull")
def post_pull(
    request: PullRequest, catalog: ModelCatalog = Depends(get_model_catalog)
) -> NDJSONResponse:
    return NDJSONResponse(PullRequestHandler(catalog).stream_response(request))
