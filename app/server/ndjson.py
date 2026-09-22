import json
from collections.abc import AsyncIterator

from starlette.responses import StreamingResponse


class NDJSONResponse(StreamingResponse):
    """Streams an async iterator of dicts as newline-delimited JSON.

    No compression middleware may sit in front of routes using this response —
    that would buffer the whole body and destroy token-by-token latency.
    """

    media_type = "application/x-ndjson"

    def __init__(self, body_iterator: AsyncIterator[dict], **kwargs: object) -> None:
        super().__init__(content=self._encode(body_iterator), **kwargs)

    @staticmethod
    async def _encode(body_iterator: AsyncIterator[dict]) -> AsyncIterator[bytes]:
        async for item in body_iterator:
            yield (json.dumps(item) + "\n").encode("utf-8")
