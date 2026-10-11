import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


class MatricxonError(Exception):
    """Base class for all matricxon domain errors."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnknownModelError(MatricxonError):
    status_code = 404


class ModelInUseError(MatricxonError):
    """A model can't be removed because something still depends on it (its aliases)."""

    status_code = 409


class NotLeaseHolderError(MatricxonError):
    """A cluster job (pulling models, cleaning the cache) that only the lease owner may do."""

    status_code = 409


class UnsupportedArchitectureError(MatricxonError):
    status_code = 400


class UnsupportedQuantTypeError(MatricxonError):
    status_code = 400


class InvalidGGUFError(MatricxonError):
    status_code = 400


class MissingMetadataError(MatricxonError):
    status_code = 400


class PromptTooLongError(MatricxonError):
    status_code = 400


class ModelResolutionError(MatricxonError):
    status_code = 404


class PullConnectionError(MatricxonError):
    status_code = 502


class PullCancelledError(MatricxonError):
    """Raised when a pull is aborted because its client disconnected mid-download
    (app.pull.download_coordinator says why an abandoned download running on invisibly is a real
    problem). Reaches PullJob.run's `except MatricxonError` like every pull failure, though nobody
    listens on an already-disconnected stream by then."""

    status_code = 499


class UnsupportedChatRoleError(MatricxonError):
    status_code = 400


class PushNotSupportedError(MatricxonError):
    status_code = 501


class InsufficientMemoryError(MatricxonError):
    status_code = 503


class DeviceUnavailableError(MatricxonError):
    """`MATRICXON_DEVICE` names a device this machine doesn't have, or the model can't run on it."""

    status_code = 503


class ErrorHandlerRegistrar:
    """Wires the MatricxonError hierarchy into FastAPI's exception handling."""

    def register(self, app: FastAPI) -> None:
        @app.exception_handler(MatricxonError)
        async def handle_matricxon_error(_: Request, exc: MatricxonError) -> JSONResponse:
            return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

        # Catches anything that isn't one of our typed errors (a bug: bad GGUF metadata shape, mmap
        # lifecycle issue, ...), which would reach Starlette's ServerErrorMiddleware and return a
        # bare non-JSON "Internal Server Error". pAIring's matricxon_client.py expects
        # `{"error":...}` and showed "Matricxon returned HTTP 500 with no error detail" while the
        # traceback sat in this server's log. Logs the traceback before responding, since this
        # handler intercepts the exception before Starlette's default logging.
        @app.exception_handler(Exception)
        async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
            logger.exception("Unhandled exception in request")
            return JSONResponse(status_code=500, content={"error": f"Internal error: {exc}"})
