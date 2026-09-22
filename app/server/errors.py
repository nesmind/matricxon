from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class MatricxonError(Exception):
    """Base class for all matricxon domain errors."""

    status_code: int = 500

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnknownModelError(MatricxonError):
    status_code = 404


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
    """Raised when a pull is aborted because the client that requested it disconnected
    mid-download (see app.pull.download_coordinator's own docstring on why letting an abandoned
    download keep running to completion invisibly is itself a real problem, not just a
    theoretical one). Reaches PullJob.run's existing `except MatricxonError` handler like every
    other pull failure, though in practice nobody is listening on an already-disconnected
    stream by the time this fires."""

    status_code = 499


class UnsupportedChatRoleError(MatricxonError):
    status_code = 400


class PushNotSupportedError(MatricxonError):
    status_code = 501


class InsufficientMemoryError(MatricxonError):
    status_code = 503


class ErrorHandlerRegistrar:
    """Wires the MatricxonError hierarchy into FastAPI's exception handling."""

    def register(self, app: FastAPI) -> None:
        @app.exception_handler(MatricxonError)
        async def handle_matricxon_error(_: Request, exc: MatricxonError) -> JSONResponse:
            return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
