import hashlib
import hmac
import threading
import time
from collections.abc import Callable

from fastapi import HTTPException, Request

from app.config import settings

TIME_HEADER = "X-Matricxon-Time"
NONCE_HEADER = "X-Matricxon-Nonce"
SIGNATURE_HEADER = "X-Matricxon-Signature"


class AdminAuth:
    """Checks that an admin request was signed with the shared admin secret.

    The signature covers method, path+query, body, a timestamp and a one-time nonce, so a captured
    request can't be changed or replayed (outside the time window, or twice inside it). No secret
    configured = open (protect the network instead). The content is not hidden (TLS would).
    """

    WINDOW_SECONDS = 120

    def __init__(
        self, secret_of: Callable[[], str | None], clock: Callable[[], float] = time.time
    ) -> None:
        self._secret_of = secret_of
        self._clock = clock
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def sign(secret: str, method: str, target: str, body: bytes, stamp: str, nonce: str) -> str:
        key = hashlib.sha256(b"matricxon-admin|" + secret.encode()).digest()
        text = f"{stamp}\n{nonce}\n{method.upper()}\n{target}\n{hashlib.sha256(body).hexdigest()}"
        return hmac.new(key, text.encode(), hashlib.sha256).hexdigest()

    @property
    def enabled(self) -> bool:
        return bool(self._secret_of())

    async def check(self, request: Request) -> None:
        secret = self._secret_of()
        if not secret:
            return
        stamp = request.headers.get(TIME_HEADER, "")
        nonce = request.headers.get(NONCE_HEADER, "")
        given = request.headers.get(SIGNATURE_HEADER, "")
        target = request.url.path + (f"?{request.url.query}" if request.url.query else "")
        expected = self.sign(secret, request.method, target, await request.body(), stamp, nonce)
        if not given or not hmac.compare_digest(expected, given):
            raise HTTPException(
                status_code=401, detail="admin request not signed with the admin secret"
            )
        try:
            fresh = abs(self._clock() - float(stamp)) <= self.WINDOW_SECONDS
        except ValueError:
            fresh = False
        if not fresh or not self._first_use(nonce):
            raise HTTPException(status_code=401, detail="admin request expired or replayed")

    def _first_use(self, nonce: str) -> bool:
        now = self._clock()
        with self._lock:
            self._seen = {n: t for n, t in self._seen.items() if now - t <= self.WINDOW_SECONDS}
            if nonce in self._seen:
                return False
            self._seen[nonce] = now
            return True


_auth = AdminAuth(lambda: settings.admin_secret)


def get_admin_auth() -> AdminAuth:
    return _auth


async def require_admin(request: Request) -> None:
    """FastAPI dependency for routes that change the cluster, models or caches."""
    await _auth.check(request)
