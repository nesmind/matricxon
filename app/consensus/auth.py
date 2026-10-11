import hashlib
import hmac


class MessageSigner:
    """HMAC-SHA256 over the exact bytes sent, with a key every cluster node shares: proves a message
    came from a
    cluster node and wasn't changed, but doesn't hide content (TLS later). Consensus messages are
    idempotent, so a replay is harmless.
    """

    def __init__(self, secret: bytes) -> None:
        if not secret:
            raise ValueError("the consensus secret must not be empty")
        self._key = hashlib.sha256(b"matricxon-consensus|" + secret).digest()

    def sign(self, body: bytes) -> str:
        return hmac.new(self._key, body, hashlib.sha256).hexdigest()

    def verify(self, body: bytes, signature: str) -> bool:
        return hmac.compare_digest(self.sign(body), signature or "")
