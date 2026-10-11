import random

from app.consensus.messages import CatchUp
from app.consensus.replica import Replica


class CatchUpPolicy:
    """When and whom a node asks for the commands it is missing. A node learns it is behind from a
    newer-epoch
    message, a heartbeat showing a longer log, or by being new; it asks, at most once per interval,
    the node that
    told it or (when idle) a random peer.
    """

    # While behind, the next batch is asked for once the last was taken in, and again after this
    # long without progress.
    # Asking sooner (on a timer, or per arriving slot) floods the sender's queue (it drops the
    # oldest, exactly the
    # missing slots): a stall, or a crawl of a few slots a second.
    PULL_RETRY_SECONDS = 1.0

    def __init__(
        self, node_id: str, replica: Replica, rng: random.Random, interval: float, batch: int = 100
    ) -> None:
        self._batch = batch
        self._seen: tuple[int, int] | None = None
        self._moved_at = -1e9
        self._id = node_id
        self._replica = replica
        self._rng = rng
        self._interval = interval
        self._last = -1e9
        self._ahead: tuple[str, int, int] | None = None
        self._asked_at: tuple[int, int] | None = None

    def retune(self, interval: float) -> None:
        self._interval = interval

    def note_ahead(self, peer: str, epoch: int, next_slot: int = 0) -> None:
        """`peer` has a longer log (a newer epoch, or more of this one): keep pulling from it until
        caught up, instead of
        waiting for another mention. Without this, a voter that missed epochs thought it was still a
        voter, never
        pulled, and caught up one batch per election.
        """
        if self._ahead is None or (epoch, next_slot) >= self._ahead[1:]:
            self._ahead = (peer, epoch, next_slot)

    def pull(self, now: float) -> tuple[str, CatchUp] | None:
        """(peer, request) while behind a peer known to be ahead, else None."""
        if self._ahead is None:
            return None
        peer, epoch, next_slot = self._ahead
        if (self._replica.epoch, self._replica.next_slot) >= (epoch, next_slot):
            self._ahead = None
            return None
        here = (self._replica.epoch, self._replica.next_slot)
        if here != self._seen:
            self._seen, self._moved_at = here, now
        if self._asked_at is not None:
            asked = self._asked_at
            taken_in = here[0] > asked[0] or here[1] >= asked[1] + self._batch
            if not taken_in and now - self._moved_at < self.PULL_RETRY_SECONDS:
                return None
        self._asked_at = here
        return peer, self.request(peer, now, force=True)

    def request(self, peer: str, now: float, force: bool = False) -> CatchUp | None:
        """The request to send to `peer`, or None if one went out very recently (unless `force`)."""
        if peer == self._id or (not force and now - self._last < self._interval):
            return None
        self._last = now
        return CatchUp(self._id, self._replica.epoch, self._replica.next_slot)

    def random_peer(self, members: list[str], seeds: list[str]) -> str | None:
        peers = [p for p in {*members, *seeds} if p != self._id]
        return self._rng.choice(sorted(peers)) if peers else None
