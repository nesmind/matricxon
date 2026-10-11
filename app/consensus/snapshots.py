"""The node's side of snapshots: compacting its log, handing a snapshot to a node asking for a gone
log, and jumping to one (replica.py)."""

from app.consensus.messages import CatchUp, Snapshot
from app.consensus.roles import LeaderRole


class SnapshotRole(LeaderRole):
    SEND_EVERY = 5.0

    def _raise_floor(self) -> None:
        base = self.replica.base
        self.acceptor.floor = base["slot"] if base and base["epoch"] == self.epoch else 0

    def _compact(self) -> None:
        """The base moved: forget the log before it, here and on disk."""
        state, base = self._durable, self.replica.base
        state.snapshot = base
        for number, slots in self.replica.chosen.items():
            state.epoch(number).chosen = dict(slots)
        state.prune()
        self._store.compact(state)
        self._raise_floor()

    def _on_snapshot(self, msg: Snapshot, now: float) -> None:
        before = self.epoch
        snapshot = msg.state
        if (msg.at_epoch, msg.at_slot) != (snapshot["epoch"], snapshot["slot"]):
            return
        if not self.replica.install(snapshot):
            return
        self._compact()
        if self.epoch != before:
            self._enter_epoch()
            self._reset_election(now)
        self._route_pending()
        self._ask_for_log(msg.sender, now, force=True)

    def _serve_snapshot(self, msg: CatchUp, now: float) -> bool:
        """True when `msg` asks for log before the base: it gets the snapshot (not too often)."""
        if not self.replica.before_base(msg.epoch, msg.from_slot):
            return False
        if now - self._snapshot_sent.get(msg.sender, -1e9) >= self.SEND_EVERY:
            self._snapshot_sent[msg.sender] = now
            base = self.replica.base
            self._send(msg.sender, Snapshot(self.id, base["epoch"], base["slot"], base))
        return True
