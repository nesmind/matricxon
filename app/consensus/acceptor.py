from app.consensus.ballot import Ballot
from app.consensus.messages import Accept, Accepted, Nack, Prepare, Promise
from app.consensus.storage import EpochState, Store


class Acceptor:
    """The Paxos acceptor of one epoch: promises and accepts, saving both to stable storage
    before it replies."""

    def __init__(self, node_id: str, epoch: int, state: EpochState, store: Store) -> None:
        self._id = node_id
        self._epoch = epoch
        self._state = state
        self._store = store
        self.floor = 0  # slots below were applied and compacted away: nothing new may be accepted

    @property
    def promised(self) -> Ballot:
        return self._state.promised

    @property
    def max_round(self) -> int:
        rounds = [self._state.promised.round] + [b.round for b, _ in self._state.accepted.values()]
        return max(rounds)

    def prepare(self, msg: Prepare) -> Promise | Nack:
        """Phase 1b. A repeated Prepare of the same ballot gets the same answer."""
        if msg.ballot < self._state.promised:
            return Nack(self._id, self._epoch, msg.ballot, self._state.promised)
        self._promise(msg.ballot)
        entries = tuple(
            (slot, ballot, command)
            for slot, (ballot, command) in sorted(self._state.accepted.items())
            if slot >= msg.from_slot
        )
        return Promise(self._id, self._epoch, msg.ballot, entries)

    def accept(self, msg: Accept) -> Accepted | Nack:
        """Phase 2b."""
        if msg.slot < self.floor:
            return Nack(self._id, self._epoch, msg.ballot, self._state.promised)
        if msg.ballot < self._state.promised:
            return Nack(self._id, self._epoch, msg.ballot, self._state.promised)
        self._promise(msg.ballot)
        if self._state.accepted.get(msg.slot) == (msg.ballot, msg.command):
            return Accepted(self._id, self._epoch, msg.ballot, msg.slot)
        self._state.accepted[msg.slot] = (msg.ballot, msg.command)
        self._store.save_accept(self._epoch, msg.slot, msg.ballot, msg.command)
        return Accepted(self._id, self._epoch, msg.ballot, msg.slot)

    def _promise(self, ballot: Ballot) -> None:
        if ballot > self._state.promised:
            self._state.promised = ballot
            self._store.save_promise(self._epoch, ballot)
