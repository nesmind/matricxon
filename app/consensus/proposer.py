from app.consensus.ballot import Ballot
from app.consensus.messages import NOOP, Command


class Campaign:
    """Phase 1 of one ballot: collects promises. Once a majority promised, `plan` says what the new
    leader must
    re-propose: per slot, the value accepted under the highest ballot (it may be chosen, so it must
    be kept), and a no-op for a gap.
    """

    def __init__(self, ballot: Ballot, quorum: int) -> None:
        self.ballot = ballot
        self._quorum = quorum
        self._promises: dict[str, tuple[tuple[int, Ballot, Command], ...]] = {}

    def add(self, sender: str, accepted: tuple[tuple[int, Ballot, Command], ...]) -> bool:
        """True exactly once: when this promise completes the majority."""
        before = len(self._promises) >= self._quorum
        self._promises[sender] = accepted
        return not before and len(self._promises) >= self._quorum

    def plan(self, from_slot: int, chosen: dict[int, Command]) -> dict[int, Command]:
        best: dict[int, tuple[Ballot, Command]] = {}
        for accepted in self._promises.values():
            for slot, ballot, command in accepted:
                if slot not in best or ballot > best[slot][0]:
                    best[slot] = (ballot, command)
        if not best:
            return {}
        last = max(best)
        return {
            slot: best[slot][1] if slot in best else NOOP
            for slot in range(from_slot, last + 1)
            if slot not in chosen
        }


class Leadership:
    """Phase 2 for a leader: the slots it has proposed and who accepted each."""

    def __init__(self, ballot: Ballot, quorum: int, first_free_slot: int) -> None:
        self.ballot = ballot
        self._quorum = quorum
        self.next_free_slot = first_free_slot
        self._inflight: dict[int, tuple[Command, set[str]]] = {}
        self.command_ids: set[str] = set()

    def propose(self, slot: int, command: Command) -> None:
        self._inflight[slot] = (command, set())
        self.next_free_slot = max(self.next_free_slot, slot + 1)
        self.command_ids.add(command.cmd_id)

    def next_slot_for(self, command: Command) -> int:
        slot = self.next_free_slot
        self.propose(slot, command)
        return slot

    def accepted(self, slot: int, sender: str) -> Command | None:
        """The command, once a majority accepted `slot` (only the first time)."""
        entry = self._inflight.get(slot)
        if entry is None:
            return None
        command, acks = entry
        acks.add(sender)
        if len(acks) < self._quorum:
            return None
        del self._inflight[slot]
        return command

    def unchosen(self) -> list[tuple[int, Command, set[str]]]:
        """Slots proposed but short of a majority: (slot, command, nodes that accepted it)."""
        return [
            (slot, command, set(acks)) for slot, (command, acks) in sorted(self._inflight.items())
        ]
