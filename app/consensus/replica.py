from collections import OrderedDict
from typing import Any

from app.consensus.messages import NOOP, Command
from app.consensus.state_machine import StateMachine
from app.consensus.storage import DurableState, Store

RECONFIGURE = "reconfigure"


class SafetyViolation(Exception):
    """Two different commands were chosen for one slot: Paxos was broken. Never expected."""


def reconfigure_payload(epoch: int, members: dict[str, str]) -> dict[str, Any]:
    """The command that moves the cluster from `epoch` to a new voter set (id -> address)."""
    return {"op": RECONFIGURE, "epoch": epoch, "members": dict(members)}


def is_reconfigure(command: Command) -> bool:
    return isinstance(command.payload, dict) and command.payload.get("op") == RECONFIGURE


class Replica:
    """The chosen log and the state machine it drives, one log per epoch.

    An epoch is one fixed set of voters running one Paxos instance. A `reconfigure` chosen in epoch
    e seals it: commands up to and including it count, anything chosen after it in that epoch is
    void, and epoch e+1 starts empty with the new voters. Fixed membership per instance keeps voter
    changes safe. All epochs' commands, in order, are what a new node replays to catch up.

    Within an epoch a slot is applied once every slot below it is known. An already-applied command
    id is skipped, so a client retry or a command proposed twice changes nothing.
    """

    MAX_REMEMBERED_RESULTS = 10_000

    def __init__(
        self, machine: StateMachine, store: Store, state: DurableState, snapshot_every: int = 5000
    ) -> None:
        self.machine = machine
        self._store = store
        self._every = snapshot_every
        self.epoch = 0
        self.next_slot = 0
        self.members: dict[str, str] = dict(state.genesis or {})
        # Every address any node has had in any epoch. A removed or demoted node leaves `members`
        # yet must stay reachable
        # (to tell it it is behind, to answer its catch-up).
        self.addresses: dict[str, str] = dict(self.members)
        self.chosen: dict[int, dict[int, Command]] = {}
        self._stop_slot: dict[int, int] = {}
        self._results: OrderedDict[str, Any] = OrderedDict()
        # The oldest snapshot still needed (the log before it is gone). A newer one waits in
        # `_pending` and becomes the
        # base at the next interval: the log keeps one to two intervals.
        self.base: dict | None = state.snapshot
        self._pending: dict | None = None
        if self.base is not None:
            self._restore(self.base)
        for epoch in sorted(state.epochs):
            for slot, command in sorted(state.epochs[epoch].chosen.items()):
                self.chosen.setdefault(epoch, {})[slot] = command
        self._advance()
        self._pending_total = self.applied_total()

    def address_of(self, node_id: str, configured: dict[str, str]) -> str | None:
        """Where to reach a node: a voter's address, else one configured for it, else the last
        address it ever had in the cluster."""
        return self.members.get(node_id) or configured.get(node_id) or self.addresses.get(node_id)

    def commit(self, epoch: int, slot: int, command: Command) -> bool:
        """Records epoch/slot -> command as chosen. False if it was already known."""
        known = self.chosen.get(epoch, {}).get(slot)
        if known is None and (epoch, slot) < (self.epoch, self.next_slot):
            return False
        if known is not None:
            if known != command:
                raise SafetyViolation(
                    f"epoch {epoch} slot {slot}: {known} is chosen, now {command}"
                )
            return False
        self._store.save_chosen(epoch, slot, command)
        self.chosen.setdefault(epoch, {})[slot] = command
        self._advance()
        return True

    def _advance(self) -> None:
        while (command := self.chosen.get(self.epoch, {}).get(self.next_slot)) is not None:
            if command.cmd_id in self._results or command == NOOP:
                self.next_slot += 1
            elif is_reconfigure(command):
                self._reconfigure(command)
            else:
                self._remember(command, self.machine.apply(command.payload))
                self.next_slot += 1

    def applied_total(self) -> int:
        """Slots applied over every epoch so far: how far this node is through the whole log."""
        return sum(self._stop_slot.get(e, 0) for e in range(self.epoch)) + self.next_slot

    def _reconfigure(self, command: Command) -> None:
        if command.payload["epoch"] != self.epoch:
            self._remember(command, False)
            self.next_slot += 1
            return
        self._remember(command, True)
        self._stop_slot[self.epoch] = self.next_slot
        self.members = dict(command.payload["members"])
        self.addresses.update(self.members)
        self.epoch, self.next_slot = self.epoch + 1, 0

    def _remember(self, command: Command, result: Any) -> None:
        self._results[command.cmd_id] = result
        if len(self._results) > self.MAX_REMEMBERED_RESULTS:
            self._results.popitem(last=False)

    def applied(self, cmd_id: str) -> bool:
        return cmd_id in self._results

    def result(self, cmd_id: str) -> Any:
        return self._results[cmd_id]

    def chosen_from(self, epoch: int, slot: int, limit: int = 100) -> list[tuple[int, Command]]:
        """The chosen commands of `epoch` from `slot` on, up to the one that sealed it."""
        last = self._stop_slot.get(epoch)
        entries = self.chosen.get(epoch, {})
        wanted = [s for s in sorted(entries) if s >= slot and (last is None or s <= last)]
        return [(s, entries[s]) for s in wanted[:limit]]

    # snapshots: the log before the base is gone

    def before_base(self, epoch: int, slot: int) -> bool:
        """True when the log from (`epoch`, `slot`) on is no longer here: only the snapshot is."""
        return self.base is not None and (epoch, slot) < (self.base["epoch"], self.base["slot"])

    def rotate(self) -> bool:
        """Every `snapshot_every` applied slots takes a snapshot; the previous one becomes the base
        and the log below it is dropped. True when the base moved (the journal is rewritten)."""
        total = self.applied_total()
        if total - self._pending_total < self._every:
            return False
        moved = self._pending is not None
        if moved:
            self.base = self._pending
            self._drop_below(self.base)
        self._pending, self._pending_total = self._capture(), total
        return moved

    def install(self, snapshot: dict) -> bool:
        """Jumps to a snapshot from a peer, if it is ahead of what is applied here."""
        if (snapshot["epoch"], snapshot["slot"]) <= (self.epoch, self.next_slot):
            return False
        self._restore(snapshot)
        self.base, self._pending = snapshot, None
        self._drop_below(snapshot)
        self._advance()
        self._pending_total = self.applied_total()
        return True

    def _capture(self) -> dict:
        return {
            "epoch": self.epoch,
            "slot": self.next_slot,
            "machine": self.machine.snapshot(),
            "results": [[k, v] for k, v in self._results.items()],
            "members": dict(self.members),
            "addresses": dict(self.addresses),
            "stops": {str(e): s for e, s in self._stop_slot.items()},
        }

    def _restore(self, snap: dict) -> None:
        self.machine.restore(snap["machine"])
        self._results = OrderedDict((k, v) for k, v in snap["results"])
        self.epoch, self.next_slot = snap["epoch"], snap["slot"]
        self.members = dict(snap["members"])
        self.addresses.update(snap["addresses"])
        self._stop_slot = {int(e): s for e, s in snap["stops"].items()}

    def _drop_below(self, snap: dict) -> None:
        for number in [e for e in self.chosen if e < snap["epoch"]]:
            del self.chosen[number]
        here = self.chosen.get(snap["epoch"], {})
        for slot in [s for s in here if s < snap["slot"]]:
            del here[slot]
