import json
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from app.consensus.ballot import ZERO, Ballot
from app.consensus.messages import Command


@dataclass
class EpochState:
    """One Paxos instance (one epoch): what its acceptor promised and accepted, what is chosen."""

    promised: Ballot = ZERO
    accepted: dict[int, tuple[Ballot, Command]] = field(default_factory=dict)
    chosen: dict[int, Command] = field(default_factory=dict)


@dataclass
class DurableState:
    """What a node must not forget across a crash. `genesis`: the first voters (id -> address)."""

    genesis: dict[str, str] | None = None
    epochs: dict[int, EpochState] = field(default_factory=dict)
    snapshot: dict | None = None

    def epoch(self, number: int) -> EpochState:
        return self.epochs.setdefault(number, EpochState())

    def prune(self) -> None:
        """Forgets the log before the snapshot (in place: acceptors hold their epoch's state)."""
        if self.snapshot is None:
            return
        first, slot = self.snapshot["epoch"], self.snapshot["slot"]
        for number in [n for n in self.epochs if n < first]:
            del self.epochs[number]
        if first in self.epochs:
            here = self.epochs[first]
            here.accepted = {s: v for s, v in here.accepted.items() if s >= slot}
            here.chosen = {s: v for s, v in here.chosen.items() if s >= slot}


class Store(ABC):
    """Each save must be on stable storage before it returns: a reply sent before that could be
    forgotten by a crash, and a forgotten promise breaks Paxos."""

    @abstractmethod
    def load(self) -> DurableState: ...

    @abstractmethod
    def save_genesis(self, members: dict[str, str]) -> None: ...

    @abstractmethod
    def save_promise(self, epoch: int, ballot: Ballot) -> None: ...

    @abstractmethod
    def save_accept(self, epoch: int, slot: int, ballot: Ballot, command: Command) -> None: ...

    @abstractmethod
    def save_chosen(self, epoch: int, slot: int, command: Command) -> None: ...

    @abstractmethod
    def compact(self, state: DurableState) -> None:
        """Replaces everything stored with `state` (a snapshot and the log after it)."""


class MemoryStore(Store):
    """Survives a simulated crash (the node object is replaced, the store is kept)."""

    def __init__(self) -> None:
        self._state = DurableState()

    def load(self) -> DurableState:
        return deepcopy(self._state)

    def save_genesis(self, members: dict[str, str]) -> None:
        self._state.genesis = dict(members)

    def save_promise(self, epoch: int, ballot: Ballot) -> None:
        self._state.epoch(epoch).promised = ballot

    def save_accept(self, epoch: int, slot: int, ballot: Ballot, command: Command) -> None:
        self._state.epoch(epoch).accepted[slot] = (ballot, command)

    def save_chosen(self, epoch: int, slot: int, command: Command) -> None:
        self._state.epoch(epoch).chosen[slot] = command

    def compact(self, state: DurableState) -> None:
        self._state = deepcopy(state)


class FileStore(Store):
    """An append-only journal of JSON lines, synced on every write and replayed on load. A torn last
    line (a crash mid-write) is ignored: it was never acknowledged.

    The file is kept ahead of the data with zero padding (stripped on load), so a write overwrites
    existing blocks and `fdatasync` has no file size or block map to commit; on a busy disk a plain
    fsync of a growing file waits for the whole filesystem journal (seconds).
    """

    CHUNK = 1 << 20

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None
        self._end = 0
        self._size = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> DurableState:
        state = DurableState()
        if not self._path.exists():
            return state
        lines = self._path.read_bytes().rstrip(b"\0").decode().splitlines()
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                break
            kind = entry["k"]
            if kind == "g":
                state.genesis = entry["m"]
            elif kind == "s":
                state.snapshot = entry["v"]
            elif kind == "p":
                state.epoch(entry["e"]).promised = Ballot(*entry["b"])
            elif kind == "a":
                state.epoch(entry["e"]).accepted[entry["s"]] = (
                    Ballot(*entry["b"]),
                    Command(*entry["c"]),
                )
            elif kind == "c":
                state.epoch(entry["e"]).chosen[entry["s"]] = Command(*entry["c"])
        state.prune()
        if len(lines) > 2 * self._distinct(state) + self.COMPACT_SLACK:
            self._rewrite(state)
        return state

    COMPACT_SLACK = 1000

    @staticmethod
    def _distinct(state: DurableState) -> int:
        return 1 + sum(1 + len(e.accepted) + len(e.chosen) for e in state.epochs.values())

    def _rewrite(self, state: DurableState) -> None:
        """One line per fact (what load rebuilt): same state, atomic swap."""
        self._close()
        tmp = self._path.with_name(self._path.name + ".tmp")
        with tmp.open("w") as handle:
            if state.genesis is not None:
                handle.write(json.dumps({"k": "g", "m": state.genesis}) + "\n")
            if state.snapshot is not None:
                handle.write(json.dumps({"k": "s", "v": state.snapshot}) + "\n")
            for number, epoch in sorted(state.epochs.items()):
                handle.write(json.dumps({"k": "p", "e": number, "b": list(epoch.promised)}) + "\n")
                for slot, (ballot, command) in sorted(epoch.accepted.items()):
                    entry = {"k": "a", "e": number, "s": slot, "b": list(ballot)}
                    handle.write(
                        json.dumps({**entry, "c": [command.cmd_id, command.payload]}) + "\n"
                    )
                for slot, command in sorted(epoch.chosen.items()):
                    entry = {
                        "k": "c",
                        "e": number,
                        "s": slot,
                        "c": [command.cmd_id, command.payload],
                    }
                    handle.write(json.dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self._path)

    def compact(self, state: DurableState) -> None:
        self._rewrite(state)

    def save_genesis(self, members: dict[str, str]) -> None:
        self._append({"k": "g", "m": members})

    def save_promise(self, epoch: int, ballot: Ballot) -> None:
        self._append({"k": "p", "e": epoch, "b": list(ballot)})

    def save_accept(self, epoch: int, slot: int, ballot: Ballot, command: Command) -> None:
        self._append(
            {
                "k": "a",
                "e": epoch,
                "s": slot,
                "b": list(ballot),
                "c": [command.cmd_id, command.payload],
            }
        )

    def save_chosen(self, epoch: int, slot: int, command: Command) -> None:
        self._append({"k": "c", "e": epoch, "s": slot, "c": [command.cmd_id, command.payload]})

    def _append(self, entry: dict) -> None:
        line = (json.dumps(entry) + "\n").encode()
        fd = self._fd if self._fd is not None else self._open()
        if self._end + len(line) > self._size:
            padding = max(self.CHUNK, len(line))
            os.pwrite(fd, bytes(padding), self._size)
            self._size += padding
        os.pwrite(fd, line, self._end)
        self._end += len(line)
        os.fdatasync(fd)

    def _open(self) -> int:
        """Opens the journal at the end of its last whole line (a torn tail is cut off)."""
        self._fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o664)
        data = os.pread(self._fd, os.fstat(self._fd).st_size, 0).rstrip(b"\0")
        self._end = data.rfind(b"\n") + 1
        self._size = os.fstat(self._fd).st_size
        if self._end < len(data):
            os.pwrite(self._fd, bytes(len(data) - self._end), self._end)
        return self._fd

    def _close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
