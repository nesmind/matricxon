from dataclasses import dataclass
from typing import Any

from app.consensus.ballot import Ballot


@dataclass(frozen=True)
class Command:
    """What the cluster agrees on. `cmd_id` makes a retry harmless: a command id is applied once."""

    cmd_id: str
    payload: Any


NOOP = Command("", None)

# Every message of one Paxos instance carries the epoch it was sent in. An epoch is one fixed set of
# voters; a
# `reconfigure` command seals it and starts the next (see replica.py).


@dataclass(frozen=True)
class Prepare:
    sender: str
    epoch: int
    ballot: Ballot
    from_slot: int


@dataclass(frozen=True)
class Promise:
    sender: str
    epoch: int
    ballot: Ballot
    accepted: tuple[tuple[int, Ballot, Command], ...]


@dataclass(frozen=True)
class Accept:
    sender: str
    epoch: int
    ballot: Ballot
    slot: int
    command: Command


@dataclass(frozen=True)
class Accepted:
    sender: str
    epoch: int
    ballot: Ballot
    slot: int


@dataclass(frozen=True)
class Nack:
    """Refusal: the acceptor already promised a higher ballot."""

    sender: str
    epoch: int
    ballot: Ballot
    promised: Ballot


@dataclass(frozen=True)
class Commit:
    sender: str
    epoch: int
    slot: int
    command: Command


@dataclass(frozen=True)
class Heartbeat:
    sender: str
    epoch: int
    ballot: Ballot
    next_slot: int


@dataclass(frozen=True)
class HeartbeatAck:
    """A follower telling the leader it heard its heartbeat - proof of a majority."""

    sender: str
    epoch: int
    ballot: Ballot


@dataclass(frozen=True)
class CatchUp:
    """Asks for the chosen commands of `epoch` from `from_slot` on (any epoch, however old)."""

    sender: str
    epoch: int
    from_slot: int


@dataclass(frozen=True)
class Snapshot:
    """The state as of (`at_epoch`, `at_slot`), for a node that asks for a log that is gone."""

    sender: str
    at_epoch: int
    at_slot: int
    state: dict


@dataclass(frozen=True)
class Forward:
    """A command a follower hands to the leader."""

    sender: str
    command: Command


Message = (
    Prepare
    | Promise
    | Accept
    | Accepted
    | Nack
    | Commit
    | Heartbeat
    | HeartbeatAck
    | CatchUp
    | Snapshot
    | Forward
)
