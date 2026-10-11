import random
from collections import deque

from app.consensus.acceptor import Acceptor
from app.consensus.ballot import ZERO
from app.consensus.catchup import CatchUpPolicy
from app.consensus.config import PaxosConfig, Role, Send
from app.consensus.messages import (
    CatchUp,
    Command,
    Commit,
    Forward,
    Heartbeat,
    Message,
)
from app.consensus.pending import PendingCommands
from app.consensus.proposer import Campaign, Leadership
from app.consensus.quorum import QuorumWatch
from app.consensus.replica import Replica, reconfigure_payload
from app.consensus.roles import LeaderRole
from app.consensus.snapshots import SnapshotRole
from app.consensus.state_machine import StateMachine
from app.consensus.storage import Store


class PaxosNode(SnapshotRole, LeaderRole):
    """One member of a Multi-Paxos cluster. It touches no network or clock: the caller feeds it
    messages (`receive`) and time (`tick`) and gets messages out via `send`, so the same code runs
    in a seeded deterministic simulation and over real connections.

    A node hearing no leader campaigns (phase 1) with a higher ballot; once a majority promised, it
    leads and runs phase 2 per command, one phase 1 covering all later slots. Leadership is only a
    performance device: safety comes from the acceptors.

    The voters can change: `reconfigure` seals the current epoch and starts the next with the new
    voters (replica.py). A new, removed or behind node catches up from a peer, epoch by epoch.
    `genesis` is the first epoch's voters (id -> address); a later node passes `seeds` (ids to catch
    up from).
    """

    def __init__(
        self,
        node_id: str,
        store: Store,
        machine: StateMachine,
        send: Send,
        rng: random.Random,
        config: PaxosConfig | None = None,
        genesis: dict[str, str] | None = None,
        seeds: list[str] | None = None,
    ) -> None:
        self._durable = store.load()
        if self._durable.genesis is None and genesis is not None:
            store.save_genesis(genesis)
            self._durable.genesis = dict(genesis)
        self.id = node_id
        self.config = config or PaxosConfig()
        self.replica = Replica(machine, store, self._durable, self.config.snapshot_every)
        self.role: Role = "follower"
        self.leader: str | None = None
        self._store, self._send_out, self._rng = store, send, rng
        self._seeds = list(seeds or [])
        self._local: deque[Message] = deque()
        self._stale_told: dict[str, float] = {}
        self._snapshot_sent: dict[str, float] = {}
        self._catchup = CatchUpPolicy(
            node_id, self.replica, rng, self.config.retry_interval / 4, self.config.catchup_batch
        )
        self._pending = PendingCommands(self.replica)
        self._watch = QuorumWatch(self.config.election_timeout, rng)
        self._next_heartbeat = self._next_retry = self._election_deadline = 0.0
        self._enter_epoch()

    # the voters of the current epoch

    @property
    def epoch(self) -> int:
        return self.replica.epoch

    @property
    def members(self) -> list[str]:
        return sorted(self.replica.members)

    @property
    def quorum(self) -> int:
        return len(self.replica.members) // 2 + 1

    @property
    def is_voter(self) -> bool:
        return self.id in self.replica.members

    @property
    def is_leader(self) -> bool:
        return self.role == "leader"

    def quorum_status(self, now: float) -> tuple[bool, int | None]:
        """(is a majority of the voters in touch?, voters the leader can reach: leaders only).
        A leader needs a majority answering its heartbeats; a follower must hear the leader."""
        if not self.is_voter:
            return False, None
        if self.role == "leader":
            reachable = self._watch.reachable(self.id, self.members, now)
            return reachable >= self.quorum, reachable
        return self.role == "follower" and self._watch.leader_fresh(now), None

    def _enter_epoch(self) -> None:
        """A fresh Paxos instance: its own acceptor, no campaign, no leader."""
        self.acceptor = Acceptor(self.id, self.epoch, self._durable.epoch(self.epoch), self._store)
        self._raise_floor()
        self._max_round = self.acceptor.max_round
        self._campaign: Campaign | None = None
        self._leadership: Leadership | None = None
        self._leader_ballot = ZERO
        self.leader = None
        self.role = "follower"

    # the outside world

    def start(self, now: float) -> None:
        self._reset_election(now)

    def new_command(self, payload: object) -> Command:
        return Command(f"{self.id}:{self._rng.getrandbits(48):012x}", payload)

    def submit(self, command: Command, now: float) -> None:
        """Asks the cluster to apply `command` (once, however often this is retried)."""
        self._pending.add(command)
        self._route(command)
        self._drain(now)

    def reconfigure(self, members: dict[str, str], now: float) -> Command:
        """Asks the cluster to move to a new voter set (id -> address)."""
        command = self.new_command(reconfigure_payload(self.epoch, members))
        self.submit(command, now)
        return command

    def receive(self, msg: Message, now: float) -> None:
        self._handle(msg, now)
        self._drain(now)

    def tick(self, now: float) -> None:
        if self.role == "leader" and self._watch.must_step_down(
            self.id, self.members, self.quorum, now
        ):
            self._step_down()
            self._reset_election(now)
        if self.role == "leader":
            if now >= self._next_heartbeat:
                self._next_heartbeat = now + self.config.heartbeat
                self._broadcast(
                    Heartbeat(self.id, self.epoch, self._leadership.ballot, self.replica.next_slot)
                )
        elif self.is_voter and now >= self._election_deadline:
            self._start_campaign(now)
        if self.replica.rotate():
            self._compact()
        pull = self._catchup.pull(now)
        if pull is not None:
            self._send(*pull)
        if now >= self._next_retry:
            self._next_retry = now + self.config.retry_interval
            self._resend_unchosen()
            self._route_pending()
            if not self.is_voter or self.role == "candidate":
                self._ask_for_log(self._catchup.random_peer(self.members, self._seeds), now)
        self._drain(now)

    # plumbing

    def _send(self, to: str, msg: Message) -> None:
        if to == self.id:
            self._local.append(msg)
        else:
            self._send_out(to, msg)

    def _broadcast(self, msg: Message) -> None:
        for member in self.members:
            self._send(member, msg)

    def _drain(self, now: float) -> None:
        while self._local:
            self._handle(self._local.popleft(), now)

    def _handle(self, msg: Message, now: float) -> None:
        epoch = getattr(msg, "epoch", None)
        if epoch is not None and not isinstance(msg, CatchUp):
            if epoch < self.epoch:
                self._tell_it_is_stale(msg.sender, now)
                return
            if epoch > self.epoch:
                self._catchup.note_ahead(msg.sender, epoch)
                self._ask_for_log(msg.sender, now)
                return
            if not self.is_voter and not isinstance(msg, Commit):
                return
        if self.role == "leader" and msg.sender in self.replica.members:
            self._watch.heard_from(msg.sender, now)
        # Prepare -> _on_prepare, CatchUp -> _on_catchup,...: one handler per message type
        getattr(self, f"_on_{type(msg).__name__.lower()}")(msg, now)

    def _reset_election(self, now: float) -> None:
        timeout = self.config.election_timeout * self.config.election_patience
        self._election_deadline = now + timeout * (1 + self._rng.random())

    def _ask_for_log(self, peer: str | None, now: float, force: bool = False) -> None:
        request = self._catchup.request(peer, now, force) if peer else None
        if request is not None:
            self._send(peer, request)

    def _tell_it_is_stale(self, peer: str, now: float) -> None:
        """A node talking in a sealed epoch (it was cut off) learns there's a newer one: our
        heartbeat names it and the node asks for the log."""
        if (
            peer != self.id
            and now - self._stale_told.get(peer, -1e9) >= self.config.retry_interval / 4
        ):
            self._stale_told[peer] = now
            self._send(peer, Heartbeat(self.id, self.epoch, ZERO, self.replica.next_slot))

    def _commit(
        self, epoch: int, slot: int, command: Command, now: float, peer: str | None = None
    ) -> None:
        before = self.epoch
        self.replica.commit(epoch, slot, command)
        if self.epoch != before:
            self._enter_epoch()
            self._reset_election(now)
            self._route_pending()
            self._ask_for_log(peer, now, force=True)

    def _on_catchup(self, msg: CatchUp, now: float) -> None:
        if self._serve_snapshot(msg, now):
            return
        # Any epoch, however old; a node that hasn't reached `msg.epoch` has nothing.
        for slot, command in self.replica.chosen_from(
            msg.epoch, msg.from_slot, self.config.catchup_batch
        ):
            self._send(msg.sender, Commit(self.id, msg.epoch, slot, command))

    # commands from clients

    def _route(self, command: Command) -> None:
        if self.replica.applied(command.cmd_id):
            return
        if self.role == "leader":
            self._propose(command)
        elif self.leader is not None and self.leader != self.id:
            self._send(self.leader, Forward(self.id, command))

    def _route_pending(self) -> None:
        for command in self._pending.open():
            self._route(command)

    def _on_forward(self, msg: Forward, now: float) -> None:
        if self.role == "leader":
            self._propose(msg.command)
