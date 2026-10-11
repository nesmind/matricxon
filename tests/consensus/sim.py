"""A deterministic simulated network for Paxos nodes: seeded drops, duplicates, random delays
(messages overtake
each other), partitions and crashes with durable storage. Same seed, same run.
"""

import heapq
import random
from typing import Any

from app.consensus.config import PaxosConfig
from app.consensus.messages import Command, Message
from app.consensus.node import PaxosNode
from app.consensus.state_machine import ClusterStateMachine
from app.consensus.storage import MemoryStore


class RecordingMachine(ClusterStateMachine):
    """Remembers the order commands were applied in."""

    def __init__(self) -> None:
        super().__init__()
        self.applied: list[Any] = []

    def apply(self, payload: Any) -> Any:
        self.applied.append(payload)
        return super().apply(payload)


class SimCluster:
    TICK = 0.01

    def __init__(
        self,
        seed: int,
        size: int = 3,
        drop: float = 0.0,
        duplicate: float = 0.0,
        delay: tuple[float, float] = (0.001, 0.02),
        config: PaxosConfig | None = None,
    ) -> None:
        self.rng = random.Random(seed)
        self.seed = seed
        self.ids = [f"n{i}" for i in range(size)]
        self.genesis = {node_id: f"sim://{node_id}" for node_id in self.ids}
        self.drop, self.duplicate, self.delay = drop, duplicate, delay
        self.config = config or PaxosConfig()
        self.now = 0.0
        self.stores = {i: MemoryStore() for i in self.ids}
        self.machines: dict[str, RecordingMachine] = {}
        self.nodes: dict[str, PaxosNode | None] = {}
        self.blocked: set[frozenset[str]] = set()
        self.submitted: list[Command] = []
        self.at_risk: set[str] = set()
        self._events: list[tuple[float, int, str, Any]] = []
        self._seq = 0
        for node_id in self.ids:
            self._boot(node_id, genesis=self.genesis)
        self._push(self.TICK, "tick", None)

    # nodes

    def _boot(self, node_id: str, genesis: dict[str, str] | None = None) -> None:
        machine = RecordingMachine()
        self.stores.setdefault(node_id, MemoryStore())
        node = PaxosNode(
            node_id,
            self.stores[node_id],
            machine,
            lambda to, msg, src=node_id: self._send(src, to, msg),
            random.Random(f"{self.seed}-{node_id}-{self._seq}"),
            self.config,
            genesis=genesis,
            seeds=list(self.genesis),
        )
        self.machines[node_id], self.nodes[node_id] = machine, node
        node.start(self.now)

    def add_node(self, node_id: str) -> None:
        """A brand-new node: it knows nobody's log yet, only whom to ask for it."""
        self.ids.append(node_id)
        self._boot(node_id)

    def crash(self, node_id: str) -> None:
        node = self.nodes[node_id]
        if node is not None:
            self.at_risk.update(c.cmd_id for c in node._pending.open())
        self.nodes[node_id] = None

    def restart(self, node_id: str) -> None:
        if self.nodes[node_id] is None:
            self._boot(node_id, genesis=self.genesis if node_id in self.genesis else None)

    def live(self) -> list[PaxosNode]:
        return [n for n in self.nodes.values() if n is not None]

    def leaders(self) -> list[PaxosNode]:
        return [n for n in self.live() if n.is_leader and n.is_voter]

    def voters(self) -> list[str]:
        """The voter set of the newest epoch any live node has reached."""
        newest = max(self.live(), key=lambda n: n.epoch)
        return newest.members

    def reconfigure(self, members: list[str]) -> Command:
        """Asks the cluster (through any live voter) to move to exactly `members`."""
        node = next(n for n in self.live() if n.is_voter)
        command = node.reconfigure({m: f"sim://{m}" for m in members}, self.now)
        self.submitted.append(command)
        return command

    # network

    def partition(self, *groups: set[str]) -> None:
        """Nodes in different groups cannot talk; a node in no group is cut off from everyone."""
        grouped = [set(g) for g in groups]
        self.blocked = {
            frozenset((a, b))
            for a in self.ids
            for b in self.ids
            if a < b and not any(a in g and b in g for g in grouped)
        }

    def heal(self) -> None:
        self.blocked = set()

    def _send(self, src: str, dst: str, msg: Message) -> None:
        if frozenset((src, dst)) in self.blocked or self.rng.random() < self.drop:
            return
        copies = 2 if self.rng.random() < self.duplicate else 1
        for _ in range(copies):
            self._push(self.now + self.rng.uniform(*self.delay), "msg", (dst, msg))

    def _push(self, when: float, kind: str, data: Any) -> None:
        self._seq += 1
        heapq.heappush(self._events, (when, self._seq, kind, data))

    # running

    def run(self, seconds: float, check_every: int = 25) -> None:
        end, count = self.now + seconds, 0
        while self._events and self._events[0][0] <= end:
            self.now, _, kind, data = heapq.heappop(self._events)
            if kind == "tick":
                for node in self.live():
                    node.tick(self.now)
                self._push(self.now + self.TICK, "tick", None)
            else:
                dst, msg = data
                node = self.nodes[dst]
                if node is not None:
                    node.receive(msg, self.now)
            count += 1
            if count % check_every == 0:
                self.check_safety()
        self.now = end
        self.check_safety()

    def submit(self, node_id: str, payload: Any) -> Command:
        node = self.nodes[node_id]
        command = node.new_command(payload)
        self.submitted.append(command)
        node.submit(command, self.now)
        return command

    # invariants

    def check_safety(self) -> None:
        """No two nodes hold different commands for one slot, and every node's applied sequence is a
        prefix of the longest (same commands, same order)."""
        chosen: dict[tuple[int, int], Command] = {}
        for node_id in self.ids:
            for epoch, state in self.stores[node_id].load().epochs.items():
                for slot, command in state.chosen.items():
                    known = chosen.setdefault((epoch, slot), command)
                    assert known == command, f"epoch {epoch} slot {slot} diverged"
        sequences = sorted((m.applied for m in self.machines.values()), key=len)
        for shorter, longer in zip(sequences, sequences[1:], strict=False):
            assert longer[: len(shorter)] == shorter, "nodes applied different histories"

    def applied_everywhere(self, command: Command) -> bool:
        return all(n.replica.applied(command.cmd_id) for n in self.live())

    def applied_nowhere(self, command: Command) -> bool:
        return not any(n.replica.applied(command.cmd_id) for n in self.live())

    def converged(self) -> bool:
        states = {
            (repr(sorted(m.config.items())), repr(sorted(m.nodes.items())))
            for m in self.machines.values()
        }
        positions = {(n.epoch, n.replica.next_slot) for n in self.live()}
        return len(states) == 1 and len(positions) == 1
