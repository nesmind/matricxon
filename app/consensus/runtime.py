import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx

from app.consensus.auth import MessageSigner
from app.consensus.cluster_id import ClusterIdKeeper
from app.consensus.codec import decode, encode
from app.consensus.events import ClusterEvents
from app.consensus.learners import Learners
from app.consensus.lease import LeaseKeeper, Leases
from app.consensus.messages import CatchUp, Command, Message
from app.consensus.node import PaxosNode
from app.consensus.status import RuntimeStatus
from app.consensus.timers import apply_timers
from app.consensus.voter_keeper import VoterKeeper
from app.consensus.voters import VoterChanges

logger = logging.getLogger(__name__)

SIGNATURE_HEADER = "X-Matricxon-Signature"
ADDRESS_HEADER = "X-Matricxon-Address"
ADDRESS_SIGNATURE_HEADER = "X-Matricxon-Address-Signature"
MESSAGE_PATH = "/api/consensus/message"
Post = Callable[[str, bytes, dict[str, str]], None]


class BadSignatureError(Exception):
    pass


class _PeerQueue:
    """The messages waiting for one peer, with their own sender thread. A peer that doesn't answer
    (down,
    partitioned) holds up only its own queue, never traffic to others; when full, the oldest are
    dropped (Paxos sends fresh ones anyway).
    """

    LIMIT = 200

    def __init__(self, peer: str, send: Callable[[str, list[Message]], None]) -> None:
        self._queue: queue.Queue[Message | None] = queue.Queue(maxsize=self.LIMIT)
        self._send = send
        self._peer = peer
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"consensus-send-{peer}"
        )
        self._thread.start()

    def put(self, msg: Message) -> None:
        while True:
            try:
                self._queue.put_nowait(msg)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass

    def close(self) -> None:
        self.put(None)
        self._thread.join(timeout=3.0)

    BATCH = 100

    def _run(self) -> None:
        closing = False
        while not closing and (msg := self._queue.get()) is not None:
            batch = [msg]
            while len(batch) < self.BATCH:
                try:
                    more = self._queue.get_nowait()
                except queue.Empty:
                    break
                if more is None:
                    closing = True
                    break
                batch.append(more)
            self._send(self._peer, batch)


class ConsensusRuntime:
    """Runs one PaxosNode for real: a ticker thread feeds it time, one sender thread per peer posts
    its messages
    (signed, fire-and-forget: Paxos tolerates a lost one) and `handle` feeds it what arrives. One
    lock serialises everything touching the node.
    """

    TICK_SECONDS = 0.05

    def __init__(
        self,
        node: PaxosNode,
        signer: MessageSigner,
        known_peers: dict[str, str],
        post: Post | None = None,
        clock: Callable[[], float] = time.monotonic,
        lease_names: tuple[str, ...] = (),
        lease_ttl: float = 15.0,
        probe: Callable[[str, str], bool] | None = None,
        own_address: str | None = None,
        target_voters: int = 0,
    ) -> None:
        self._own_address = own_address.rstrip("/") if own_address else None
        self._probe = probe  # (node id, address) -> does that address answer as that node?
        self.node = node
        self._signer = signer
        self._known = dict(known_peers)
        self._post = post or self._http_post
        self._batching = post is None
        self._clock = clock
        self._lock = threading.RLock()
        self._peers: dict[str, _PeerQueue] = {}
        self._peers_lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self.events = ClusterEvents()
        self.note: str | None = None
        self._client: httpx.Client | None = None
        node._send_out = self._enqueue
        self.learners = Learners()
        self.voter_keeper = VoterKeeper(self, target_voters)
        self.cluster_id = ClusterIdKeeper(self)
        self.leases = Leases(node.id, self.submit, self._config_view)
        self.keeper = LeaseKeeper(self.leases, lease_names, lease_ttl) if lease_names else None

    # lifecycle

    def start(self) -> None:
        with self._lock:
            self.node.start(self._clock())
            self.events.started(self.node)
        self._stop.clear()
        thread = threading.Thread(target=self._tick_loop, daemon=True, name="consensus-tick")
        thread.start()
        self._threads.append(thread)
        if self.keeper is not None:
            self.keeper.start()
        self.voter_keeper.start()
        self.cluster_id.start()

    def stop(self) -> None:
        self.events.stopped(self.node)
        self._stop.set()
        self.voter_keeper.stop()
        self.cluster_id.stop()
        if self.keeper is not None:
            self.keeper.stop()
        for thread in self._threads:
            thread.join(timeout=2.0)
        self._threads.clear()
        with self._peers_lock:
            peers, self._peers = list(self._peers.values()), {}
        for peer in peers:
            peer.close()
        if self._client is not None:
            self._client.close()

    # the node's two directions

    def _headers(self, body: bytes) -> dict[str, str]:
        headers = {SIGNATURE_HEADER: self._signer.sign(body)}
        if self._own_address:
            headers[ADDRESS_HEADER] = self._own_address
            headers[ADDRESS_SIGNATURE_HEADER] = self._signer.sign(self._own_address.encode())
        return headers

    def handle(
        self, body: bytes, signature: str, address: str = "", address_signature: str = ""
    ) -> None:
        """A message from another node. Raises BadSignatureError / CodecError if it is not valid."""
        if not self._signer.verify(body, signature):
            raise BadSignatureError("bad consensus signature")
        parsed = json.loads(body)
        msgs = [decode(item) for item in (parsed if isinstance(parsed, list) else [parsed])]
        with self._lock:
            for msg in msgs:
                if address and self._signer.verify(address.encode(), address_signature):
                    self._known[msg.sender] = address.rstrip("/")
                if isinstance(msg, CatchUp):
                    known = self._known.get(msg.sender, "")
                    self.learners.seen(msg.sender, known, msg.epoch, msg.from_slot, self._clock())
                self.node.receive(msg, self._clock())

    def _enqueue(self, to: str, msg: Message) -> None:
        if self._stop.is_set():
            return
        with self._peers_lock:
            peer = self._peers.get(to)
            if peer is None:
                peer = self._peers[to] = _PeerQueue(to, self._deliver)
        peer.put(msg)

    def _tick_loop(self) -> None:
        while not self._stop.wait(self.TICK_SECONDS):
            with self._lock:
                self.node.tick(self._clock())
                self.events.observe(self.node, self._config_view())

    def _deliver(self, to: str, msgs: list[Message]) -> None:
        url = self.address_of(to)
        if url is None:
            return
        wire = [encode(m) for m in msgs]
        # many go as one JSON list (a catch-up is hundreds); a custom `post` gets them singly
        parts = [wire] if self._batching and len(wire) > 1 else wire
        bodies = [json.dumps(part).encode() for part in parts]
        try:
            for body in bodies:
                self._post(url + MESSAGE_PATH, body, self._headers(body))
        except Exception:  # noqa: BLE001 - an unreachable peer is normal; the protocol retries
            logger.debug("consensus message to %s not delivered", to)
            self.events.delivery(to, url, ok=False)
        else:
            self.events.delivery(to, url, ok=True)

    def address_of(self, node_id: str) -> str | None:
        with self._lock:
            return self.node.replica.address_of(node_id, self._known)

    def _http_post(self, url: str, body: bytes, headers: dict[str, str]) -> None:
        if self._client is None:
            # a peer stalled by a slow disk should still get its message
            wait = max(1.5, self.node.config.election_timeout)
            self._client = httpx.Client(timeout=httpx.Timeout(wait, connect=0.5))
        self._client.post(
            url, content=body, headers={**headers, "Content-Type": "application/json"}
        )

    # what the rest of the server uses

    def submit(self, payload: Any, wait: float = 5.0) -> tuple[Command, bool, Any]:
        """Asks the cluster to apply `payload` and waits up to `wait` seconds for it.
        Returns (command, applied, result)."""
        with self._lock:
            command = self.node.new_command(payload)
            self.node.submit(command, self._clock())
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            with self._lock:
                if self.node.replica.applied(command.cmd_id):
                    return command, True, self.node.replica.result(command.cmd_id)
            time.sleep(0.02)
        return command, False, None

    def _config_view(self) -> dict[str, Any]:
        with self._lock:
            return dict(getattr(self.node.replica.machine, "config", {}))

    def add_voter(self, node_id: str, address: str, wait: float = 10.0) -> bool:
        """Makes `node_id` a voter, or records its new address if it already is one."""
        return VoterChanges(self).add(node_id, address, wait)

    def remove_voter(self, node_id: str, wait: float = 10.0, leave: bool = False) -> bool:
        """Takes `node_id` out of the voters (a new epoch); `leave`: on purpose, never re-added."""
        return VoterChanges(self).remove(node_id, wait, leave)

    def shared_config(self) -> dict[str, Any]:
        return self._config_view()

    def retune(self, election: float, heartbeat: float) -> None:
        with self._lock:
            apply_timers(self.node, election, heartbeat)

    def status(self) -> dict[str, Any]:
        return RuntimeStatus(self).report()
