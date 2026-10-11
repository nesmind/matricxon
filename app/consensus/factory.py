import logging
import random
import socket
from pathlib import Path

from app.config import Settings
from app.consensus.auth import MessageSigner
from app.consensus.config import PaxosConfig
from app.consensus.node import PaxosNode
from app.consensus.probe import answers_as
from app.consensus.runtime import ConsensusRuntime
from app.consensus.state_machine import ClusterStateMachine
from app.consensus.storage import FileStore


def own_address(port: int) -> str | None:
    """This machine's address on its usual network, as `http://ip:port`: what other nodes most
    likely use to reach it
    (nothing is sent; connecting a UDP socket only picks the route).
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))
            return f"http://{probe.getsockname()[0]}:{port}"
    except OSError:
        return None


class ConsensusConfigError(ValueError):
    """The consensus settings cannot start a node (the message says what to fix)."""


logger = logging.getLogger("matricxon.cluster")


class ConsensusFactory:
    """Builds this server's consensus runtime from its settings; None while consensus is off."""

    @staticmethod
    def parse_genesis(text: str) -> dict[str, str]:
        """ "a=http://h1:8420,b=http://h2:8420" -> {"a": "http://h1:8420", "b":...}"""
        members: dict[str, str] = {}
        for part in filter(None, (p.strip() for p in text.split(","))):
            node_id, sep, url = part.partition("=")
            if not sep or not node_id.strip() or not url.strip():
                raise ConsensusConfigError(f"consensus_genesis entry must be id=url: {part!r}")
            members[node_id.strip()] = url.strip().rstrip("/")
        return members

    def __init__(
        self, settings: Settings, default_node_id: str, journal: Path, key_file: Path
    ) -> None:
        self._settings = settings
        self._default_node_id = default_node_id
        self._journal = journal
        self._key_file = key_file

    def _has_history(self) -> bool:
        return self._journal.is_file() and self._journal.stat().st_size > 0

    LEASES = ("model_pull", "cache_clean")

    def build(self) -> ConsensusRuntime | None:
        """None while consensus is off. On when `consensus_genesis` is set (a fixed first group), or
        when the node has a
        cluster role and an address: a seed starts a group of one, a joiner is told which voters to
        catch up from.
        """
        settings = self._settings
        node_id = settings.consensus_node_id or self._default_node_id
        genesis: dict[str, str] | None = None
        seeds: dict[str, str] = {}
        if settings.consensus_genesis.strip():
            genesis = self.parse_genesis(settings.consensus_genesis)
            seeds = genesis
        elif settings.cluster_role == "seed" and (
            address := settings.advertise_url or own_address(settings.port)
        ):
            if not settings.advertise_url:
                logger.info(
                    "cluster: no advertise address set, starting the cluster as %s "
                    "(set MATRICXON_ADVERTISE_URL if the other nodes reach this one elsewhere)",
                    address,
                )
            genesis = {node_id: address.rstrip("/")}
        elif settings.cluster_role == "joiner" and settings.consensus_seeds.strip():
            seeds = self.parse_genesis(settings.consensus_seeds)
        elif settings.cluster_role in ("seed", "joiner") and self._has_history():
            # A node that has been in a cluster keeps its place whatever its settings say now (a
            # missing advertise address
            # once made a restarted seed run without consensus while still a voter, and the cluster
            # lost its quorum). Only role "off" leaves.
            if settings.consensus_seeds.strip():
                seeds = self.parse_genesis(settings.consensus_seeds)
        else:
            return None
        if genesis is not None and node_id not in genesis:
            raise ConsensusConfigError(
                f"this node ({node_id}) is not in the voters {sorted(genesis)}"
            )
        node = PaxosNode(
            node_id,
            FileStore(self._journal),
            ClusterStateMachine(),
            lambda to, msg: None,
            random.SystemRandom(),
            self._paxos_config(),
            genesis=genesis,
            seeds=list(seeds),
        )
        return ConsensusRuntime(
            node,
            MessageSigner(self._secret()),
            {**seeds, **(genesis or {})},
            lease_names=self.LEASES,
            probe=answers_as,
            own_address=settings.advertise_url or None,
            target_voters=settings.consensus_target_voters,
        )

    def _paxos_config(self) -> PaxosConfig:
        """Timers for a real network: a heartbeat every 0.2 s and a 1-2 s wait before a campaign
        (the simulation's faster defaults cost many messages a second per peer)."""
        election = self._settings.consensus_election_seconds
        return PaxosConfig(
            heartbeat=self._settings.consensus_heartbeat_seconds,
            election_timeout=election,
            election_patience=(
                1.0
                if self._settings.node_kind == "voter"
                else self._settings.consensus_compute_patience
            ),
            retry_interval=election / 2,
            snapshot_every=self._settings.consensus_snapshot_every,
        )

    def _secret(self) -> bytes:
        for configured in (self._settings.consensus_secret, self._settings.cache_key):
            if configured:
                return configured.encode()
        if self._key_file.exists():
            return self._key_file.read_bytes()
        raise ConsensusConfigError(
            "set MATRICXON_CONSENSUS_SECRET (or the shared cache key) to sign messages"
        )
