from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from app.consensus.messages import Message

Send = Callable[[str, Message], None]
Role = Literal["follower", "candidate", "leader"]


@dataclass(frozen=True)
class PaxosConfig:
    heartbeat: float = 0.05  # the leader says it is alive this often
    election_timeout: float = 0.3  # silence this long (x1 to x2, random) starts a campaign
    election_patience: float = (
        1.0  # waits this many times longer before campaigning (a busy node yields)
    )
    retry_interval: float = 0.2  # a command not applied yet is handed to the leader again
    catchup_batch: int = 100
    snapshot_every: int = (
        5000  # applied slots between snapshots; the log keeps one to two intervals
    )
