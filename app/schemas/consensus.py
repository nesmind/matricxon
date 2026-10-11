from typing import Any

from pydantic import BaseModel


class ConsensusStatus(BaseModel):
    enabled: bool
    node_id: str | None = None
    epoch: int = 0
    role: str = "off"
    leader: str | None = None
    has_quorum: bool = False
    reachable: int | None = None
    is_voter: bool = False
    voters: list[str] = []
    members: dict[str, str] = {}
    leases: dict[str, Any] = {}
    holds: list[str] = []
    cluster_id: str | None = None
    applied_slot: int = 0
    applied_total: int = 0
    config: dict[str, Any] = {}
    timers: dict[str, float] = {}
    target_voters: int = 0
    nodes: dict[str, str] = {}
    removed: list[str] = []
    note: str | None = None
    advertise_url: str | None = None
    node_kind: str = "compute"
    admin_auth: bool = False
    addresses: dict[str, str] = {}
    seeds: dict[str, str] = {}


class VoterChange(BaseModel):
    id: str
    address: str


class ReachRequest(BaseModel):
    address: str


class ReachResult(BaseModel):
    reachable: bool
    node_id: str = ""  # which node answered there (a name like 127.0.0.1 can be this very node)
    detail: str = ""


class VoterResult(BaseModel):
    ok: bool


class ConfigChange(BaseModel):
    key: str
    value: Any = None


class ConfigResult(BaseModel):
    applied: bool
    result: Any = None
