from typing import Literal

from pydantic import BaseModel


class ReadinessCheckInfo(BaseModel):
    id: str
    status: str
    what: str
    why: str
    setting: str | None
    size_bytes: int
    hint: str


class ReadinessResponse(BaseModel):
    role: str
    node_id: str
    ready: bool
    active: bool
    checks: list[ReadinessCheckInfo]
    nodes: list[str]
    node_kind: str = "compute"
    admin_auth: bool = False


class FreshnessResponse(BaseModel):
    fresh: bool
    reasons: list[str]


class ProvisionBody(BaseModel):
    models_dir: str
    prompt_cache_dir: str
    restart: bool = True
    advertise_url: str | None = None
    consensus_seeds: dict[str, str] | None = None
    admin_secret: str | None = None
    node_kind: Literal["compute", "voter"] | None = None


class ProvisionResponse(BaseModel):
    restarting: bool


class ConfigureBody(BaseModel):
    role: Literal["off", "seed"]
    models_dir: str | None = None
    prompt_cache_dir: str | None = None
    advertise_url: str | None = None
    restart: bool = True
    admin_secret: str | None = None
    node_kind: Literal["compute", "voter"] | None = None


class AdminSecretBody(BaseModel):
    secret: str = ""  # "" removes it (the routes become open)


class AdminSecretResponse(BaseModel):
    admin_auth: bool
