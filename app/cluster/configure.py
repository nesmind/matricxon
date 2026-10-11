import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from app.cluster.provisioning import EnvFile, InvalidProvisionError
from app.config import Settings
from app.storage.paths import StoragePaths


class ConfigureRefusedError(Exception):
    """This node may not be configured this way (the message says why)."""


@dataclass(frozen=True)
class ConfigureRequest:
    """What the admin wants the seed (the existing, populated node) to be. None leaves a
    setting alone; an empty string removes it."""

    role: Literal["off", "seed"]
    models_dir: str | None = None
    prompt_cache_dir: str | None = None
    advertise_url: str | None = None
    restart: bool = True
    admin_secret: str | None = None  # "" removes it
    node_kind: str | None = None


class NodeConfigurator:
    """Turns cluster mode on or off for a running node, managed by pAIring or not: writes its `.env`
    and
    restarts it gracefully (models unloaded, chat caches saved). Nothing is moved; the admin moved
    the folders.
    """

    def __init__(
        self,
        settings: Settings,
        paths: StoragePaths,
        env: EnvFile,
        restart: Callable[[], None],
        delay: float = 1.5,
    ) -> None:
        self._settings = settings
        self._paths = paths
        self._env = env
        self._restart = restart
        self._delay = delay

    def configure(self, request: ConfigureRequest) -> bool:
        """True when a restart was scheduled (only if something actually changes)."""
        if self._settings.cluster_role == "joiner":
            raise ConfigureRefusedError(
                "this node is a joiner: it is configured by its cluster's seed"
            )
        self._validate(request)
        wanted = self._wanted(request)
        changed = {k: v for k, v in wanted.items() if v != self._current().get(k)}
        if not changed:
            return False
        self._env.set({k: v for k, v in changed.items() if v is not None})
        self._env.remove([k for k, v in changed.items() if v is None])
        if request.restart:
            threading.Timer(self._delay, self._restart).start()
        return request.restart

    def _wanted(self, request: ConfigureRequest) -> dict[str, str | None]:
        wanted: dict[str, str | None] = {
            "MATRICXON_CLUSTER_ROLE": None if request.role == "off" else "seed"
        }
        for key, value in (
            ("MATRICXON_MODELS_DIR", request.models_dir),
            ("MATRICXON_PROMPT_CACHE_DIR", request.prompt_cache_dir),
            ("MATRICXON_ADVERTISE_URL", request.advertise_url),
            ("MATRICXON_ADMIN_SECRET", request.admin_secret),
            ("MATRICXON_NODE_KIND", request.node_kind),
        ):
            if value is not None:
                wanted[key] = value or None
        return wanted

    def _current(self) -> dict[str, str | None]:
        settings = self._settings
        return {
            "MATRICXON_CLUSTER_ROLE": None
            if settings.cluster_role == "off"
            else settings.cluster_role,
            "MATRICXON_MODELS_DIR": str(settings.models_dir),
            "MATRICXON_PROMPT_CACHE_DIR": str(settings.prompt_cache_dir),
            "MATRICXON_ADVERTISE_URL": settings.advertise_url,
            "MATRICXON_ADMIN_SECRET": settings.admin_secret,
            "MATRICXON_NODE_KIND": settings.node_kind,
        }

    def _validate(self, request: ConfigureRequest) -> None:
        for name, value in (
            ("models_dir", request.models_dir),
            ("prompt_cache_dir", request.prompt_cache_dir),
        ):
            if value:
                folder = Path(value)
                if not folder.is_absolute() or not folder.is_dir():
                    raise InvalidProvisionError(
                        f"{name} must be an existing absolute folder: {value}"
                    )
        cache = request.prompt_cache_dir or str(self._settings.prompt_cache_dir)
        if request.role == "seed" and not self._settings.cache_key:
            key_file = Path(cache).parent / "prompt_cache.key"
            if (
                not key_file.is_file()
            ):  # a cluster node never makes a key: it would orphan the saved chats
                raise InvalidProvisionError(
                    f"move the cache key to {key_file} first, or set MATRICXON_CACHE_KEY"
                )
