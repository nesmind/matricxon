import os
import sys
import threading
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

from app.cluster.freshness import FreshnessChecker


class NotFreshError(Exception):
    """Only a fresh install may be provisioned; the message lists why this one is not."""


class InvalidProvisionError(ValueError):
    pass


@dataclass(frozen=True)
class ProvisionRequest:
    models_dir: str
    prompt_cache_dir: str
    restart: bool = True
    advertise_url: str | None = None
    consensus_seeds: dict[str, str] | None = None
    admin_secret: str | None = None
    node_kind: str | None = None


class EnvFile:
    """Sets KEY=value lines in a .env file, leaving other lines alone.

    With `environ` (the real process environment in production) the change is made there too: a
    restart in place
    keeps the process environment, and a real variable beats.env, so a node pAIring started with
    MATRICXON_CLUSTER_ROLE=seed would come back from "cluster mode off" still a seed.
    """

    def __init__(self, path: Path, environ: MutableMapping[str, str] | None = None) -> None:
        self._path = path
        self._environ = environ

    def set(self, values: dict[str, str]) -> None:
        if self._environ is not None:
            self._environ.update(values)
        lines = self._path.read_text().splitlines() if self._path.exists() else []
        remaining = dict(values)
        out: list[str] = []
        for line in lines:
            key = next((k for k in remaining if line.strip().startswith(f"{k}=")), None)
            out.append(f"{key}={remaining.pop(key)}" if key else line)
        out += [f"{k}={v}" for k, v in remaining.items()]
        self._path.write_text("\n".join(out) + "\n")

    def remove(self, keys: list[str]) -> None:
        """Drops the lines of these keys (a setting back to its default)."""
        for key in keys if self._environ is not None else []:
            self._environ.pop(key, None)
        if not keys or not self._path.exists():
            return
        kept = [
            line
            for line in self._path.read_text().splitlines()
            if not any(line.strip().startswith(f"{key}=") for key in keys)
        ]
        self._path.write_text("\n".join(kept) + "\n" if kept else "")


def restart_in_place() -> None:
    """Replaces this process with a fresh one running the same command (same pid and arguments), so
    the new.env is
    read without a supervisor. Only for a fresh node: nothing loaded or cached.
    """
    os.execv(sys.executable, [sys.executable, *sys.argv])


class NodeProvisioner:
    """Turns a fresh install into a cluster joiner: writes the cluster settings into its .env and,
    when asked,
    restarts it in place. The cache key isn't sent: a joiner reads the shared key file beside the
    shared cache folder.
    """

    def __init__(self, freshness: FreshnessChecker, env: EnvFile, kill=None) -> None:
        self._freshness = freshness
        self._env = env
        self._kill = kill or restart_in_place

    def provision(self, request: ProvisionRequest) -> bool:
        """True when a restart was scheduled. Refuses a node that is not fresh."""
        report = self._freshness.report()
        if not report.fresh:
            raise NotFreshError("; ".join(report.reasons))
        self._validate(request)
        settings = {
            "MATRICXON_CLUSTER_ROLE": "joiner",
            "MATRICXON_MODELS_DIR": request.models_dir,
            "MATRICXON_PROMPT_CACHE_DIR": request.prompt_cache_dir,
        }
        if request.advertise_url:
            settings["MATRICXON_ADVERTISE_URL"] = request.advertise_url
        if request.consensus_seeds:
            settings["MATRICXON_CONSENSUS_SEEDS"] = ",".join(
                f"{node_id}={url}" for node_id, url in request.consensus_seeds.items()
            )
        if request.admin_secret:
            settings["MATRICXON_ADMIN_SECRET"] = request.admin_secret
        if request.node_kind:
            settings["MATRICXON_NODE_KIND"] = request.node_kind
        self._env.set(settings)
        if request.restart:
            threading.Timer(1.0, self._kill).start()
        return request.restart

    def set_admin_secret(self, secret: str) -> None:
        """Applies the secret now (no restart) and keeps it across restarts."""
        from app.config import settings

        settings.admin_secret = secret or None
        if secret:
            self._env.set({"MATRICXON_ADMIN_SECRET": secret})
        else:
            self._env.remove(["MATRICXON_ADMIN_SECRET"])

    @staticmethod
    def _validate(request: ProvisionRequest) -> None:
        for name, value in (
            ("models_dir", request.models_dir),
            ("prompt_cache_dir", request.prompt_cache_dir),
        ):
            folder = Path(value)
            if not folder.is_absolute() or not folder.is_dir():
                raise InvalidProvisionError(
                    f"{name} must be an existing absolute folder on this node: {value}"
                )
        key_file = Path(request.prompt_cache_dir).parent / "prompt_cache.key"
        if not key_file.is_file():
            raise InvalidProvisionError(
                f"the shared cache key {key_file} is not visible on this node"
            )
