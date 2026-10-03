import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PersistencePolicy:
    """Admin-editable limits of the on-disk prompt cache. Changes are saved to `path` so they
    survive a restart; the .env-based `Settings` values are only the first-run defaults."""

    enabled: bool
    budget_mb: int
    ttl_hours: int
    path: Path | None = None

    @property
    def budget_bytes(self) -> int:
        return self.budget_mb << 20

    @property
    def ttl_seconds(self) -> float:
        return self.ttl_hours * 3600.0

    @classmethod
    def load(cls, path: Path, enabled: bool, budget_mb: int, ttl_hours: int) -> "PersistencePolicy":
        """The saved policy if `path` holds a valid one, else the given defaults."""
        policy = cls(enabled, budget_mb, ttl_hours, path)
        try:
            saved = json.loads(path.read_text())
            policy.enabled = bool(saved["enabled"])
            policy.budget_mb = max(0, int(saved["budget_mb"]))
            policy.ttl_hours = max(1, int(saved["ttl_hours"]))
        except FileNotFoundError:
            pass
        except (ValueError, KeyError, TypeError, OSError):
            logger.warning("ignoring unreadable cache policy file %s", path)
        return policy

    def update(
        self,
        enabled: bool | None = None,
        budget_mb: int | None = None,
        ttl_hours: int | None = None,
    ) -> None:
        if enabled is not None:
            self.enabled = enabled
        if budget_mb is not None:
            self.budget_mb = max(0, budget_mb)
        if ttl_hours is not None:
            self.ttl_hours = max(1, ttl_hours)
        self.save()

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        self.path.write_text(json.dumps(data))
