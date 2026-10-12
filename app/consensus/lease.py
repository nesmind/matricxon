import threading
import time
from collections.abc import Callable
from typing import Any

Submit = Callable[[Any, float], tuple[Any, bool, Any]]


class Leases:
    """Named, time-limited leases on the cluster's compare-and-set: one node owns "who pulls models"
    or "who cleans the cache"; when it dies the lease expires and another takes over. Taking or
    renewing goes through consensus, so two nodes never both win.

    Expiry compares each node's wall clock with the stored time, so clocks differing by more than
    the safety margin can let a node act briefly after its lease ended. Acceptable for these jobs
    (duplicate work, not damage: files are written atomically); never guard data that must not be
    touched twice.
    """

    PREFIX = "lease/"

    def __init__(
        self,
        node_id: str,
        submit: Submit,
        read_config: Callable[[], dict[str, Any]],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._id = node_id
        self._submit = submit
        self._read = read_config
        self._clock = clock

    def holder(self, name: str) -> str | None:
        """Who holds `name` right now according to this node's copy (None if free or expired)."""
        lease = self._read().get(self.PREFIX + name)
        if lease and lease.get("holder") and lease["expires"] > self._clock():
            return lease["holder"]
        return None

    def holds(self, name: str, margin: float = 0.0) -> bool:
        """Does this node hold `name` with at least `margin` seconds left?"""
        lease = self._read().get(self.PREFIX + name)
        return bool(
            lease and lease.get("holder") == self._id and lease["expires"] - margin > self._clock()
        )

    def try_acquire(self, name: str, ttl: float, wait: float = 2.0) -> bool:
        """Takes the lease if it is free, expired or already ours (which renews it)."""
        current = self._read().get(self.PREFIX + name)
        now = self._clock()
        if current and current.get("holder") not in (None, self._id) and current["expires"] > now:
            return False
        value = {"holder": self._id, "expires": now + ttl}
        payload = {"op": "cas", "key": self.PREFIX + name, "expected": current, "value": value}
        _, applied, result = self._submit(payload, wait)
        return bool(applied and result)

    def release(self, name: str, wait: float = 2.0) -> bool:
        """Gives the lease up early (only if ours)."""
        current = self._read().get(self.PREFIX + name)
        if not current or current.get("holder") != self._id:
            return False
        payload = {
            "op": "cas",
            "key": self.PREFIX + name,
            "expected": current,
            "value": {"holder": None, "expires": 0},
        }
        _, applied, result = self._submit(payload, wait)
        return bool(applied and result)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Every live lease: name -> {"holder", "expires"}."""
        now = self._clock()
        return {
            key[len(self.PREFIX) :]: dict(value)
            for key, value in self._read().items()
            if key.startswith(self.PREFIX) and value.get("holder") and value["expires"] > now
        }


class LeaseKeeper:
    """Keeps this node's claim on a set of leases: renews held ones well before expiry and tries
    free ones. Every
    node runs one; whichever compare-and-set lands first owns the job.
    """

    def __init__(
        self,
        leases: Leases,
        names: tuple[str, ...],
        ttl: float = 15.0,
        interval: float | None = None,
        ready: Callable[[], bool] | None = None,
    ) -> None:
        self._ready = ready  # False while the cluster cannot decide: renewals would only pile up
        self.leases = leases
        self.names = names
        self.ttl = ttl
        self._interval = interval or ttl / 3
        self._tasks: list[tuple[str, Callable[[], None]]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="consensus-leases")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def holds(self, name: str) -> bool:
        """True while this node owns `name` with a third of the lease left; past that it must assume
        it may have lost it."""
        return self.leases.holds(name, margin=self.ttl / 3)

    def add_task(self, lease: str, task: Callable[[], None]) -> None:
        """Runs `task` every round while this node owns `lease` (the owner does that job)."""
        self._tasks.append((lease, task))

    def tick(self) -> None:
        if self._ready is None or self._ready():
            for name in self.names:
                self.leases.try_acquire(name, self.ttl)
        for lease, task in self._tasks:
            if self.holds(lease):
                task()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a missed round is retried at the next one
                continue
