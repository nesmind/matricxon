"""The cluster's id: a UUID made once by the first leader, kept in the replicated config. It lives
in the journal, so resetting every node (a new cluster) makes a new one."""

import logging
import threading
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.consensus.runtime import ConsensusRuntime

logger = logging.getLogger("matricxon.cluster")
KEY = "cluster/id"


class ClusterIdKeeper:
    def __init__(self, runtime: "ConsensusRuntime", interval: float = 2.0) -> None:
        self._runtime = runtime
        self._interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="cluster-id")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                if self.step():
                    return
            except Exception:  # noqa: BLE001 - looked at again next round
                logger.exception("cluster: making the cluster id failed")

    def step(self) -> bool:
        """True once the cluster has an id. Only the leader, with a majority behind it, makes one (a
        compare-and-set, so two racing leaders still end with one)."""
        runtime = self._runtime
        if KEY in runtime.shared_config():
            return True
        with runtime._lock:
            node = runtime.node
            leading = node.role == "leader" and node.quorum_status(runtime._clock())[0]
        if not leading:
            return False
        value = uuid.uuid4().hex
        payload = {"op": "cas", "key": KEY, "expected": None, "value": value}
        if runtime.submit(payload)[1]:
            logger.info("cluster: the cluster id is %s", runtime.shared_config().get(KEY))
        return KEY in runtime.shared_config()
