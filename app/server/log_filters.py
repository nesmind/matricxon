import logging
import re

CONSENSUS_PATH = "/api/consensus/message"

# Settings.log_level: 0 warnings/errors, 1 per-request milestones, 2 per-layer trace.
LOG_LEVELS_BY_SETTING = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}


class ConsensusTrafficFilter(logging.Filter):
    """Drops the access-log line of each consensus message that went through (a heartbeat alone is
    several a second
    per peer) and of each turned away because consensus is off here (one warning a minute instead).
    A message refused for a bad signature or format still shows.
    """

    QUIET = re.compile(r" (204|503)( |$)")

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not (CONSENSUS_PATH in message and self.QUIET.search(message))


def quiet_cluster_logging(include_traffic: bool = False) -> None:
    """Keeps consensus chatter and the HTTP client's debug output out of the server log (constant
    while a cluster runs). `include_traffic` restores the consensus lines."""
    access = logging.getLogger("uvicorn.access")
    if not include_traffic and not any(
        isinstance(f, ConsensusTrafficFilter) for f in access.filters
    ):
        access.addFilter(ConsensusTrafficFilter())
    for noisy in ("httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
