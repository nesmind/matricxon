"""`python -m app.consensus.recover` (see recovery.py)."""

import sys

from app.consensus.recovery import run

if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))
