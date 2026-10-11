"""Test isolation: the suite must never see the live server's configuration.

Settings come from the environment and the working directory's `.env`, which is the real server's
(cluster role, shared folders, consensus address...). Real environment variables win over `.env`, so
every setting that can reach a real path or the cluster is set here, before `app.config` is imported
(it builds settings at import).
"""

import atexit
import os
import shutil
import tempfile

_SANDBOX = tempfile.mkdtemp(prefix="matricxon-tests-")
atexit.register(shutil.rmtree, _SANDBOX, ignore_errors=True)

os.environ.update(
    {
        "MATRICXON_MODELS_DIR": f"{_SANDBOX}/models",
        "MATRICXON_PROMPT_CACHE_DIR": f"{_SANDBOX}/state/prompt_cache",
        "MATRICXON_NODE_STATE_DIR": f"{_SANDBOX}/node",
        "MATRICXON_PID_FILE": f"{_SANDBOX}/run/matricxon.pid",
        "MATRICXON_CLUSTER_ROLE": "off",
        "MATRICXON_ADVERTISE_URL": "",
        "MATRICXON_CONSENSUS_GENESIS": "",
        "MATRICXON_CONSENSUS_SEEDS": "",
        "MATRICXON_CONSENSUS_NODE_ID": "",
        "MATRICXON_CONSENSUS_SECRET": "",
        "MATRICXON_CACHE_KEY": "",
    }
)
