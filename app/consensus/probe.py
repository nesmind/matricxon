"""Does an address really answer as the node it is supposed to belong to?"""

import httpx


def answers_as(node_id: str, address: str) -> bool:
    """True if GET <address>/api/consensus/status comes from the node called `node_id`.

    A voter list with a wrong address (a name meaning something else on another machine, such as
    `localhost`)
    wedges the cluster: nobody reaches the node meant to vote.
    """
    try:
        resp = httpx.get(
            address.rstrip("/") + "/api/consensus/status", timeout=httpx.Timeout(3.0, connect=2.0)
        )
        return resp.status_code == 200 and resp.json().get("node_id") == node_id
    except (httpx.HTTPError, ValueError):
        return False


def is_someone_else(node_id: str, address: str) -> bool:
    """True only if the address answers as another node or with no cluster (a reset node gets a new
    identity). A
    node that doesn't answer isn't "someone else": it may be down.
    """
    try:
        resp = httpx.get(
            address.rstrip("/") + "/api/consensus/status", timeout=httpx.Timeout(3.0, connect=2.0)
        )
        if resp.status_code != 200:
            return False
        return resp.json().get("node_id") != node_id
    except (httpx.HTTPError, ValueError):
        return False
