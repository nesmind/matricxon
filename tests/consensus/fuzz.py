"""Harsher random fault injection than the unit tests: more nodes, up to 50% loss, long runs.
Usage: python -m tests.consensus.fuzz FIRST_SEED LAST_SEED
"""

import random
import sys

from tests.consensus.sim import SimCluster


def run(seed: int) -> str | None:
    rng = random.Random(seed)
    size = rng.choice([3, 5, 7])
    cluster = SimCluster(
        seed,
        size=size,
        drop=rng.uniform(0, 0.5),
        duplicate=rng.uniform(0, 0.3),
        delay=(0.001, 0.15),
    )
    cluster.run(1.0)
    for step in range(80):
        action = rng.random()
        if action < 0.25:
            cluster.crash(rng.choice(cluster.ids))
        elif action < 0.5:
            cluster.restart(rng.choice(cluster.ids))
        elif action < 0.65:
            ids = list(cluster.ids)
            rng.shuffle(ids)
            cut = rng.randint(1, size - 1)
            cluster.partition(set(ids[:cut]), set(ids[cut:]))
        elif action < 0.8:
            cluster.heal()
        live = [i for i in cluster.ids if cluster.nodes[i] is not None]
        if live:
            cluster.submit(rng.choice(live), {"op": "set", "key": f"k{step % 7}", "value": step})
        cluster.run(rng.uniform(0.02, 1.5))
    cluster.heal()
    cluster.drop = cluster.duplicate = 0.0
    for node_id in cluster.ids:
        cluster.restart(node_id)
    cluster.run(90.0)
    if len(cluster.leaders()) != 1:
        return f"{len(cluster.leaders())} leaders after healing"
    if not cluster.converged():
        return "nodes did not converge"
    for command in cluster.submitted:
        everywhere = cluster.applied_everywhere(command)
        if not everywhere and (
            command.cmd_id not in cluster.at_risk or not cluster.applied_nowhere(command)
        ):
            return f"{command.cmd_id} lost or partially applied"
    return None


if __name__ == "__main__":
    first, last = int(sys.argv[1]), int(sys.argv[2])
    failures = {}
    for seed in range(first, last + 1):
        try:
            problem = run(seed)
        except AssertionError as error:
            problem = f"SAFETY: {error}"
        if problem:
            failures[seed] = problem
            print(f"seed {seed}: {problem}", flush=True)
    print(f"done: {last - first + 1} seeds, {len(failures)} failures")
