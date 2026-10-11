"""Harsh random faults WITH voter changes (nodes added, voters replaced, leaders removed).

Usage: python -m tests.consensus.fuzz_reconfig FIRST_SEED LAST_SEED
"""

import random
import sys

from tests.consensus.sim import SimCluster


def run(seed: int) -> str | None:
    rng = random.Random(seed)
    size, drop, dup = rng.choice([3, 5]), rng.uniform(0, 0.4), rng.uniform(0, 0.25)
    cluster = SimCluster(seed, size=size, drop=drop, duplicate=dup, delay=(0.001, 0.12))
    spare = [f"x{i}" for i in range(6)]
    cluster.run(1.0)
    for step in range(70):
        action = rng.random()
        if action < 0.15:
            cluster.crash(rng.choice(cluster.ids))
        elif action < 0.30:
            cluster.restart(rng.choice(cluster.ids))
        elif action < 0.42:
            ids = list(cluster.ids)
            rng.shuffle(ids)
            cut = rng.randint(1, len(ids) - 1)
            cluster.partition(set(ids[:cut]), set(ids[cut:]))
        elif action < 0.52:
            cluster.heal()
        elif action < 0.62 and spare:
            cluster.add_node(spare.pop(0))
        elif action < 0.82:
            voters = [n for n in cluster.live() if n.is_voter]
            if voters:
                members = rng.sample(cluster.ids, rng.randint(2, min(6, len(cluster.ids))))
                cluster.submitted.append(
                    voters[0].reconfigure({m: f"sim://{m}" for m in members}, cluster.now)
                )
        voters = [n.id for n in cluster.live() if n.is_voter]
        if voters:
            cluster.submit(rng.choice(voters), {"op": "set", "key": f"k{step % 7}", "value": step})
        cluster.run(rng.uniform(0.02, 1.2))
    cluster.heal()
    cluster.drop = cluster.duplicate = 0.0
    for node_id in cluster.ids:
        cluster.restart(node_id)
    cluster.run(120.0)
    for node in cluster.live():
        if not node.is_voter:
            cluster.at_risk.update(c.cmd_id for c in node._pending.open())
    if len(cluster.leaders()) != 1:
        return f"{len(cluster.leaders())} leaders after healing (voters {cluster.voters()})"
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
    failures = 0
    for seed in range(first, last + 1):
        try:
            problem = run(seed)
        except AssertionError as error:
            problem = f"SAFETY: {error}"
        if problem:
            failures += 1
            print(f"seed {seed}: {problem}", flush=True)
    print(f"done: {last - first + 1} seeds, {failures} failures")
