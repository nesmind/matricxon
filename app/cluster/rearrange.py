"""`--force` of the cluster doctor: rebuild the cluster from the nodes that answer. The best node
keeps the cluster's state and becomes the only voter; every other node is reset to fresh and joins
as a learner, and enough become voters to reach three. Pure planning; scripts/cluster_doctor.sh
carries it out.

`python -m app.cluster.rearrange DIR` prints tab-separated lines: seed URL ID ADDRESS, quorum
URL|none, paths MODELS CACHE, drop VOTER_ID, reset URL, join URL ADVERTISE, down URL. Exit 1: no.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

from app.cluster.doctor import Probe
from app.cluster.doctor_cli import load

VOTERS = 3


@dataclass
class Rearrangement:
    seed: Probe
    address: str
    quorum_via: str | None  # a leader with a majority: other voters can be dropped through it
    models: str
    cache: str
    drop: list[str] = field(default_factory=list)
    others: list[Probe] = field(default_factory=list)
    down: list[str] = field(default_factory=list)

    def advertise(self, probe: Probe) -> str:
        """How the other nodes reach `probe`: what it advertises, else the address it was given."""
        return ((probe.status or {}).get("advertise_url") or probe.url).rstrip("/")

    def lines(self) -> list[str]:
        out = [
            f"seed\t{self.seed.url}\t{self.seed.node_id}\t{self.address}",
            f"quorum\t{self.quorum_via or 'none'}",
            f"paths\t{self.models}\t{self.cache}",
        ]
        out += [f"drop\t{v}" for v in self.drop] + [f"reset\t{p.url}" for p in self.others]
        return (
            out
            + [f"join\t{p.url}\t{self.advertise(p)}" for p in self.others]
            + [f"down\t{u}" for u in self.down]
        )


class RearrangeRefused(Exception):
    pass


def plan(probes: list[Probe]) -> Rearrangement:
    running = [p for p in probes if p.running]
    if not running:
        raise RearrangeRefused(
            "no node runs consensus: there is no cluster state to keep (start one from pAIring)"
        )
    seed = max(
        running, key=lambda p: (p.node_id in p.status["voters"], p.status["epoch"], p.total())
    )
    members = seed.status.get("members", {})
    paths = seed.status.get("config", {}).get("cluster/paths") or {}
    if not paths.get("models_dir") or not paths.get("prompt_cache_dir"):
        raise RearrangeRefused(
            "the cluster has no shared folders set (cluster/paths): set them in pAIring first"
        )
    leader = next(
        (p for p in running if p.status["role"] == "leader" and p.status.get("has_quorum")), None
    )
    return Rearrangement(
        seed=seed,
        address=(members.get(seed.node_id) or seed.status.get("advertise_url") or seed.url).rstrip(
            "/"
        ),
        quorum_via=leader.url if leader else None,
        models=paths["models_dir"],
        cache=paths["prompt_cache_dir"],
        drop=[v for v in seed.status["voters"] if v != seed.node_id],
        others=[p for p in probes if p.up and p is not seed],
        down=[p.url for p in probes if not p.up],
    )


def main(argv: list[str]) -> int:
    try:
        print("\n".join(plan(load(Path(argv[0]))).lines()))
    except RearrangeRefused as refusal:
        print(refusal, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
