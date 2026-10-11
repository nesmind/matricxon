"""`python -m app.cluster.doctor_cli DIR [--prev DIR | --discover]`: a tab-separated line per
finding
(severity, code, node, text, action); exit 1 when something is not info. `--discover` prints nodes
the cluster
knows that were not probed. DIR: N.url/.status/.ready per node.
"""

import json
import sys
from pathlib import Path

from app.cluster.doctor import ClusterDoctor, Probe, same_node


def load(directory: Path) -> list[Probe]:
    def read(name: str) -> dict | None:
        try:
            return json.loads((directory / name).read_text())
        except (OSError, ValueError):
            return None

    numbers = sorted((p.stem for p in directory.glob("*.url")), key=int)
    return [
        Probe((directory / f"{i}.url").read_text().strip(), read(f"{i}.status"), read(f"{i}.ready"))
        for i in numbers
    ]


def discover(probes: list[Probe]) -> list[str]:
    """Addresses of the cluster's other nodes, not probed yet. A probed node is recognised by its
    id, whatever address the cluster has for it."""
    asked = {p.node_id for p in probes if p.node_id}
    found: list[str] = []
    for p in (p for p in probes if p.running):
        for node_id, address in {
            **p.status.get("members", {}),
            **p.status.get("nodes", {}),
        }.items():
            address = address.rstrip("/")
            if node_id not in asked and not any(
                same_node(address, o) for o in [*(q.url for q in probes), *found]
            ):
                found.append(address)
    return found


def main(argv: list[str]) -> int:
    if "--discover" in argv:
        print("\n".join(discover(load(Path(argv[0])))))
        return 0
    previous = None
    if "--prev" in argv:
        previous = {p.url: p.total() for p in load(Path(argv[argv.index("--prev") + 1]))}
    findings = ClusterDoctor(load(Path(argv[0])), previous).findings()
    for finding in findings:
        print(finding.line())
    return 1 if any(f.severity != "info" for f in findings) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
