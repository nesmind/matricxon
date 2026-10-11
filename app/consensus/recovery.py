"""Manual recovery when most voters are gone for good (docs/CLUSTER_PAXOS.md).

A voter change needs a majority, so a cluster that lost most voters can't self-repair. This forces a
new group
of one from a surviving voter's journal, recording that the epoch ended with the survivor as sole
voter. A
command-line tool for a stopped server, never an endpoint. It can lose the newest writes (chosen
only on the lost voters).
"""

import argparse
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from app.config import settings
from app.consensus.messages import Command
from app.consensus.replica import Replica, reconfigure_payload
from app.consensus.state_machine import ClusterStateMachine
from app.consensus.storage import FileStore, MemoryStore, Store
from app.storage.node_identity import NodeIdentity
from app.storage.paths import StoragePaths


class RecoveryRefusedError(Exception):
    """The recovery must not (or cannot) be done; the message says why."""


@dataclass(frozen=True)
class RecoveryPlan:
    node_id: str
    epoch: int
    voters: dict[str, str]
    next_slot: int
    unconfirmed: int
    problem: str | None


class GroupRecovery:
    def __init__(self, store: Store) -> None:
        self._store = store

    def plan(self, node_id: str) -> RecoveryPlan:
        state = self._store.load()
        replica = Replica(ClusterStateMachine(), MemoryStore(), state)
        voters = dict(replica.members)
        here = state.epochs.get(replica.epoch)
        unconfirmed = sum(1 for s in here.accepted if s not in here.chosen) if here else 0
        problem = None
        if not voters:
            problem = "this journal knows no group (consensus never ran here)"
        elif node_id not in voters:
            problem = f"{node_id} is not a voter of {sorted(voters)}: use a survivor"
        elif list(voters) == [node_id]:
            problem = "this node is already a group of one: nothing to recover"
        return RecoveryPlan(node_id, replica.epoch, voters, replica.next_slot, unconfirmed, problem)

    def force_new_group(self, node_id: str) -> RecoveryPlan:
        """Ends the current epoch with `node_id` as the only voter (written to its journal)."""
        plan = self.plan(node_id)
        if plan.problem:
            raise RecoveryRefusedError(plan.problem)
        payload = reconfigure_payload(plan.epoch, {node_id: plan.voters[node_id]})
        command = Command(f"recover:{node_id}:{int(time.time())}", payload)
        self._store.save_chosen(plan.epoch, plan.next_slot, command)
        return plan


def _describe(plan: RecoveryPlan) -> str:
    voters = ", ".join(f"{i} ({a})" for i, a in sorted(plan.voters.items())) or "none"
    return (
        f"node {plan.node_id}: group of epoch {plan.epoch} = {voters}; "
        f"next slot {plan.next_slot}; {plan.unconfirmed} accepted here but not known chosen"
    )


def _server_running(pid_file: Path) -> bool:
    try:
        os.kill(int(pid_file.read_text().strip()), 0)
    except (OSError, ValueError):
        return False
    return True


def run(argv: list[str], out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m app.consensus.recover",
        description="Force a one-voter group from this node's journal (others lost for good).",
    )
    parser.add_argument("--force-new-group", action="store_true", help="do it (default: report)")
    parser.add_argument("--confirm", metavar="NODE_ID", help="this node's id, to confirm")
    args = parser.parse_args(argv)

    paths = StoragePaths(settings)
    node_id = settings.consensus_node_id or NodeIdentity(paths.node_id_file).load()
    journal = paths.consensus_journal
    if not journal.exists():
        out(f"No consensus journal at {journal}: nothing to recover here.")
        return 1
    if _server_running(paths.pid_file):
        out(f"The server is running (pid file {paths.pid_file}). Stop it first.")
        return 1
    recovery = GroupRecovery(FileStore(journal))
    plan = recovery.plan(node_id)
    out(_describe(plan))
    if plan.problem:
        out(f"Cannot recover: {plan.problem}")
        return 1
    if not args.force_new_group:
        out(f"Dry run, nothing changed. To do it add: --force-new-group --confirm {node_id}")
        return 0
    if args.confirm != node_id:
        out(f"Refusing: pass --confirm {node_id}. Be sure the other voters are gone for good.")
        return 1
    backup = journal.with_name(f"{journal.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(journal, backup)
    recovery.force_new_group(node_id)
    out(f"Done. Backup: {backup}. {node_id} is now the only voter of epoch {plan.epoch + 1}.")
    out("Next: start this node; reinstall other nodes FRESH and add them again (Add nodes).")
    out("Never bring an old voter back. Writes chosen only on the lost voters are gone.")
    return 0
