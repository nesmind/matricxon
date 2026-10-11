"""The nodes that follow the log without voting, as the leader knows them: from their catch-up
requests (how far
they got; the address comes with the signed sender address).
"""


class Learners:
    FRESH = 30.0

    def __init__(self) -> None:
        self._seen: dict[str, tuple[str, int, int, float]] = {}

    def seen(self, node_id: str, address: str, epoch: int, next_slot: int, now: float) -> None:
        self._seen[node_id] = (address, epoch, next_slot, now)

    def caught_up(
        self, voters: list[str], epoch: int, next_slot: int, slack: int, now: float
    ) -> list[tuple[str, str]]:
        """(id, address) of learners within `slack` slots of this epoch's log, closest first."""
        found = []
        for node_id, (address, their_epoch, theirs, at) in self._seen.items():
            if node_id in voters or now - at > self.FRESH or their_epoch != epoch or not address:
                continue
            if next_slot - theirs <= slack:
                found.append((next_slot - theirs, node_id, address))
        return [(node_id, address) for _, node_id, address in sorted(found)]

    def known(self, now: float) -> dict[str, str]:
        """id -> address of every learner heard from lately that gave an address."""
        return {n: a for n, (a, _, _, at) in self._seen.items() if a and now - at <= self.FRESH}
