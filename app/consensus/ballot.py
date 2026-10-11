from typing import NamedTuple


class Ballot(NamedTuple):
    """A proposal number. Ordered by round, then by node id, so two nodes never share a ballot."""

    round: int
    node: str


ZERO = Ballot(0, "")
