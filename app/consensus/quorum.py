import random


class QuorumWatch:
    """What a node knows about whether a majority of the voters can hear each other.

    A leader counts the voters that answered its heartbeats lately: with no majority for a whole
    election timeout
    it must step down (it can't commit and shouldn't look like the leader). A follower is fine while
    it hears the
    leader. A node that keeps losing elections waits longer each time, so a cluster without a
    majority isn't hammered.
    """

    MAX_BACKOFF = 4

    def __init__(self, election_timeout: float, rng: random.Random) -> None:
        self._timeout = election_timeout
        self._rng = rng
        self._heard: dict[str, float] = {}
        self._leader_since = 0.0
        self._leader_contact: float | None = None
        self._failures = 0

    def retune(self, election_timeout: float) -> None:
        self._timeout = election_timeout

    # the leader's view

    def became_leader(self, now: float) -> None:
        self._heard = {}
        self._leader_since = now
        self._failures = 0

    def heard_from(self, voter: str, now: float) -> None:
        self._heard[voter] = now

    def heard_within(self, voter: str, now: float) -> bool:
        """The leader heard from this voter within an election timeout."""
        return now - self._heard.get(voter, -1e9) < self._timeout

    def reachable(self, me: str, voters: list[str], now: float) -> int:
        """This node plus the voters that answered within an election timeout."""
        fresh = sum(1 for v in voters if v != me and now - self._heard.get(v, -1e9) < self._timeout)
        return 1 + fresh

    def must_step_down(self, me: str, voters: list[str], quorum: int, now: float) -> bool:
        """No majority for a full election timeout (not counting just after the election)."""
        settled = now - self._leader_since >= self._timeout
        return settled and self.reachable(me, voters, now) < quorum

    # a follower's view

    def leader_contact(self, now: float) -> None:
        self._leader_contact = now

    def leader_found(self) -> None:
        self._failures = 0

    def leader_fresh(self, now: float) -> bool:
        return self._leader_contact is not None and now - self._leader_contact < 2 * self._timeout

    # campaigns

    def next_campaign_wait(self) -> float:
        """Wait before the next campaign: one to two timeouts, doubling after each failure."""
        self._failures += 1
        factor = min(2 ** (self._failures - 1), self.MAX_BACKOFF)
        return self._timeout * (1 + self._rng.random()) * factor
