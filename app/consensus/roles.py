from app.consensus.acceptor import Acceptor
from app.consensus.ballot import ZERO, Ballot
from app.consensus.catchup import CatchUpPolicy
from app.consensus.config import PaxosConfig, Role
from app.consensus.messages import (
    Accept,
    Accepted,
    Command,
    Commit,
    Heartbeat,
    HeartbeatAck,
    Message,
    Nack,
    Prepare,
    Promise,
)
from app.consensus.proposer import Campaign, Leadership
from app.consensus.quorum import QuorumWatch
from app.consensus.replica import Replica


class LeaderRole:
    """The election and the two Paxos phases from one node: campaigning, leading, following. Mixed
    into PaxosNode,
    which owns the state used here (epoch, replica,...).
    """

    # What PaxosNode provides (declared for readers and type checkers).
    id: str
    config: PaxosConfig
    replica: Replica
    acceptor: Acceptor
    role: Role
    leader: str | None
    members: list[str]
    quorum: int
    epoch: int
    _max_round: int
    _campaign: Campaign | None
    _leadership: Leadership | None
    _leader_ballot: Ballot
    _watch: QuorumWatch
    _catchup: CatchUpPolicy
    _election_deadline: float

    def _send(self, to: str, msg: Message) -> None: ...
    def _broadcast(self, msg: Message) -> None: ...
    def _reset_election(self, now: float) -> None: ...
    def _ask_for_log(self, peer: str | None, now: float, force: bool = False) -> None: ...
    def _commit(
        self, epoch: int, slot: int, command: Command, now: float, peer: str | None = None
    ) -> None: ...
    def _route_pending(self) -> None: ...

    # phase 1: becoming leader

    def _start_campaign(self, now: float) -> None:
        ballot = Ballot(self._max_round + 1, self.id)
        self._max_round = ballot.round
        self.role, self._leadership, self.leader = "candidate", None, None
        self._campaign = Campaign(ballot, self.quorum)
        self._broadcast(Prepare(self.id, self.epoch, ballot, self.replica.next_slot))
        self._election_deadline = now + self._watch.next_campaign_wait()

    def _on_prepare(self, msg: Prepare, now: float) -> None:
        reply = self.acceptor.prepare(msg)
        if isinstance(reply, Promise):
            self._max_round = max(self._max_round, msg.ballot.round)
            if msg.sender != self.id:
                if msg.ballot > self._own_ballot():
                    self._step_down()
                self._reset_election(now)
        self._send(msg.sender, reply)

    def _on_promise(self, msg: Promise, now: float) -> None:
        campaign = self._campaign
        if self.role != "candidate" or campaign is None or msg.ballot != campaign.ballot:
            return
        if campaign.add(msg.sender, msg.accepted):
            self._become_leader(campaign, now)

    def _become_leader(self, campaign: Campaign, now: float) -> None:
        chosen = self.replica.chosen.get(self.epoch, {})
        plan = campaign.plan(self.replica.next_slot, chosen)
        known = max([*chosen, *plan], default=-1)
        first_free = max(self.replica.next_slot, known + 1)
        self._leadership = Leadership(campaign.ballot, self.quorum, first_free)
        self.role, self._campaign = "leader", None
        self.leader, self._leader_ballot = self.id, campaign.ballot
        self._watch.became_leader(now)
        self._next_heartbeat = now
        for slot, command in sorted(plan.items()):
            self._propose_at(slot, command)
        self._route_pending()

    # phase 2: choosing commands

    def _propose_at(self, slot: int, command: Command) -> None:
        self._leadership.propose(slot, command)
        self._broadcast(Accept(self.id, self.epoch, self._leadership.ballot, slot, command))

    def _propose(self, command: Command) -> None:
        if self.replica.applied(command.cmd_id) or command.cmd_id in self._leadership.command_ids:
            return
        self._propose_at(self._leadership.next_free_slot, command)

    def _resend_unchosen(self) -> None:
        """Proposals lost on the way: ask the nodes that have not accepted them again."""
        if self.role != "leader":
            return
        for slot, command, acks in self._leadership.unchosen():
            accept = Accept(self.id, self.epoch, self._leadership.ballot, slot, command)
            for member in self.members:
                if member not in acks:
                    self._send(member, accept)

    def _on_accept(self, msg: Accept, now: float) -> None:
        reply = self.acceptor.accept(msg)
        if isinstance(reply, Accepted):
            self._max_round = max(self._max_round, msg.ballot.round)
            self._heard_leader(msg.ballot, msg.sender, now)
        self._send(msg.sender, reply)

    def _on_accepted(self, msg: Accepted, now: float) -> None:
        leadership = self._leadership
        if self.role != "leader" or leadership is None or msg.ballot != leadership.ballot:
            return
        command = leadership.accepted(msg.slot, msg.sender)
        if command is not None:
            voters, epoch = self.members, self.epoch
            self._commit(epoch, msg.slot, command, now)
            for member in voters:
                if member != self.id:
                    self._send(member, Commit(self.id, epoch, msg.slot, command))

    def _on_nack(self, msg: Nack, now: float) -> None:
        self._max_round = max(self._max_round, msg.promised.round)
        if msg.ballot == self._own_ballot():
            self._step_down()
            self._reset_election(now)

    def _on_commit(self, msg: Commit, now: float) -> None:
        self._commit(msg.epoch, msg.slot, msg.command, now, peer=msg.sender)
        if not self.is_voter and msg.sender != self.id:
            self._catchup.note_ahead(msg.sender, msg.epoch, msg.slot + 2)

    # following a leader

    def _on_heartbeat(self, msg: Heartbeat, now: float) -> None:
        if msg.ballot >= self.acceptor.promised or msg.sender == self.id:
            self._heard_leader(msg.ballot, msg.sender, now)
            if msg.sender != self.id:
                self._send(msg.sender, HeartbeatAck(self.id, self.epoch, msg.ballot))
            if msg.next_slot > self.replica.next_slot:
                self._catchup.note_ahead(msg.sender, msg.epoch, msg.next_slot)
                self._ask_for_log(msg.sender, now)

    def _on_heartbeatack(self, msg: HeartbeatAck, now: float) -> None:
        """Nothing to do: any message from a voter already counted as 'reachable' on arrival."""

    def _heard_leader(self, ballot: Ballot, sender: str, now: float) -> None:
        if ballot < self._leader_ballot:
            return
        changed = sender != self.leader
        self._leader_ballot, self.leader = ballot, sender
        if sender != self.id:
            self._watch.leader_contact(now)
            if changed:
                self._watch.leader_found()
            if ballot > self._own_ballot():
                self._step_down()
            self._reset_election(now)
            if changed:
                self._route_pending()

    def _own_ballot(self) -> Ballot:
        if self._campaign is not None:
            return self._campaign.ballot
        if self._leadership is not None:
            return self._leadership.ballot
        return ZERO

    def _step_down(self) -> None:
        self.role, self._campaign, self._leadership = "follower", None, None
