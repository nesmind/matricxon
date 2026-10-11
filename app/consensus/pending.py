from app.consensus.messages import Command
from app.consensus.replica import Replica


class PendingCommands:
    """Commands this node submitted that aren't applied yet. They go to the leader again on every
    retry (it may have
    changed or lost them); applying one more than once is harmless.
    """

    def __init__(self, replica: Replica) -> None:
        self._replica = replica
        self._commands: dict[str, Command] = {}

    def add(self, command: Command) -> None:
        self._commands[command.cmd_id] = command

    def open(self) -> list[Command]:
        for cmd_id in [c for c in self._commands if self._replica.applied(c)]:
            del self._commands[cmd_id]
        return list(self._commands.values())
