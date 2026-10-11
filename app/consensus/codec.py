"""Messages <-> JSON-able dicts for the wire. Decoding is strict: an unknown type or a missing
field is an error, never a guess."""

from dataclasses import fields
from typing import Any

from app.consensus import messages as m
from app.consensus.ballot import Ballot

_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        m.Prepare,
        m.Promise,
        m.Accept,
        m.Accepted,
        m.Nack,
        m.Commit,
        m.Heartbeat,
        m.HeartbeatAck,
        m.CatchUp,
        m.Snapshot,
        m.Forward,
    )
}


class CodecError(ValueError):
    pass


def encode(msg: m.Message) -> dict[str, Any]:
    out: dict[str, Any] = {"type": type(msg).__name__}
    for field in fields(msg):
        out[field.name] = _encode_value(getattr(msg, field.name))
    return out


def _encode_value(value: Any) -> Any:
    if isinstance(value, m.Command):
        return {"id": value.cmd_id, "payload": value.payload}
    if isinstance(value, Ballot):
        return [value.round, value.node]
    if isinstance(value, tuple):  # Promise.accepted: ((slot, ballot, command),...)
        return [[slot, _encode_value(ballot), _encode_value(cmd)] for slot, ballot, cmd in value]
    return value


def decode(data: dict[str, Any]) -> m.Message:
    try:
        cls = _TYPES[data["type"]]
        kwargs = {field.name: _decode_value(field.name, data[field.name]) for field in fields(cls)}
        return cls(**kwargs)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise CodecError(f"bad consensus message: {error!r}") from error


def _decode_value(name: str, value: Any) -> Any:
    if name == "ballot" or name == "promised":
        return Ballot(int(value[0]), str(value[1]))
    if name == "command":
        return _command(value)
    if name == "accepted":
        return tuple((int(s), Ballot(int(b[0]), str(b[1])), _command(c)) for s, b, c in value)
    return value


def _command(value: dict[str, Any]) -> m.Command:
    return m.Command(str(value["id"]), value["payload"])
