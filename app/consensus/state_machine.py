from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


class StateMachine(ABC):
    """What the replicated log drives. `apply` must be deterministic: every node applies the same
    commands in the same order and has to end in the same state."""

    @abstractmethod
    def apply(self, payload: Any) -> Any: ...

    @abstractmethod
    def snapshot(self) -> Any:
        """The whole state as JSON-able data (a copy)."""

    @abstractmethod
    def restore(self, state: Any) -> None: ...


class ClusterStateMachine(StateMachine):
    """The cluster's shared facts: membership and config (key/value). Payloads are dicts:
    {"op": "set", "key", "value"} | {"op": "delete", "key"}
    | {"op": "cas", "key", "expected", "value"}
    | {"op": "add_node", "id", "address"} | {"op": "remove_node", "id"}."""

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.nodes: dict[str, str] = {}

    def snapshot(self) -> Any:
        return {"config": deepcopy(self.config), "nodes": dict(self.nodes)}

    def restore(self, state: Any) -> None:
        self.config, self.nodes = deepcopy(state["config"]), dict(state["nodes"])

    def apply(self, payload: Any) -> Any:
        op = payload["op"]
        if op == "set":
            self.config[payload["key"]] = payload["value"]
            return True
        if op == "delete":
            return self.config.pop(payload["key"], None) is not None
        if op == "cas":
            if self.config.get(payload["key"]) != payload["expected"]:
                return False
            self.config[payload["key"]] = payload["value"]
            return True
        if op == "add_node":
            self.nodes[payload["id"]] = payload["address"]
            return True
        if op == "remove_node":
            return self.nodes.pop(payload["id"], None) is not None
        raise ValueError(f"unknown op: {op}")
