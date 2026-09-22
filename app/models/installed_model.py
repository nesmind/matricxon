from dataclasses import dataclass


@dataclass(frozen=True)
class InstalledModel:
    tag: str
    path: str
    architecture: str
    capabilities: list[str]
    size_bytes: int
    family: str
    parameter_size: str
    context_length: int
