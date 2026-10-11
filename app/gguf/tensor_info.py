from dataclasses import dataclass


@dataclass(frozen=True)
class GGUFTensorInfo:
    """One entry from a GGUF file's tensor-info table. `shape` is in ggml's `ne[]` order (dims[0]
    fastest-changing), the reverse of PyTorch/numpy row-major; reversing it is the loader's job
    (app/gguf/loader.py), not this dataclass's."""

    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int

    @property
    def n_elements(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n
