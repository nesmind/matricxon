"""Fused CPU ops for the decode hot path: RMSNorm, rotary embedding, causal attention over the KV
cache, and the sampler's draw. Each is a few tiny torch dispatches (tens to hundreds of
microseconds apiece, per layer, per token) or - the sampler - a full-vocabulary sort; here each is
one native call. See app/native/src/mx_fused_ops.c, mx_attention.c, mx_sample.c (C) and
fused_ops_numba.py (the Numba twin, used when `Settings.gemv_backend` is "numba" or the C library
didn't build).

Every method returns None when it can't handle the inputs (non-float32, a GPU tensor, a batched
decode, a shape outside what the kernel covers) and the caller runs its original torch code - so
this is purely an accelerator, never a behavior switch. Off with `MATRICXON_ENABLE_FUSED_OPS=false`.
"""

import ctypes
import logging

import numpy as np
import torch

from app.native import fused_ops_numba as numba_ops
from app.native.gemm import NativeGemm

logger = logging.getLogger(__name__)

# Above this many query tokens attention is a compute-bound matmul (a prefill): torch's SDPA wins.
MAX_FUSED_QUERY_TOKENS = 16


def _ptr(tensor: torch.Tensor) -> int:
    return tensor.data_ptr()


class _NativeBackend:
    def __init__(self, lib: ctypes.CDLL, n_threads: int) -> None:
        self._lib, self._n_threads = lib, n_threads
        f32p, i64, c_int, c_float = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_float
        lib.mx_rms_norm.argtypes = [f32p, f32p, f32p, c_int, c_int, c_float]
        lib.mx_rope.argtypes = [f32p] * 4 + [c_int] * 3 + [i64] * 2
        lib.mx_attention.argtypes = (
            [f32p] * 4 + [c_int] * 5 + [i64] * 6 + [c_float, c_int, c_int, c_int]
        )
        lib.mx_attention.restype = c_int
        lib.mx_sample.argtypes = [
            f32p,
            c_int,
            f32p,
            c_int,
            c_float,
            c_float,
            c_int,
            c_float,
            c_float,
        ]
        lib.mx_sample.restype = c_int

    def rms_norm(self, x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, eps: float) -> None:
        self._lib.mx_rms_norm(_ptr(x), _ptr(w), _ptr(out), x.shape[0], x.shape[1], eps)

    def rope(
        self, x: torch.Tensor, out: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> None:
        heads, tokens, dim = x.shape
        self._lib.mx_rope(
            _ptr(x), _ptr(out), _ptr(cos), _ptr(sin), heads, tokens, dim, x.stride(0), x.stride(1)
        )

    def attention(self, q, k, v, out, scale: float, offset: int, window: int) -> None:
        status = self._lib.mx_attention(
            _ptr(q), _ptr(k), _ptr(v), _ptr(out),
            q.shape[0], k.shape[0], q.shape[1], k.shape[1], q.shape[2],
            q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
            scale, offset, window, self._n_threads,
        )  # fmt: skip
        if status != 0:
            raise RuntimeError("mx_attention ran out of memory")

    def sample(self, logits, ids, penalty, temperature, top_k, top_p, u) -> int:
        token = self._lib.mx_sample(
            _ptr(logits), logits.shape[0], _ptr(ids), ids.shape[0],
            penalty, temperature, top_k, top_p, u,
        )  # fmt: skip
        if token < 0:
            raise RuntimeError("mx_sample ran out of memory")
        return token


class _NumbaBackend:
    """The Numba twins. Attention is off by default: its inner loops run over the cache's strided
    views, which Numba can't vectorize, and it measured ~4x SLOWER than torch's SDPA at decode
    sizes (1.9 ms vs 0.45 ms, 300 cached tokens) - so this backend leaves attention to torch. The
    kernel stays (and is tested) in sync with the C one; `attention=True` turns it on."""

    def __init__(self, attention: bool = False) -> None:
        self.supports_attention = attention

    @staticmethod
    def rms_norm(x, w, out, eps: float) -> None:
        numba_ops.rms_norm(x.numpy(), w.numpy(), out.numpy(), np.float32(eps))

    @staticmethod
    def rope(x, out, cos, sin) -> None:
        numba_ops.rope(x.numpy(), out.numpy(), cos.numpy(), sin.numpy())

    @staticmethod
    def attention(q, k, v, out, scale: float, offset: int, window: int) -> None:
        numba_ops.attention(
            q.numpy(), k.numpy(), v.numpy(), out.numpy(), np.float32(scale), offset, window
        )

    @staticmethod
    def sample(logits, ids, penalty, temperature, top_k, top_p, u) -> int:
        return int(
            numba_ops.sample(
                logits.numpy(), ids.numpy(), np.float32(penalty), np.float32(temperature),
                top_k, np.float32(top_p), u,
            )
        )  # fmt: skip


def _is_f32_cpu(*tensors: torch.Tensor) -> bool:
    return all(t.dtype == torch.float32 and t.device.type == "cpu" for t in tensors)


class FusedOps:
    _active: "FusedOps | None" = None

    def __init__(self, backend: "_NativeBackend | _NumbaBackend") -> None:
        self._backend = backend

    @classmethod
    def configure(cls, enabled: bool) -> None:
        """Called at startup after `NativeGemm.configure`: native C when that library is loaded
        (`Settings.gemv_backend == "native"` and it built), else the Numba twins."""
        cls._active = None
        if not enabled:
            return
        gemm = NativeGemm.active()
        backend = _NativeBackend(gemm.library, gemm.n_threads) if gemm else _NumbaBackend()
        cls._active = cls(backend)
        logger.info("fused decode ops enabled (%s)", "native C" if gemm else "numba")

    @classmethod
    def active(cls) -> "FusedOps | None":
        return cls._active

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor | None:
        if not _is_f32_cpu(x, weight) or weight.dim() != 1 or x.shape[-1] != weight.shape[0]:
            return None
        rows = x.reshape(-1, x.shape[-1]).contiguous()
        out = torch.empty_like(rows)
        self._backend.rms_norm(rows, weight.detach().contiguous(), out, eps)
        return out.view(x.shape)

    def rope(self, q, k, cos, sin) -> tuple[torch.Tensor, torch.Tensor] | None:
        """q (1, H, T, D), k (1, Hkv, T, D) - views of a projection, any strides - and cos/sin
        (T, D). The result is new contiguous tensors, like the torch version's."""
        if not _is_f32_cpu(q, k, cos, sin) or q.dim() != 4 or q.shape[0] != 1 or cos.dim() != 2:
            return None
        if k.shape[0] != 1 or q.shape[-1] % 2 or cos.shape != (q.shape[2], q.shape[3]):
            return None
        if k.shape[2] != q.shape[2] or k.shape[3] != q.shape[3] or cos.shape != sin.shape:
            return None
        cos, sin = cos.contiguous(), sin.contiguous()
        outputs = []
        for x in (q, k):
            x = x[0] if x.stride(-1) == 1 else x[0].contiguous()
            out = torch.empty(x.shape, dtype=torch.float32)
            self._backend.rope(x, out, cos, sin)
            outputs.append(out.unsqueeze(0))
        return outputs[0], outputs[1]

    def attention(
        self, q, k, v, offset: object, scale: float, window: int = 0
    ) -> torch.Tensor | None:
        """Causal attention of q (1, H, Tq, D) over the cache's k/v (1, Hkv, L, D) - query i is at
        absolute position i + offset. Only for a few query tokens (decode)."""
        if not getattr(self._backend, "supports_attention", True):
            return None
        if not _is_f32_cpu(q, k, v) or not isinstance(offset, int):
            return None
        if (
            q.dim() != 4
            or q.shape[0] != 1
            or k.shape[0] != 1
            or q.shape[2] > MAX_FUSED_QUERY_TOKENS
        ):
            return None
        if q.shape[1] % k.shape[1] or q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            return None
        out = torch.empty(q.shape, dtype=torch.float32)
        self._backend.attention(q[0], k[0], v[0], out[0], scale, offset, window)
        return out

    @staticmethod
    def can_sample(logits: torch.Tensor, sampling: object) -> bool:
        """Whether `sample` covers this request: temperature > 0, and top-k on or top-p off (top-p
        over the whole vocabulary needs a full sort - torch keeps that case)."""
        if not _is_f32_cpu(logits) or logits.dim() != 1 or sampling.temperature <= 0.0:
            return False
        return sampling.top_k > 0 or sampling.top_p >= 1.0

    def sample(self, logits, generated_ids, sampling, u: float) -> int:
        """The whole sampler - repetition penalty, temperature, top-k, top-p, softmax, draw - in one
        pass (mx_sample.c). `u` is a uniform in [0, 1); only after `can_sample` said yes."""
        penalized = sampling.repeat_penalty != 1.0 and bool(generated_ids)
        ids = torch.tensor(
            [i for i in sorted(set(generated_ids)) if 0 <= i < logits.shape[0]]
            if penalized
            else [],
            dtype=torch.int32,
        )
        return self._backend.sample(
            logits.contiguous(), ids, sampling.repeat_penalty, sampling.temperature,
            sampling.top_k, sampling.top_p, u,
        )  # fmt: skip
