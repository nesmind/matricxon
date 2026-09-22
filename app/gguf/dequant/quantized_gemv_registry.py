"""Dispatch table for real quantized-native GEMV compute (Phase 4 of the plan): maps a GGUF tensor's
real `GGMLQuantizationType` to the fused kernel + real on-disk `type_size` that
`quantized_gemv.py`/`quantized_gemv_legacy.py`/`quantized_gemv_extended.py` implement for it - the
same 11 real packed types `QuantStrategyRegistry` already dequantizes (F32/F16/BF16 excluded on
purpose, see `QuantizedLinear`'s own docstring for why). One place to look up "does a real fused
kernel exist for this type" rather than three separate imports per caller.
"""

from app.gguf.constants import GGMLQuantizationType
from app.gguf.dequant.quantized_gemv import Q4_K_TYPE_SIZE, Q6_K_TYPE_SIZE, qgemv_q4_k, qgemv_q6_k
from app.gguf.dequant.quantized_gemv_extended import (
    Q2_K_TYPE_SIZE,
    Q3_K_TYPE_SIZE,
    Q5_K_TYPE_SIZE,
    Q8_K_TYPE_SIZE,
    qgemv_q2_k,
    qgemv_q3_k,
    qgemv_q5_k,
    qgemv_q8_k,
)
from app.gguf.dequant.quantized_gemv_legacy import (
    Q4_0_TYPE_SIZE,
    Q4_1_TYPE_SIZE,
    Q5_0_TYPE_SIZE,
    Q5_1_TYPE_SIZE,
    Q8_0_TYPE_SIZE,
    qgemv_q4_0,
    qgemv_q4_1,
    qgemv_q5_0,
    qgemv_q5_1,
    qgemv_q8_0,
)

# {ggml_type: (gemv_fn, real on-disk type_size)} - every real packed type a QuantStrategy exists
# for except F32/F16/BF16 (never packed, nothing for a fused kernel to buy - see QuantizedLinear).
GEMV_KERNELS: dict[GGMLQuantizationType, tuple[callable, int]] = {
    GGMLQuantizationType.Q4_0: (qgemv_q4_0, Q4_0_TYPE_SIZE),
    GGMLQuantizationType.Q4_1: (qgemv_q4_1, Q4_1_TYPE_SIZE),
    GGMLQuantizationType.Q5_0: (qgemv_q5_0, Q5_0_TYPE_SIZE),
    GGMLQuantizationType.Q5_1: (qgemv_q5_1, Q5_1_TYPE_SIZE),
    GGMLQuantizationType.Q8_0: (qgemv_q8_0, Q8_0_TYPE_SIZE),
    GGMLQuantizationType.Q2_K: (qgemv_q2_k, Q2_K_TYPE_SIZE),
    GGMLQuantizationType.Q3_K: (qgemv_q3_k, Q3_K_TYPE_SIZE),
    GGMLQuantizationType.Q4_K: (qgemv_q4_k, Q4_K_TYPE_SIZE),
    GGMLQuantizationType.Q5_K: (qgemv_q5_k, Q5_K_TYPE_SIZE),
    GGMLQuantizationType.Q6_K: (qgemv_q6_k, Q6_K_TYPE_SIZE),
    GGMLQuantizationType.Q8_K: (qgemv_q8_k, Q8_K_TYPE_SIZE),
}


def has_gemv_kernel(ggml_type: int) -> bool:
    try:
        return GGMLQuantizationType(ggml_type) in GEMV_KERNELS
    except ValueError:
        return False
