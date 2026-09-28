"""Dispatch table for real quantized-native GEMV compute (Phase 4 of the plan): maps a GGUF tensor's
real `GGMLQuantizationType` to the fused kernel + real on-disk `type_size` that
`quantized_gemv.py`/`quantized_gemv_legacy.py`/`quantized_gemv_extended.py`/
`quantized_gemv_iq_ternary.py`/`quantized_gemv_iq2.py`/`quantized_gemv_iq3.py`/
`quantized_gemv_iq1.py` implement for it - every real packed type `QuantStrategyRegistry`
dequantizes except F32/F16/BF16 (never packed, nothing for a fused kernel to buy - see
`QuantizedLinear`'s own docstring for why). One place to look up "does a real fused kernel exist
for this type" rather than several separate imports per caller.
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
from app.gguf.dequant.quantized_gemv_iq1 import (
    IQ1_M_TYPE_SIZE,
    IQ1_S_TYPE_SIZE,
    qgemv_iq1_m,
    qgemv_iq1_s,
)
from app.gguf.dequant.quantized_gemv_iq2 import (
    IQ2_S_TYPE_SIZE,
    IQ2_XS_TYPE_SIZE,
    IQ2_XXS_TYPE_SIZE,
    qgemv_iq2_s,
    qgemv_iq2_xs,
    qgemv_iq2_xxs,
)
from app.gguf.dequant.quantized_gemv_iq3 import (
    IQ3_S_TYPE_SIZE,
    IQ3_XXS_TYPE_SIZE,
    qgemv_iq3_s,
    qgemv_iq3_xxs,
)
from app.gguf.dequant.quantized_gemv_iq_ternary import (
    IQ4_NL_TYPE_SIZE,
    IQ4_XS_TYPE_SIZE,
    TQ1_0_TYPE_SIZE,
    TQ2_0_TYPE_SIZE,
    qgemv_iq4_nl,
    qgemv_iq4_xs,
    qgemv_tq1_0,
    qgemv_tq2_0,
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
    GGMLQuantizationType.IQ4_NL: (qgemv_iq4_nl, IQ4_NL_TYPE_SIZE),
    GGMLQuantizationType.IQ4_XS: (qgemv_iq4_xs, IQ4_XS_TYPE_SIZE),
    GGMLQuantizationType.TQ1_0: (qgemv_tq1_0, TQ1_0_TYPE_SIZE),
    GGMLQuantizationType.TQ2_0: (qgemv_tq2_0, TQ2_0_TYPE_SIZE),
    GGMLQuantizationType.IQ2_XXS: (qgemv_iq2_xxs, IQ2_XXS_TYPE_SIZE),
    GGMLQuantizationType.IQ2_XS: (qgemv_iq2_xs, IQ2_XS_TYPE_SIZE),
    GGMLQuantizationType.IQ2_S: (qgemv_iq2_s, IQ2_S_TYPE_SIZE),
    GGMLQuantizationType.IQ3_XXS: (qgemv_iq3_xxs, IQ3_XXS_TYPE_SIZE),
    GGMLQuantizationType.IQ3_S: (qgemv_iq3_s, IQ3_S_TYPE_SIZE),
    GGMLQuantizationType.IQ1_S: (qgemv_iq1_s, IQ1_S_TYPE_SIZE),
    GGMLQuantizationType.IQ1_M: (qgemv_iq1_m, IQ1_M_TYPE_SIZE),
}


def has_gemv_kernel(ggml_type: int) -> bool:
    try:
        return GGMLQuantizationType(ggml_type) in GEMV_KERNELS
    except ValueError:
        return False
