from app.gguf.constants import GGMLQuantizationType
from app.gguf.dequant.base import QuantStrategy
from app.gguf.dequant.iq1_family import IQ1_MStrategy, IQ1_SStrategy
from app.gguf.dequant.iq2_family import IQ2_SStrategy, IQ2_XSStrategy, IQ2_XXSStrategy
from app.gguf.dequant.iq3_family import IQ3_SStrategy, IQ3_XXSStrategy
from app.gguf.dequant.iq_ternary import (
    IQ4_NLStrategy,
    IQ4_XSStrategy,
    TQ1_0Strategy,
    TQ2_0Strategy,
)
from app.gguf.dequant.kquants import Q4_KStrategy, Q5_KStrategy, Q6_KStrategy
from app.gguf.dequant.kquants_extended import Q2_KStrategy, Q3_KStrategy, Q8_KStrategy
from app.gguf.dequant.legacy import (
    Q4_0Strategy,
    Q4_1Strategy,
    Q5_0Strategy,
    Q5_1Strategy,
    Q8_0Strategy,
)
from app.gguf.dequant.simple import BF16Strategy, F16Strategy, F32Strategy
from app.server.errors import UnsupportedQuantTypeError


class QuantStrategyRegistry:
    """Looks up the QuantStrategy for a GGML quant type; fails closed."""

    _STRATEGIES: dict[GGMLQuantizationType, QuantStrategy] = {
        GGMLQuantizationType.F32: F32Strategy(),
        GGMLQuantizationType.F16: F16Strategy(),
        GGMLQuantizationType.BF16: BF16Strategy(),
        GGMLQuantizationType.Q8_0: Q8_0Strategy(),
        GGMLQuantizationType.Q4_0: Q4_0Strategy(),
        GGMLQuantizationType.Q4_1: Q4_1Strategy(),
        GGMLQuantizationType.Q5_0: Q5_0Strategy(),
        GGMLQuantizationType.Q5_1: Q5_1Strategy(),
        GGMLQuantizationType.Q2_K: Q2_KStrategy(),
        GGMLQuantizationType.Q3_K: Q3_KStrategy(),
        GGMLQuantizationType.Q4_K: Q4_KStrategy(),
        GGMLQuantizationType.Q5_K: Q5_KStrategy(),
        GGMLQuantizationType.Q6_K: Q6_KStrategy(),
        GGMLQuantizationType.Q8_K: Q8_KStrategy(),
        GGMLQuantizationType.IQ4_NL: IQ4_NLStrategy(),
        GGMLQuantizationType.IQ4_XS: IQ4_XSStrategy(),
        GGMLQuantizationType.TQ1_0: TQ1_0Strategy(),
        GGMLQuantizationType.TQ2_0: TQ2_0Strategy(),
        GGMLQuantizationType.IQ2_XXS: IQ2_XXSStrategy(),
        GGMLQuantizationType.IQ2_XS: IQ2_XSStrategy(),
        GGMLQuantizationType.IQ2_S: IQ2_SStrategy(),
        GGMLQuantizationType.IQ3_XXS: IQ3_XXSStrategy(),
        GGMLQuantizationType.IQ3_S: IQ3_SStrategy(),
        GGMLQuantizationType.IQ1_S: IQ1_SStrategy(),
        GGMLQuantizationType.IQ1_M: IQ1_MStrategy(),
    }

    def get(self, ggml_type: int) -> QuantStrategy:
        try:
            quant_type = GGMLQuantizationType(ggml_type)
        except ValueError:
            raise UnsupportedQuantTypeError(
                f"Unrecognized GGML quant type id: {ggml_type}"
            ) from None

        strategy = self._STRATEGIES.get(quant_type)
        if strategy is None:
            raise UnsupportedQuantTypeError(f"Unsupported GGUF quant type: {quant_type.name}")
        return strategy

    def supported_names(self) -> list[str]:
        """Every `GGMLQuantizationType` name this registry can currently

        dequantize - real data for `GET /api/health`, so a caller can ask
        "can you run this?" without hand-maintaining its own separate copy.
        """
        return [quant_type.name for quant_type in self._STRATEGIES]
