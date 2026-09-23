/* f16 lookup table and the Q8_K activation quantizer - see mx_common.h. */
#include <math.h>

#include "mx_common.h"

float mx_fp16_table[1 << 16];

static float mx_fp16_to_fp32_calc(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            /* Subnormal f16: renormalize into a normal f32. */
            exp = 127 - 15 + 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x3FF;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

__attribute__((constructor)) static void mx_init_fp16_table(void) {
    for (uint32_t i = 0; i < (1u << 16); ++i) {
        mx_fp16_table[i] = mx_fp16_to_fp32_calc((uint16_t)i);
    }
}

void mx_quantize_row_q8_k(const float *x, mx_block_q8_k *y, int n) {
    const int nb = n / MX_QK_K;
    for (int i = 0; i < nb; ++i) {
        const float *xb = x + i * MX_QK_K;
        float amax = 0.0f;
        for (int j = 0; j < MX_QK_K; ++j) {
            float ax = fabsf(xb[j]);
            amax = ax > amax ? ax : amax;
        }
        if (amax == 0.0f) {
            memset(&y[i], 0, sizeof(y[i]));
            continue;
        }
        const float iscale = 127.0f / amax;
        for (int j = 0; j < MX_QK_K; ++j) {
            int v = (int)lrintf(xb[j] * iscale);
            y[i].qs[j] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v));
        }
        for (int s = 0; s < MX_QK_K / 16; ++s) {
            int sum = 0;
            for (int l = 0; l < 16; ++l) {
                sum += y[i].qs[16 * s + l];
            }
            y[i].bsums[s] = (int16_t)sum;
        }
        y[i].d = amax / 127.0f;
    }
}
