/*
 * IQ2_XXS, IQ2_XS, IQ2_S - ported directly from their own already-oracle-validated Numba
 * kernels (app/gguf/dequant/quantized_gemv_iq2.py). Plain portable C, no SIMD - same rationale
 * as mx_dot_f32.c: correctness first for these new, numerically sensitive grid lookups.
 */
#include "mx_common.h"
#include "mx_iq_grids.h"

static inline float mx_iq2_sign(uint8_t signs, int j) {
    return (signs & MX_KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
}

float mx_vec_dot_iq2_xxs_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ2_XXS_BYTES;
        const float d = mx_fp16(blk);
        const uint8_t *qs = blk + 2;
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t *word = qs + ib32 * 8;
            uint32_t word1;
            memcpy(&word1, word + 4, sizeof(word1));
            const float dl = d * (0.5f + (float)(word1 >> 28)) * 0.25f;
            const float *base = xb + ib32 * 32;
            for (int group = 0; group < 4; ++group) {
                const uint8_t grid_idx = word[group];
                const uint8_t signs = MX_KSIGNS_IQ2XS[(word1 >> (7 * group)) & 127];
                const int8_t *grid = MX_IQ2XXS_GRID + (size_t)grid_idx * 8;
                const float *lane = base + group * 8;
                for (int j = 0; j < 8; ++j) {
                    acc += lane[j] * (dl * (float)grid[j] * mx_iq2_sign(signs, j));
                }
            }
        }
    }
    return acc;
}

float mx_vec_dot_iq2_xs_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ2_XS_BYTES;
        const float d = mx_fp16(blk);
        const uint8_t *qs_bytes = blk + 2;
        const uint8_t *scales = blk + 66;
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t sc = scales[ib32];
            const float db0 = d * (0.5f + (float)(sc & 0x0F)) * 0.25f;
            const float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
            const float *base = xb + ib32 * 32;
            for (int group = 0; group < 4; ++group) {
                uint16_t qval;
                memcpy(&qval, qs_bytes + (ib32 * 4 + group) * 2, sizeof(qval));
                const int grid_idx = qval & 511;
                const uint8_t signs = MX_KSIGNS_IQ2XS[qval >> 9];
                const float dl = group < 2 ? db0 : db1;
                const int8_t *grid = MX_IQ2XS_GRID + (size_t)grid_idx * 8;
                const float *lane = base + group * 8;
                for (int j = 0; j < 8; ++j) {
                    acc += lane[j] * (dl * (float)grid[j] * mx_iq2_sign(signs, j));
                }
            }
        }
    }
    return acc;
}

float mx_vec_dot_iq2_s_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ2_S_BYTES;
        const float d = mx_fp16(blk);
        const uint8_t *qs_lo = blk + 2;
        const uint8_t *sign_bytes = blk + 34;
        const uint8_t *qh = blk + 66;
        const uint8_t *scales = blk + 74;
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint8_t sc = scales[ib32];
            const float db0 = d * (0.5f + (float)(sc & 0x0F)) * 0.25f;
            const float db1 = d * (0.5f + (float)(sc >> 4)) * 0.25f;
            const int qh_val = qh[ib32];
            const float *base = xb + ib32 * 32;
            for (int group = 0; group < 4; ++group) {
                const int base_idx = qs_lo[ib32 * 4 + group];
                const int extra = (qh_val << (8 - 2 * group)) & 0x300;
                const int grid_idx = base_idx | extra;
                const uint8_t signs = sign_bytes[ib32 * 4 + group];
                const float dl = group < 2 ? db0 : db1;
                const int8_t *grid = MX_IQ2S_GRID + (size_t)grid_idx * 8;
                const float *lane = base + group * 8;
                for (int j = 0; j < 8; ++j) {
                    acc += lane[j] * (dl * (float)grid[j] * mx_iq2_sign(signs, j));
                }
            }
        }
    }
    return acc;
}
