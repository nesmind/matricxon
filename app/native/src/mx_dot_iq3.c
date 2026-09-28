/*
 * IQ3_XXS, IQ3_S - ported directly from their own already-oracle-validated Numba kernels
 * (app/gguf/dequant/quantized_gemv_iq3.py). Grid entries here are 4-value (not 8 like the IQ2
 * family), so each group of 8 activations pulls from two adjacent grid rows.
 */
#include "mx_common.h"
#include "mx_iq_grids.h"

static inline float mx_iq3_sign(uint8_t signs, int j) {
    return (signs & MX_KMASK_IQ2XS[j]) ? -1.0f : 1.0f;
}

float mx_vec_dot_iq3_xxs_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ3_XXS_BYTES;
        const float d = mx_fp16(blk);
        const uint8_t *grid_idx_bytes = blk + 2;
        const uint8_t *aux_bytes = blk + 66;
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            uint32_t aux;
            memcpy(&aux, aux_bytes + ib32 * 4, sizeof(aux));
            const float dl = d * (0.5f + (float)(aux >> 28)) * 0.5f;
            const float *base = xb + ib32 * 32;
            for (int group = 0; group < 4; ++group) {
                const uint8_t signs = MX_KSIGNS_IQ2XS[(aux >> (7 * group)) & 127];
                const uint8_t idx1 = grid_idx_bytes[ib32 * 8 + 2 * group];
                const uint8_t idx2 = grid_idx_bytes[ib32 * 8 + 2 * group + 1];
                const int8_t *g1 = MX_IQ3XXS_GRID + (size_t)idx1 * 4;
                const int8_t *g2 = MX_IQ3XXS_GRID + (size_t)idx2 * 4;
                const float *lane = base + group * 8;
                for (int j = 0; j < 4; ++j) {
                    acc += lane[j] * (dl * (float)g1[j] * mx_iq3_sign(signs, j));
                }
                for (int j = 0; j < 4; ++j) {
                    acc += lane[4 + j] * (dl * (float)g2[j] * mx_iq3_sign(signs, j + 4));
                }
            }
        }
    }
    return acc;
}

float mx_vec_dot_iq3_s_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ3_S_BYTES;
        const float d = mx_fp16(blk);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh = blk + 66;
        const uint8_t *signs = blk + 74;
        const uint8_t *scales = blk + 106;
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int i2 = 0; i2 < 4; ++i2) {
            const uint8_t sc = scales[i2];
            const float db1 = d * (float)(1 + 2 * (sc & 0x0F));
            const float db2 = d * (float)(1 + 2 * (sc >> 4));
            for (int half = 0; half < 2; ++half) {
                const float dl = half == 0 ? db1 : db2;
                const int qh_byte = qh[i2 * 2 + half];
                const uint8_t *qs_off = qs + i2 * 16 + half * 8;
                const uint8_t *signs_off = signs + i2 * 8 + half * 4;
                const float *out_base = xb + i2 * 64 + half * 32;
                for (int group = 0; group < 4; ++group) {
                    const int idx1 = qs_off[2 * group] | ((qh_byte << (8 - 2 * group)) & 256);
                    const int idx2 =
                        qs_off[2 * group + 1] | ((qh_byte << (7 - 2 * group)) & 256);
                    const uint8_t sign_byte = signs_off[group];
                    const int8_t *g1 = MX_IQ3S_GRID + (size_t)idx1 * 4;
                    const int8_t *g2 = MX_IQ3S_GRID + (size_t)idx2 * 4;
                    const float *lane = out_base + group * 8;
                    for (int j = 0; j < 4; ++j) {
                        acc += lane[j] * (dl * (float)g1[j] * mx_iq3_sign(sign_byte, j));
                    }
                    for (int j = 0; j < 4; ++j) {
                        acc += lane[4 + j] * (dl * (float)g2[j] * mx_iq3_sign(sign_byte, j + 4));
                    }
                }
            }
        }
    }
    return acc;
}
