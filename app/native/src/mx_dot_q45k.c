/*
 * Integer dot products for Q4_K and Q5_K weight rows against a Q8_K activation row.
 *
 * Both share one layout family: 256 values per block as 8 sub-blocks of 32, each with its own
 * 6-bit (scale, min) pair packed into 12 bytes, value = d*scale*q - dmin*min. So per block:
 *   dot = y.d * (d * sum_j scale_j * sum(q*q8)_j  -  dmin * sum_j min_j * sum(q8)_j)
 * where sum(q8)_j comes straight from the activation's precomputed bsums - the min term costs
 * one multiply per sub-block, never one per value. Element order mirrors Q4_KStrategy/
 * Q5_KStrategy.dequantize in app/gguf/dequant/kquants.py exactly.
 */
#include "mx_common.h"

/* Same bit-packing as kquants.py's _get_scale_min_k4, one (scale, min) pair j in [0, 8). */
static inline void mx_scale_min_k4(int j, const uint8_t *s, int32_t *sc, int32_t *m) {
    if (j < 4) {
        *sc = s[j] & 63;
        *m = s[j + 4] & 63;
    } else {
        *sc = (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4);
        *m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

/* Folds one block's per-16 integer sums into its float contribution (shared by Q4_K/Q5_K). */
static inline float mx_k4_block(const uint8_t *scales, float d, float dmin, const int32_t *s16,
                                const mx_block_q8_k *y) {
    int32_t sumi = 0;
    int32_t summ = 0;
    for (int j = 0; j < 8; ++j) {
        int32_t sc, m;
        mx_scale_min_k4(j, scales, &sc, &m);
        sumi += sc * (s16[2 * j] + s16[2 * j + 1]);
        summ += m * (y->bsums[2 * j] + y->bsums[2 * j + 1]);
    }
    return y->d * (d * (float)sumi - dmin * (float)summ);
}

/* Q4_K block: {f16 d; f16 dmin; u8 scales[12]; u8 qs[128]} - 144 bytes. */
float mx_vec_dot_q4_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    int8_t aux[MX_QK_K];
    int32_t s16[16];
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q4_K_BYTES;
        const uint8_t *qs = b + 16;
        for (int c = 0; c < 4; ++c) {
            for (int l = 0; l < 32; ++l) {
                aux[c * 64 + l] = (int8_t)(qs[c * 32 + l] & 0x0F);
                aux[c * 64 + 32 + l] = (int8_t)(qs[c * 32 + l] >> 4);
            }
        }
        mx_sum16(aux, y[i].qs, s16);
        sumf += mx_k4_block(b + 4, mx_fp16(b), mx_fp16(b + 2), s16, &y[i]);
    }
    return sumf;
}

/* Q5_K block: {f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]} - 176 bytes. Each
 * nibble gets a 5th bit from qh, a different bit pair of the same 32 qh bytes per chunk. */
float mx_vec_dot_q5_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    int8_t aux[MX_QK_K];
    int32_t s16[16];
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q5_K_BYTES;
        const uint8_t *qh = b + 16;
        const uint8_t *qs = b + 48;
        for (int c = 0; c < 4; ++c) {
            for (int l = 0; l < 32; ++l) {
                uint8_t q = qs[c * 32 + l];
                aux[c * 64 + l] = (int8_t)((q & 0x0F) | (((qh[l] >> (2 * c)) & 1) << 4));
                aux[c * 64 + 32 + l] = (int8_t)((q >> 4) | (((qh[l] >> (2 * c + 1)) & 1) << 4));
            }
        }
        mx_sum16(aux, y[i].qs, s16);
        sumf += mx_k4_block(b + 4, mx_fp16(b), mx_fp16(b + 2), s16, &y[i]);
    }
    return sumf;
}
