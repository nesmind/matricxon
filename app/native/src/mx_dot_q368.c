/*
 * Integer dot products for Q3_K, Q6_K and Q8_0 weight rows against a Q8_K activation row.
 *
 * Q3_K/Q6_K use 16 sub-blocks of 16 values with one signed scale each and no min term, so
 * per block: dot = y.d * d * sum_s scale_s * sum(q*q8)_s. Element order mirrors
 * Q3_KStrategy (kquants_extended.py) and Q6_KStrategy (kquants.py) dequantize exactly.
 */
#include "mx_common.h"

static inline float mx_scaled_block(const int8_t *scales, float d, const int32_t *s16,
                                    const mx_block_q8_k *y) {
    int32_t sumi = 0;
    for (int s = 0; s < 16; ++s) {
        sumi += scales[s] * s16[s];
    }
    return y->d * d * (float)sumi;
}

/* Port of kquants_extended.py's _unpack_q3k_scales: 16 signed 6-bit scales packed into 12
 * bytes, reassembled at uint32 granularity (bits carry across bytes before masking), -32. */
static inline void mx_unpack_q3k_scales(const uint8_t *raw, int8_t *out) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0F0F0F0Fu;
    uint32_t w[3];
    uint32_t a[4];
    memcpy(w, raw, 12);
    const uint32_t tmp = w[2];
    a[0] = (w[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    a[1] = (w[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    a[2] = ((w[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    a[3] = ((w[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    memcpy(out, a, 16);
    for (int s = 0; s < 16; ++s) {
        out[s] = (int8_t)(out[s] - 32);
    }
}

/* Q3_K block: {u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d} - 110 bytes. Each value is 2
 * low bits from qs plus hmask's high bit, inverted: bit clear means -4. */
float mx_vec_dot_q3_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    int8_t aux[MX_QK_K];
    int8_t scales[16];
    int32_t s16[16];
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q3_K_BYTES;
        const uint8_t *hmask = b;
        const uint8_t *qs = b + 32;
        for (int n = 0; n < 2; ++n) {
            for (int j = 0; j < 4; ++j) {
                const int shift = 2 * j;
                const int hbit = 4 * n + j;
                for (int l = 0; l < 32; ++l) {
                    int q = (qs[n * 32 + l] >> shift) & 3;
                    int neg = ((hmask[l] >> hbit) & 1) ? 0 : 4;
                    aux[n * 128 + j * 32 + l] = (int8_t)(q - neg);
                }
            }
        }
        mx_unpack_q3k_scales(b + 96, scales);
        mx_sum16(aux, y[i].qs, s16);
        sumf += mx_scaled_block(scales, mx_fp16(b + 108), s16, &y[i]);
    }
    return sumf;
}

/* Q6_K block: {u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d} - 210 bytes. 6-bit values are
 * 4 bits from ql plus 2 from qh, offset by -32; two 128-value halves of four 32-lane groups. */
float mx_vec_dot_q6_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    int8_t aux[MX_QK_K];
    int32_t s16[16];
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q6_K_BYTES;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql = b + half * 64;
            const uint8_t *qh = b + 128 + half * 32;
            int8_t *out = aux + half * 128;
            for (int l = 0; l < 32; ++l) {
                out[l] = (int8_t)(((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
                out[l + 32] = (int8_t)(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
                out[l + 64] = (int8_t)(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                out[l + 96] = (int8_t)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
        }
        mx_sum16(aux, y[i].qs, s16);
        sumf += mx_scaled_block((const int8_t *)(b + 192), mx_fp16(b + 208), s16, &y[i]);
    }
    return sumf;
}

/* Q8_0 block: {f16 d; i8 qs[32]} - 34 bytes, eight per 256-value activation block. */
float mx_vec_dot_q8_0(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        float block_sum = 0.0f;
        for (int k = 0; k < MX_QK_K / MX_QK8_0; ++k) {
            const uint8_t *b = row + (size_t)(i * 8 + k) * MX_Q8_0_BYTES;
            const int8_t *qs = (const int8_t *)(b + 2);
            const int8_t *q8 = y[i].qs + k * MX_QK8_0;
            int32_t acc = 0;
            for (int l = 0; l < MX_QK8_0; ++l) {
                acc += (int16_t)qs[l] * (int16_t)q8[l];
            }
            block_sum += mx_fp16(b) * (float)acc;
        }
        sumf += y[i].d * block_sum;
    }
    return sumf;
}
