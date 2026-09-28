/*
 * IQ4_NL, IQ4_XS, TQ1_0, TQ2_0 - ported directly from their own already-oracle-validated Numba
 * kernels (app/gguf/dequant/quantized_gemv_iq_ternary.py) - see mx_common.h's own comment on
 * why these compute against the raw float activation row directly instead of mx_block_q8_k.
 * Plain portable C (no SIMD) - these types are new here, and getting the real bit-unpacking
 * right matters more than a first speed pass; a vectorized version is real, separate follow-up
 * work once correctness is proven the same way every other kernel here already was.
 */
#include "mx_common.h"

static const float MX_KVALUES_IQ4NL[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f,
                                            -22.0f,  -10.0f,  1.0f,   13.0f,  25.0f,  38.0f,
                                            53.0f,   69.0f,   89.0f,  113.0f};
static const uint8_t MX_POW3[6] = {1, 3, 9, 27, 81, 243};

/* Mirrors ggml's `((uint16_t)(byte*p) * 3) >> 8` base-3-digit-extraction trick - the uint8_t
 * multiply wraps mod 256 exactly like Numba's own np.uint8 port. Returns a digit in {0, 1, 2}. */
static inline int mx_tq_digit(uint8_t byte, uint8_t power) {
    uint8_t q = (uint8_t)(byte * power);
    return (int)(((uint16_t)q * 3u) >> 8);
}

float mx_vec_dot_iq4_nl_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = row + (size_t)b * MX_IQ4_NL_BYTES;
        const float d = mx_fp16(blk);
        const uint8_t *qs = blk + 2;
        const float *xb = x + (size_t)b * MX_IQ4_NL_BLOCK;
        for (int j = 0; j < 16; ++j) {
            const uint8_t qbyte = qs[j];
            const float lo = MX_KVALUES_IQ4NL[qbyte & 0x0F];
            const float hi = MX_KVALUES_IQ4NL[qbyte >> 4];
            acc += xb[j] * (d * lo) + xb[16 + j] * (d * hi);
        }
    }
    return acc;
}

float mx_vec_dot_iq4_xs_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ4_XS_BYTES;
        const float d = mx_fp16(blk);
        uint16_t sh;
        memcpy(&sh, blk + 2, sizeof(sh));
        const uint8_t *scales_l = blk + 4;
        const uint8_t *qs = blk + 8;
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int ib = 0; ib < 8; ++ib) {
            const uint8_t sl_byte = scales_l[ib / 2];
            const int ls_lo = (sl_byte >> (4 * (ib % 2))) & 0x0F;
            const int ls_hi = ((sh >> (2 * ib)) & 3) << 4;
            const float dl = d * (float)((ls_lo | ls_hi) - 32);
            const uint8_t *q = qs + ib * 16;
            const float *xo = xb + ib * 32;
            for (int j = 0; j < 16; ++j) {
                const uint8_t qbyte = q[j];
                const float lo = MX_KVALUES_IQ4NL[qbyte & 0x0F];
                const float hi = MX_KVALUES_IQ4NL[qbyte >> 4];
                acc += xo[j] * (dl * lo) + xo[16 + j] * (dl * hi);
            }
        }
    }
    return acc;
}

float mx_vec_dot_tq1_0_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_TQ1_0_BYTES;
        const uint8_t *qs = blk;
        const uint8_t *qh = blk + 48;
        const float d = mx_fp16(blk + 52);
        const float *xb = x + (size_t)sb * MX_QK_K;
        int col = 0;
        for (int n = 0; n < 5; ++n) {
            const uint8_t p = MX_POW3[n];
            for (int m = 0; m < 32; ++m) {
                acc += xb[col++] * ((float)(mx_tq_digit(qs[m], p) - 1) * d);
            }
        }
        for (int n = 0; n < 5; ++n) {
            const uint8_t p = MX_POW3[n];
            for (int m = 0; m < 16; ++m) {
                acc += xb[col++] * ((float)(mx_tq_digit(qs[32 + m], p) - 1) * d);
            }
        }
        for (int n = 0; n < 4; ++n) {
            const uint8_t p = MX_POW3[n];
            for (int m = 0; m < 4; ++m) {
                acc += xb[col++] * ((float)(mx_tq_digit(qh[m], p) - 1) * d);
            }
        }
    }
    return acc;
}

float mx_vec_dot_tq2_0_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_TQ2_0_BYTES;
        const uint8_t *qs = blk;
        const float d = mx_fp16(blk + 64);
        const float *xb = x + (size_t)sb * MX_QK_K;
        int col = 0;
        for (int j = 0; j < 64; j += 32) {
            for (int shift_idx = 0; shift_idx < 4; ++shift_idx) {
                const int shift = shift_idx * 2;
                for (int m = 0; m < 32; ++m) {
                    const int bits = (qs[j + m] >> shift) & 3;
                    acc += xb[col++] * ((float)(bits - 1) * d);
                }
            }
        }
    }
    return acc;
}
