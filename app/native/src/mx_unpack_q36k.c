/*
 * Q3_K and Q6_K rows unpacked into mx_block_unpacked (see mx_common.h).
 *
 * Both use 16 sub-blocks of 16 values with one signed scale each and no min term; their signed
 * values are stored shifted to unsigned, with the shift undone through the activation's bsums
 * (see mx_block_unpacked). Element order mirrors Q3_KStrategy (kquants_extended.py) and
 * Q6_KStrategy (kquants.py) dequantize exactly.
 */
#include "mx_common.h"

/* Sub-block scales plus the offset term (see mx_block_unpacked): m[s] = scale * off. */
static inline void mx_scales_with_offset(const int8_t *scales, int16_t off,
                                         mx_block_unpacked *restrict out) {
    for (int s = 0; s < 16; ++s) {
        out->sc[s] = scales[s];
        out->m[s] = (int16_t)(scales[s] * off);
    }
}

/* Q3_K block: {u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d} - 110 bytes. Each value is 2
 * low bits from qs plus hmask's high bit, inverted: bit clear means -4. Stored unsigned as
 * u = q2 | hbit << 2 (0..7), i.e. value + 4. */
void mx_unpack_q3_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb) {
    int8_t scales[16];
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q3_K_BYTES;
        const uint8_t *hmask = b;
        const uint8_t *qs = b + 32;
        for (int n = 0; n < 2; ++n) {
            for (int j = 0; j < 4; ++j) {
                const int shift = 2 * j;
                const int hbit = 4 * n + j;
                uint8_t *o = out[i].u + n * 128 + j * 32;
                for (int l = 0; l < 32; ++l) {
                    o[l] = (uint8_t)(((qs[n * 32 + l] >> shift) & 3) |
                                     (((hmask[l] >> hbit) & 1) << 2));
                }
            }
        }
        mx_unpack_q3k_scales(b + 96, scales);
        mx_scales_with_offset(scales, 4, &out[i]);
        out[i].d = mx_fp16(b + 108);
        out[i].dmin = 0.0f;
    }
}

/* Q6_K block: {u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d} - 210 bytes. 6-bit values are
 * 4 bits from ql plus 2 from qh, offset by -32; two 128-value halves of four 32-lane groups.
 * Stored unsigned (0..63), i.e. value + 32. */
void mx_unpack_q6_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb) {
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q6_K_BYTES;
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql = b + half * 64;
            const uint8_t *qh = b + 128 + half * 32;
            uint8_t *o = out[i].u + half * 128;
            for (int l = 0; l < 32; ++l) {
                o[l] = (uint8_t)((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4));
                o[l + 32] = (uint8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4));
                o[l + 64] = (uint8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4));
                o[l + 96] = (uint8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4));
            }
        }
        mx_scales_with_offset((const int8_t *)(b + 192), 32, &out[i]);
        out[i].d = mx_fp16(b + 208);
        out[i].dmin = 0.0f;
    }
}
