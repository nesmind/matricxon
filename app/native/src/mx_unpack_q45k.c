/*
 * Q4_K and Q5_K rows unpacked into mx_block_unpacked (see mx_common.h).
 *
 * Both share one layout family: 256 values per block as 8 sub-blocks of 32, each with its own
 * 6-bit (scale, min) pair packed into 12 bytes, value = d*scale*q - dmin*min. So per block:
 *   dot = y.d * (d * sum_s sc_s * sum(q*q8)_s  -  dmin * sum_s m_s * bsums_s)
 * with each 32-value (scale, min) spread over its two 16-value slots - the min term costs one
 * multiply per sub-block, never one per value.
 * Element order mirrors Q4_KStrategy/Q5_KStrategy.dequantize in app/gguf/dequant/kquants.py.
 */
#include "mx_common.h"

/* Header shared by Q4_K/Q5_K: {f16 d; f16 dmin; u8 scales[12]} - fills d, dmin and the per-16
 * scale and min (each 32-value pair spread over its two 16-value slots). */
static inline void mx_k4_header(const uint8_t *b, mx_block_unpacked *out) {
    uint32_t words[4];
    uint8_t sm[16];
    mx_k4_scale_min_words(b + 4, words);
    memcpy(sm, words, 16);
    for (int s = 0; s < 16; ++s) {
        out->sc[s] = sm[s >> 1];
        out->m[s] = sm[8 + (s >> 1)];
    }
    out->d = mx_fp16(b);
    out->dmin = mx_fp16(b + 2);
}

/* Q4_K block: {f16 d; f16 dmin; u8 scales[12]; u8 qs[128]} - 144 bytes. */
void mx_unpack_q4_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb) {
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q4_K_BYTES;
        const uint8_t *qs = b + 16;
        uint8_t *u = out[i].u;
        mx_k4_header(b, &out[i]);
        for (int c = 0; c < 4; ++c) {
            for (int l = 0; l < 32; ++l) {
                u[c * 64 + l] = qs[c * 32 + l] & 0x0F;
                u[c * 64 + 32 + l] = qs[c * 32 + l] >> 4;
            }
        }
    }
}

/* Q5_K block: {f16 d; f16 dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]} - 176 bytes. Each
 * nibble gets a 5th bit from qh, a different bit pair of the same 32 qh bytes per chunk. */
void mx_unpack_q5_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb) {
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q5_K_BYTES;
        const uint8_t *qh = b + 16;
        const uint8_t *qs = b + 48;
        uint8_t *u = out[i].u;
        mx_k4_header(b, &out[i]);
        for (int c = 0; c < 4; ++c) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q = qs[c * 32 + l];
                u[c * 64 + l] = (uint8_t)((q & 0x0F) | (((qh[l] >> (2 * c)) & 1) << 4));
                u[c * 64 + 32 + l] = (uint8_t)((q >> 4) | (((qh[l] >> (2 * c + 1)) & 1) << 4));
            }
        }
    }
}
