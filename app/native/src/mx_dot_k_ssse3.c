/*
 * Fused K-quant dot products for decode (one token): the packed bits are unpacked straight into
 * SSSE3 registers and fed to pmaddubsw, with no mx_block_unpacked round trip through memory.
 * Unpacking into that buffer was ~2/3 of a decode step's kernel time (measured on this project's
 * Sandy Bridge laptop), while prefill amortizes it over every token and keeps using it.
 *
 * Same integer math as mx_block_unpacked + mx_dot_unpacked: unsigned values times q8 summed per
 * 16-value sub-block, times the sub-block's integer scale; Q3_K/Q6_K's signed offset (4 / 32)
 * comes back out through bsums, Q4_K/Q5_K's min through bsums as before. The integer sums are
 * identical and the float folds keep the same order, so results match the unpack path exactly.
 * int16 headroom: two pmaddubsw results are added before widening - at most 2 * 2*63*127 =
 * 32004 (Q6_K), below 32767.
 *
 * Scales stay in registers the whole way: decoded into a vector, sign/zero-extended to int16 and
 * broadcast per sub-block with one pshufb (the loops are unrolled, so its lane mask is a
 * constant). Decoding them into a byte array first and reading it back cost ~a third of the Q3_K
 * kernel - most likely store-forwarding stalls on the narrow-store / wide-load round trip.
 *
 * Only built with SSSE3 (every x86-64 CPU since 2006, including this project's); elsewhere
 * mx_gemm keeps the unpack path for decode too.
 */
#include "mx_common.h"

#if defined(__SSSE3__)

static inline __m128i mx_load(const void *p) {
    return _mm_loadu_si128((const __m128i *)p);
}

/* `value` in every byte whose bit `mask` is set, 0 elsewhere - picks a bit out of each byte with
 * and/compare instead of shifts (Sandy Bridge runs vector shifts and multiplies on the same port,
 * and the multiplies already keep it busy). */
static inline __m128i mx_bit_to(__m128i bytes, __m128i mask, __m128i value) {
    return _mm_and_si128(_mm_cmpeq_epi8(_mm_and_si128(bytes, mask), mask), value);
}

/* int16 lane k of `v` in all 8 lanes. */
static inline __m128i mx_lane(__m128i v, int k) {
    return _mm_shuffle_epi8(v, _mm_set1_epi16((short)(((2 * k + 1) << 8) | (2 * k))));
}

/* acc += (int16 lane k of `scales`) * (sum of the 8 int16 lanes of `pairs`), as 4 int32. */
static inline __m128i mx_acc_lane(__m128i acc, __m128i pairs, __m128i scales, int k) {
    return _mm_add_epi32(acc, _mm_madd_epi16(pairs, mx_lane(scales, k)));
}

/* 16 signed bytes -> int16 lanes 0-7 (lo) and 8-15 (hi). */
static inline void mx_s8_to_s16(__m128i v, __m128i *lo, __m128i *hi) {
    *lo = _mm_srai_epi16(_mm_unpacklo_epi8(v, v), 8);
    *hi = _mm_srai_epi16(_mm_unpackhi_epi8(v, v), 8);
}

/* sum_s scale[s] * bsums[s] for 16 int16 scales split lo/hi. */
static inline int32_t mx_dot_bsums(__m128i lo, __m128i hi, const int16_t *bsums) {
    return mx_hsum_epi32(_mm_add_epi32(_mm_madd_epi16(lo, mx_load(bsums)),
                                       _mm_madd_epi16(hi, mx_load(bsums + 8))));
}

/* The 4 words as one vector - the compiler keeps them in registers, no memory round trip. */
static inline __m128i mx_words(const uint32_t *w) {
    return _mm_setr_epi32((int)w[0], (int)w[1], (int)w[2], (int)w[3]);
}

/* Q3_K's 16 signed scales (see mx_q3k_scale_words). */
static inline __m128i mx_q3k_scales(const uint8_t *raw) {
    uint32_t w[4];
    mx_q3k_scale_words(raw, w);
    return _mm_sub_epi8(mx_words(w), _mm_set1_epi8(32));
}

/* Q4_K/Q5_K's 8 scales (bytes 0-7) and 8 mins (bytes 8-15) (see mx_k4_scale_min_words). */
static inline __m128i mx_k4_scales_mins(const uint8_t *s) {
    uint32_t w[4];
    mx_k4_scale_min_words(s, w);
    return mx_words(w);
}

float mx_vec_dot_q3_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i four = _mm_set1_epi8(4);
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q3_K_BYTES;
        const int8_t *q8 = y[i].qs;
        __m128i sc[2];
        mx_s8_to_s16(mx_q3k_scales(b + 96), &sc[0], &sc[1]);
        const __m128i hm0 = mx_load(b);
        const __m128i hm1 = mx_load(b + 16);
        __m128i acc = _mm_setzero_si128();
#pragma GCC unroll 2
        for (int n = 0; n < 2; ++n) {
            __m128i q0 = mx_load(b + 32 + n * 32);
            __m128i q1 = mx_load(b + 32 + n * 32 + 16);
#pragma GCC unroll 4
            for (int j = 0; j < 4; ++j) {
                /* u = q2 | hbit << 2 - the unsigned form (value + 4) of mx_unpack_q3_k. */
                const __m128i hbit = _mm_set1_epi8((char)(1 << (4 * n + j)));
                const __m128i u0 = _mm_or_si128(_mm_and_si128(q0, m3), mx_bit_to(hm0, hbit, four));
                const __m128i u1 = _mm_or_si128(_mm_and_si128(q1, m3), mx_bit_to(hm1, hbit, four));
                const int8_t *q = q8 + n * 128 + j * 32;
                /* Sub-blocks n*8 + 2j and +1 = lanes 2j, 2j+1 of sc[n]. */
                acc = mx_acc_lane(acc, _mm_maddubs_epi16(u0, mx_load(q)), sc[n], 2 * j);
                acc = mx_acc_lane(acc, _mm_maddubs_epi16(u1, mx_load(q + 16)), sc[n], 2 * j + 1);
                q0 = _mm_srli_epi16(q0, 2);
                q1 = _mm_srli_epi16(q1, 2);
            }
        }
        const int32_t sumi = mx_hsum_epi32(acc) - 4 * mx_dot_bsums(sc[0], sc[1], y[i].bsums);
        sumf += y[i].d * mx_fp16(b + 108) * (float)sumi;
    }
    return sumf;
}

/* Q4_K/Q5_K share everything but Q5_K's 5th bit (`has_qh`: its qh[32] after the scales). */
static inline float mx_vec_dot_q45_k(const uint8_t *row, const mx_block_q8_k *y, int nb,
                                     size_t block_bytes, int has_qh) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i sixteen = _mm_set1_epi8(16);
    const __m128i zero = _mm_setzero_si128();
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * block_bytes;
        const uint8_t *qs = b + (has_qh ? 48 : 16);
        const int8_t *q8 = y[i].qs;
        const __m128i packed = mx_k4_scales_mins(b + 4);
        const __m128i scales = _mm_unpacklo_epi8(packed, zero);
        const __m128i mins = _mm_unpackhi_epi8(packed, zero);
        const __m128i h0 = has_qh ? mx_load(b + 16) : zero;
        const __m128i h1 = has_qh ? mx_load(b + 32) : zero;
        __m128i acc = _mm_setzero_si128();
#pragma GCC unroll 4
        for (int c = 0; c < 4; ++c) {
            const __m128i v0 = mx_load(qs + c * 32);
            const __m128i v1 = mx_load(qs + c * 32 + 16);
            __m128i lo0 = _mm_and_si128(v0, m4);
            __m128i lo1 = _mm_and_si128(v1, m4);
            __m128i hi0 = _mm_and_si128(_mm_srli_epi16(v0, 4), m4);
            __m128i hi1 = _mm_and_si128(_mm_srli_epi16(v1, 4), m4);
            if (has_qh) {
                /* bit 2c of qh goes on the low nibbles (as 16), bit 2c+1 on the high ones. */
                const __m128i bit_lo = _mm_set1_epi8((char)(1 << (2 * c)));
                const __m128i bit_hi = _mm_set1_epi8((char)(1 << (2 * c + 1)));
                lo0 = _mm_or_si128(lo0, mx_bit_to(h0, bit_lo, sixteen));
                lo1 = _mm_or_si128(lo1, mx_bit_to(h1, bit_lo, sixteen));
                hi0 = _mm_or_si128(hi0, mx_bit_to(h0, bit_hi, sixteen));
                hi1 = _mm_or_si128(hi1, mx_bit_to(h1, bit_hi, sixteen));
            }
            const int8_t *q = q8 + c * 64;
            const __m128i p_lo = _mm_add_epi16(_mm_maddubs_epi16(lo0, mx_load(q)),
                                               _mm_maddubs_epi16(lo1, mx_load(q + 16)));
            const __m128i p_hi = _mm_add_epi16(_mm_maddubs_epi16(hi0, mx_load(q + 32)),
                                               _mm_maddubs_epi16(hi1, mx_load(q + 48)));
            acc = mx_acc_lane(acc, p_lo, scales, 2 * c);
            acc = mx_acc_lane(acc, p_hi, scales, 2 * c + 1);
        }
        /* Each 32-value min covers two 16-value bsums: duplicate it into both int16 lanes. */
        const int32_t summ = mx_dot_bsums(_mm_unpacklo_epi16(mins, mins),
                                          _mm_unpackhi_epi16(mins, mins), y[i].bsums);
        const int32_t sumi = mx_hsum_epi32(acc);
        sumf += y[i].d * (mx_fp16(b) * (float)sumi - mx_fp16(b + 2) * (float)summ);
    }
    return sumf;
}

float mx_vec_dot_q4_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    return mx_vec_dot_q45_k(row, y, nb, MX_Q4_K_BYTES, 0);
}

float mx_vec_dot_q5_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    return mx_vec_dot_q45_k(row, y, nb, MX_Q5_K_BYTES, 1);
}

float mx_vec_dot_q6_k(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const __m128i m30 = _mm_set1_epi8(0x30);
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const uint8_t *b = row + (size_t)i * MX_Q6_K_BYTES;
        __m128i sc[2];
        mx_s8_to_s16(mx_load(b + 192), &sc[0], &sc[1]);
        __m128i acc = _mm_setzero_si128();
#pragma GCC unroll 2
        for (int half = 0; half < 2; ++half) {
            const uint8_t *ql = b + half * 64;
            const uint8_t *qh = b + 128 + half * 32;
            const int8_t *q = y[i].qs + half * 128;
#pragma GCC unroll 2
            for (int k = 0; k < 2; ++k) {
                const __m128i la = mx_load(ql + 16 * k);
                const __m128i lb = mx_load(ql + 32 + 16 * k);
                const __m128i h = mx_load(qh + 16 * k);
                /* The four 32-lane groups of mx_unpack_q6_k, 16 lanes (k) at a time. Each qh
                 * bit pair goes to bits 4-5 with one shift at most (bits 4-5 already are). */
                const __m128i g0 = _mm_or_si128(_mm_and_si128(la, m4),
                                                _mm_and_si128(_mm_slli_epi16(h, 4), m30));
                const __m128i g1 = _mm_or_si128(_mm_and_si128(lb, m4),
                                                _mm_and_si128(_mm_slli_epi16(h, 2), m30));
                const __m128i g2 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(la, 4), m4),
                                                _mm_and_si128(h, m30));
                const __m128i g3 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(lb, 4), m4),
                                                _mm_and_si128(_mm_srli_epi16(h, 2), m30));
                /* Sub-blocks half*8 + k + {0, 2, 4, 6} = lanes k + {0, 2, 4, 6} of sc[half]. */
                acc = mx_acc_lane(acc, _mm_maddubs_epi16(g0, mx_load(q + 16 * k)), sc[half], k);
                acc = mx_acc_lane(acc, _mm_maddubs_epi16(g1, mx_load(q + 32 + 16 * k)), sc[half],
                                  k + 2);
                acc = mx_acc_lane(acc, _mm_maddubs_epi16(g2, mx_load(q + 64 + 16 * k)), sc[half],
                                  k + 4);
                acc = mx_acc_lane(acc, _mm_maddubs_epi16(g3, mx_load(q + 96 + 16 * k)), sc[half],
                                  k + 6);
            }
        }
        const int32_t sumi = mx_hsum_epi32(acc) - 32 * mx_dot_bsums(sc[0], sc[1], y[i].bsums);
        sumf += y[i].d * mx_fp16(b + 208) * (float)sumi;
    }
    return sumf;
}

#endif
