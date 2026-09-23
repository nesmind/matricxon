/*
 * matricxon native kernels - shared definitions.
 *
 * A deliberately small, in-house take on llama.cpp's core trick for quantized matmuls (see
 * ROADMAP.md's "In-house native (C) quantized kernels" entry): quantize the activation vector
 * once to 8-bit (Q8_K), then compute integer dot products directly against the packed GGUF
 * weight blocks, reading the raw bytes in place. Block layouts match ggml's own structs and
 * matricxon's already oracle-validated app/gguf/dequant `...Strategy` classes exactly.
 *
 * K-quant rows are unpacked once per matmul (mx_block_unpacked below) and reused for every
 * token, so a prefill no longer pays the bit-unpacking once per token; a decode step (one token)
 * uses the fused kernels in mx_dot_k_ssse3.c instead.
 *
 * Plain portable C written so gcc -O3 -march=native auto-vectorizes the inner loops, except
 * the block dot products (mx_dot_u8_sc16, mx_dot_k_ssse3.c), which use SSSE3 intrinsics where
 * available - gcc doesn't generate pmaddubsw by itself - with a plain C fallback everywhere
 * else. Little-endian (x86/ARM) is assumed throughout, same as GGUF itself.
 */
#ifndef MX_COMMON_H
#define MX_COMMON_H

#include <stdint.h>
#include <string.h>

#if defined(__SSSE3__)
#include <tmmintrin.h>
#endif

#define MX_QK_K 256
#define MX_QK8_0 32

/* GGML type ids (app/gguf/constants.py GGMLQuantizationType). */
#define MX_TYPE_Q8_0 8
#define MX_TYPE_Q3_K 11
#define MX_TYPE_Q4_K 12
#define MX_TYPE_Q5_K 13
#define MX_TYPE_Q6_K 14

/* On-disk block sizes in bytes, per 256 (K-quants) or 32 (Q8_0) weights. */
#define MX_Q3_K_BYTES 110
#define MX_Q4_K_BYTES 144
#define MX_Q5_K_BYTES 176
#define MX_Q6_K_BYTES 210
#define MX_Q8_0_BYTES 34

/* The quantized activation block: one f32 scale per 256 values, plus 16-value partial sums so
 * a K-quant's per-sub-block "min" term collapses to one multiply per sub-block. */
typedef struct {
    float d;
    int8_t qs[MX_QK_K];
    int16_t bsums[MX_QK_K / 16];
} mx_block_q8_k;

/* f16 -> f32 via a 64K-entry table built once at library load (this project's target CPU,
 * Sandy Bridge, has no F16C instructions). */
extern float mx_fp16_table[1 << 16];

static inline float mx_fp16(const uint8_t *p) {
    uint16_t h;
    memcpy(&h, p, sizeof(h));
    return mx_fp16_table[h];
}

/* Port of kquants_extended.py's _unpack_q3k_scales: 16 signed 6-bit scales packed into 12
 * bytes, reassembled at uint32 granularity (bits carry across bytes before masking). `a` gets
 * them as 16 bytes, still +32 - callers subtract 32 (mx_unpack_q3k_scales here, a vector
 * subtract in mx_dot_k_ssse3.c, which keeps these words in registers). */
static inline void mx_q3k_scale_words(const uint8_t *raw, uint32_t *a) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0F0F0F0Fu;
    uint32_t w[3];
    memcpy(w, raw, 12);
    const uint32_t tmp = w[2];
    a[0] = (w[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    a[1] = (w[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    a[2] = ((w[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    a[3] = ((w[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
}

static inline void mx_unpack_q3k_scales(const uint8_t *raw, int8_t *out) {
    uint32_t a[4];
    mx_q3k_scale_words(raw, a);
    memcpy(out, a, 16);
    for (int s = 0; s < 16; ++s) {
        out[s] = (int8_t)(out[s] - 32);
    }
}

/* Same bit-packing as kquants.py's _get_scale_min_k4, all 8 (scale, min) pairs at once and
 * branch-free at uint32 granularity (the same trick as llama.cpp's utmp/kmask): as bytes, `r`
 * holds the 8 scales then the 8 mins. For j < 4 a pair is the low 6 bits of s[j] / s[j+4]; for
 * j >= 4 it's s[j+4]'s low / high nibble plus the top 2 bits of s[j-4] / s[j]. */
static inline void mx_k4_scale_min_words(const uint8_t *s, uint32_t *r) {
    const uint32_t kmask1 = 0x3F3F3F3Fu;
    const uint32_t kmask2 = 0x0F0F0F0Fu;
    const uint32_t kmask3 = 0x03030303u;
    uint32_t w[3];
    memcpy(w, s, 12);
    r[0] = w[0] & kmask1;
    r[1] = (w[2] & kmask2) | (((w[0] >> 6) & kmask3) << 4);
    r[2] = w[1] & kmask1;
    r[3] = ((w[2] >> 4) & kmask2) | (((w[1] >> 6) & kmask3) << 4);
}

/* One K-quant weight block unpacked for reuse across every token of a prefill (and for the
 * single token of a decode step): values as unsigned bytes `u` plus one integer scale per 16
 * values, so a block's dot product is 16 unsigned x signed byte sums - SSSE3's pmaddubsw does 16
 * byte pairs per instruction (see mx_dot_u8_sc16).
 *
 * `m` means two different things, chosen by the caller's `has_min`:
 * - Q4_K/Q5_K (has_min): the per-16 "min", folded in float as d*sumi - dmin*sum(m*bsums);
 * - Q3_K/Q6_K: the offset that makes their signed values unsigned (u = q + 4 / q + 32) times the
 *   sub-block scale, subtracted from the integer sum - sum(sc*(u-off)*q8) =
 *   sum(sc*u*q8) - sum(sc*off*bsums), so the integer result is exactly the signed one.
 * Bounds: u <= 63 so a pmaddubsw pair stays <= 2*63*127 (no int16 saturation), |sc*off| <=
 * 128*32, and every sum stays far inside int32. */
typedef struct {
    uint8_t u[MX_QK_K];
    int16_t sc[MX_QK_K / 16];
    int16_t m[MX_QK_K / 16];
    float d;
    float dmin;
} mx_block_unpacked;

/* Unpacks nb consecutive blocks of one packed row into out[0..nb). */
typedef void (*mx_unpack_fn)(const uint8_t *restrict row, mx_block_unpacked *restrict out,
                             int nb);

void mx_unpack_q3_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb);
void mx_unpack_q4_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb);
void mx_unpack_q5_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb);
void mx_unpack_q6_k(const uint8_t *restrict row, mx_block_unpacked *restrict out, int nb);

#if defined(__SSSE3__)
static inline int32_t mx_hsum_epi32(__m128i v) {
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2)));
    v = _mm_add_epi32(v, _mm_shuffle_epi32(v, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(v);
}
#endif

/* sum_s sc[s] * sum_l u[16s+l] * q[16s+l] over one 256-value block. */
static inline int32_t mx_dot_u8_sc16(const uint8_t *u, const int8_t *q, const int16_t *sc) {
#if defined(__SSSE3__)
    __m128i acc = _mm_setzero_si128();
    for (int s = 0; s < MX_QK_K / 16; ++s) {
        const __m128i a = _mm_loadu_si128((const __m128i *)(u + 16 * s));
        const __m128i b = _mm_loadu_si128((const __m128i *)(q + 16 * s));
        const __m128i pairs = _mm_maddubs_epi16(a, b);
        acc = _mm_add_epi32(acc, _mm_madd_epi16(pairs, _mm_set1_epi16(sc[s])));
    }
    return mx_hsum_epi32(acc);
#else
    int32_t total = 0;
    for (int s = 0; s < MX_QK_K / 16; ++s) {
        int32_t acc = 0;
        for (int l = 0; l < 16; ++l) {
            acc += (int32_t)u[16 * s + l] * (int32_t)q[16 * s + l];
        }
        total += sc[s] * acc;
    }
    return total;
#endif
}

/* Dot product of one unpacked weight row with one quantized activation row. The float folds
 * keep the exact operation order of the original per-type kernels. */
static inline float mx_dot_unpacked(const mx_block_unpacked *w, const mx_block_q8_k *y, int nb,
                                    int has_min) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        const int32_t sumi = mx_dot_u8_sc16(w[i].u, y[i].qs, w[i].sc);
        int32_t summ = 0;
        for (int s = 0; s < MX_QK_K / 16; ++s) {
            summ += (int32_t)w[i].m[s] * (int32_t)y[i].bsums[s];
        }
        if (has_min) {
            sumf += y[i].d * (w[i].d * (float)sumi - w[i].dmin * (float)summ);
        } else {
            sumf += y[i].d * w[i].d * (float)(sumi - summ);
        }
    }
    return sumf;
}

void mx_quantize_row_q8_k(const float *x, mx_block_q8_k *y, int n);

/* Q8_0 has a float scale per 32 values, so it keeps a direct per-token dot product. */
float mx_vec_dot_q8_0(const uint8_t *row, const mx_block_q8_k *y, int nb);

#if defined(__SSSE3__)
/* Fused decode kernels (mx_dot_k_ssse3.c): one packed weight row times one activation row. */
float mx_vec_dot_q3_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
float mx_vec_dot_q4_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
float mx_vec_dot_q5_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
float mx_vec_dot_q6_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
#endif

#endif
