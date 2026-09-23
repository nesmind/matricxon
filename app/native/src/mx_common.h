/*
 * matricxon native kernels - shared definitions.
 *
 * A deliberately small, in-house take on llama.cpp's core trick for quantized matmuls (see
 * ROADMAP.md's "In-house native (C) quantized kernels" entry): quantize the activation vector
 * once to 8-bit (Q8_K), then compute integer dot products directly against the packed GGUF
 * weight blocks, reading the raw bytes in place. Block layouts match ggml's own structs and
 * matricxon's already oracle-validated app/gguf/dequant `...Strategy` classes exactly.
 *
 * Plain portable C written so gcc -O3 -march=native auto-vectorizes the inner loops; no
 * hand-written intrinsics yet (a later phase). Little-endian (x86/ARM) is assumed throughout,
 * same as GGUF itself.
 */
#ifndef MX_COMMON_H
#define MX_COMMON_H

#include <stdint.h>
#include <string.h>

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

/* Sums a[i]*b[i] over each of the 16 consecutive 16-value sub-blocks of one 256-value block. */
static inline void mx_sum16(const int8_t *a, const int8_t *b, int32_t *out) {
    for (int s = 0; s < 16; ++s) {
        int32_t acc = 0;
        for (int l = 0; l < 16; ++l) {
            acc += (int16_t)a[16 * s + l] * (int16_t)b[16 * s + l];
        }
        out[s] = acc;
    }
}

void mx_quantize_row_q8_k(const float *x, mx_block_q8_k *y, int n);

/* Each returns the dot product of one packed weight row (nb 256-value blocks) with one
 * quantized activation row. */
float mx_vec_dot_q3_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
float mx_vec_dot_q4_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
float mx_vec_dot_q5_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
float mx_vec_dot_q6_k(const uint8_t *row, const mx_block_q8_k *y, int nb);
float mx_vec_dot_q8_0(const uint8_t *row, const mx_block_q8_k *y, int nb);

#endif
