/* Per-type kernel table shared by mx_gemm.c and mx_gemm_multi.c. */
#ifndef MX_GEMM_H
#define MX_GEMM_H

#include "mx_common.h"

typedef float (*mx_vec_dot_fn)(const uint8_t *row, const mx_block_q8_k *y, int nb);

/* How one GGML type is computed: K-quants set `unpack` (and `has_min` for Q4_K/Q5_K), plus `fused` with SSSE3; Q8_0 sets only `dot`; I-/T-quants set only `dot_f32` (see mx_common.h). `block_size` defaults to MX_QK_K; each case overrides it with its real size (IQ4_NL: 32). */
typedef struct {
    size_t row_bytes;
    int block_size;
    mx_unpack_fn unpack;
    int has_min;
    mx_vec_dot_fn fused;
    mx_vec_dot_fn dot;
    mx_vec_dot_f32_fn dot_f32;
} mx_kernel;

int mx_kernel_for(int ggml_type, int in_features, mx_kernel *k);
int mx_supports(int ggml_type, int in_features);
/* The per-row kernel for `n_tokens` tokens, or NULL when the row must be unpacked (or uses dot_f32). */
mx_vec_dot_fn mx_row_dot(const mx_kernel *k, int n_tokens);

#endif
