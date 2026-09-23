/*
 * Public entry point: y = x @ W.T for a packed GGUF weight W, called from Python via ctypes
 * (app/native/gemm.py). One function covers both decode (n_tokens == 1, a GEMV) and prefill
 * (n_tokens > 1): every activation row is quantized to Q8_K once up front, then output rows are
 * split across OpenMP threads and each weight row is reused for every token while it's still
 * hot in cache - instead of prefill's old transient full-tensor dequant + float matmul.
 */
#include <stdlib.h>

#include "mx_common.h"

typedef float (*mx_vec_dot_fn)(const uint8_t *row, const mx_block_q8_k *y, int nb);

/* Row stride in bytes and dot kernel for one GGML type; 0/NULL when unsupported. */
static int mx_kernel_for(int ggml_type, int in_features, size_t *row_bytes, mx_vec_dot_fn *fn) {
    const size_t nb = (size_t)in_features / MX_QK_K;
    switch (ggml_type) {
    case MX_TYPE_Q3_K:
        *row_bytes = nb * MX_Q3_K_BYTES;
        *fn = mx_vec_dot_q3_k;
        return 1;
    case MX_TYPE_Q4_K:
        *row_bytes = nb * MX_Q4_K_BYTES;
        *fn = mx_vec_dot_q4_k;
        return 1;
    case MX_TYPE_Q5_K:
        *row_bytes = nb * MX_Q5_K_BYTES;
        *fn = mx_vec_dot_q5_k;
        return 1;
    case MX_TYPE_Q6_K:
        *row_bytes = nb * MX_Q6_K_BYTES;
        *fn = mx_vec_dot_q6_k;
        return 1;
    case MX_TYPE_Q8_0:
        *row_bytes = nb * (MX_QK_K / MX_QK8_0) * MX_Q8_0_BYTES;
        *fn = mx_vec_dot_q8_0;
        return 1;
    default:
        return 0;
    }
}

/* 1 when mx_gemm can handle this (type, in_features) pair - Python checks this once per tensor
 * and keeps the Numba kernel otherwise. */
int mx_supports(int ggml_type, int in_features) {
    size_t row_bytes;
    mx_vec_dot_fn fn;
    return in_features > 0 && in_features % MX_QK_K == 0 &&
           mx_kernel_for(ggml_type, in_features, &row_bytes, &fn);
}

/* w: out_features packed rows; x: (n_tokens, in_features) f32; y: (n_tokens, out_features) f32.
 * Returns 0 on success, -1 unsupported type/shape, -2 out of memory. */
int mx_gemm(int ggml_type, const uint8_t *w, const float *x, float *y, int n_tokens,
            int out_features, int in_features, int n_threads) {
    size_t row_bytes;
    mx_vec_dot_fn dot;
    if (!mx_supports(ggml_type, in_features)) {
        return -1;
    }
    mx_kernel_for(ggml_type, in_features, &row_bytes, &dot);
    const int nb = in_features / MX_QK_K;

    mx_block_q8_k *xq = malloc((size_t)n_tokens * nb * sizeof(mx_block_q8_k));
    if (xq == NULL) {
        return -2;
    }
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_tokens > 1)
    for (int t = 0; t < n_tokens; ++t) {
        mx_quantize_row_q8_k(x + (size_t)t * in_features, xq + (size_t)t * nb, in_features);
    }

#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int o = 0; o < out_features; ++o) {
        const uint8_t *row = w + (size_t)o * row_bytes;
        for (int t = 0; t < n_tokens; ++t) {
            y[(size_t)t * out_features + o] = dot(row, xq + (size_t)t * nb, nb);
        }
    }

    free(xq);
    return 0;
}

/* Build details for matricxon's startup log - which SIMD paths gcc could use for this CPU. */
const char *mx_build_info(void) {
    return "simd="
#if defined(__AVX2__)
           "avx2"
#elif defined(__AVX__)
           "avx"
#elif defined(__SSSE3__)
           "ssse3"
#else
           "none"
#endif
#if defined(__FMA__)
           "+fma"
#endif
#if defined(__F16C__)
           "+f16c"
#endif
#if defined(_OPENMP)
           " openmp=yes"
#else
           " openmp=no"
#endif
        ;
}
