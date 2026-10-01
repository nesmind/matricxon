/*
 * Public entry point: y = x @ W.T for a packed GGUF weight W, called from Python via ctypes
 * (app/native/gemm.py). One function covers both decode (n_tokens == 1, a GEMV) and prefill
 * (n_tokens > 1): every activation row is quantized to Q8_K once up front, then output rows are
 * split across OpenMP threads. For K-quants each weight row is unpacked once (mx_block_unpacked)
 * and that unpacked row is reused for every token while it's hot in cache - a 40-token prefill
 * used to re-unpack every row 40 times, which made it cost 40 decode steps.
 */
#include <stdlib.h>

#include "mx_common.h"

typedef float (*mx_vec_dot_fn)(const uint8_t *row, const mx_block_q8_k *y, int nb);

/* Up to this many tokens, a K-quant uses its fused per-token kernel (`fused`) instead of
 * unpacking each row once: unpacking only pays off once it's reused across tokens. Measured on
 * this project's i7-2640M (Qwen3.5-4B Q4_K/Q6_K tensors, 2 threads): fused is faster for 2-8
 * tokens (e.g. 4 tokens 4.5ms vs 6.0ms) and level at 16 - so a small batched decode step (a
 * few users' tokens in one pass) uses the fused kernel, not the unpack path built for prompts.
 * Per-token cost there is ~constant (1.1ms for a 9216x2560 Q4_K), i.e. compute-bound, so a batch
 * doesn't get cheaper per token on that CPU the way it does on a bandwidth-bound one. */
#ifndef MX_FUSED_MAX_TOKENS
#define MX_FUSED_MAX_TOKENS 8
#endif

/* Runtime copy of the threshold, so a benchmark can find the real crossover (see
 * mx_set_fused_max_tokens and scripts/benchmark_kernels.py) without a rebuild. */
static int mx_fused_max_tokens = MX_FUSED_MAX_TOKENS;

void mx_set_fused_max_tokens(int n_tokens) {
    mx_fused_max_tokens = n_tokens;
}

/* How one GGML type is computed: K-quants set `unpack` (and `has_min` for Q4_K/Q5_K), plus
 * `fused` where SSSE3 is available; Q8_0 sets only `dot`. Every I-quant/T-quant (IQ4_NL/XS,
 * TQ1_0/TQ2_0, the IQ1/IQ2/IQ3 grid family) sets only `dot_f32` (see mx_common.h) -
 * `block_size` defaults to MX_QK_K (every K-quant/Q8_0 case below overrides it to whatever its
 * own real block size actually is; IQ4_NL's real block is 32, not 256). */
typedef struct {
    size_t row_bytes;
    int block_size;
    mx_unpack_fn unpack;
    int has_min;
    mx_vec_dot_fn fused;
    mx_vec_dot_fn dot;
    mx_vec_dot_f32_fn dot_f32;
} mx_kernel;

#if defined(__SSSE3__)
#define MX_FUSED(fn) (fn)
#else
#define MX_FUSED(fn) NULL
#endif

/* Fills `k` for one GGML type; 0 when unsupported. */
static int mx_kernel_for(int ggml_type, int in_features, mx_kernel *k) {
    k->block_size = MX_QK_K;
    k->unpack = NULL;
    k->has_min = 0;
    k->fused = NULL;
    k->dot = NULL;
    k->dot_f32 = NULL;
    const size_t nb = (size_t)in_features / MX_QK_K;
    switch (ggml_type) {
    case MX_TYPE_Q3_K:
        k->row_bytes = nb * MX_Q3_K_BYTES;
        k->unpack = mx_unpack_q3_k;
        k->fused = MX_FUSED(mx_vec_dot_q3_k);
        return 1;
    case MX_TYPE_Q4_K:
        k->row_bytes = nb * MX_Q4_K_BYTES;
        k->unpack = mx_unpack_q4_k;
        k->has_min = 1;
        k->fused = MX_FUSED(mx_vec_dot_q4_k);
        return 1;
    case MX_TYPE_Q5_K:
        k->row_bytes = nb * MX_Q5_K_BYTES;
        k->unpack = mx_unpack_q5_k;
        k->has_min = 1;
        k->fused = MX_FUSED(mx_vec_dot_q5_k);
        return 1;
    case MX_TYPE_Q6_K:
        k->row_bytes = nb * MX_Q6_K_BYTES;
        k->unpack = mx_unpack_q6_k;
        k->fused = MX_FUSED(mx_vec_dot_q6_k);
        return 1;
    case MX_TYPE_Q8_0:
        k->row_bytes = nb * (MX_QK_K / MX_QK8_0) * MX_Q8_0_BYTES;
        k->dot = mx_vec_dot_q8_0;
        return 1;
    case MX_TYPE_IQ4_NL:
        k->block_size = MX_IQ4_NL_BLOCK;
        k->row_bytes = (size_t)(in_features / MX_IQ4_NL_BLOCK) * MX_IQ4_NL_BYTES;
        k->dot_f32 = mx_vec_dot_iq4_nl_f32;
        return 1;
    case MX_TYPE_IQ4_XS:
        k->row_bytes = nb * MX_IQ4_XS_BYTES;
        k->dot_f32 = mx_vec_dot_iq4_xs_f32;
        return 1;
    case MX_TYPE_TQ1_0:
        k->row_bytes = nb * MX_TQ1_0_BYTES;
        k->dot_f32 = mx_vec_dot_tq1_0_f32;
        return 1;
    case MX_TYPE_TQ2_0:
        k->row_bytes = nb * MX_TQ2_0_BYTES;
        k->dot_f32 = mx_vec_dot_tq2_0_f32;
        return 1;
    case MX_TYPE_IQ2_XXS:
        k->row_bytes = nb * MX_IQ2_XXS_BYTES;
        k->dot_f32 = mx_vec_dot_iq2_xxs_f32;
        return 1;
    case MX_TYPE_IQ2_XS:
        k->row_bytes = nb * MX_IQ2_XS_BYTES;
        k->dot_f32 = mx_vec_dot_iq2_xs_f32;
        return 1;
    case MX_TYPE_IQ2_S:
        k->row_bytes = nb * MX_IQ2_S_BYTES;
        k->dot_f32 = mx_vec_dot_iq2_s_f32;
        return 1;
    case MX_TYPE_IQ3_XXS:
        k->row_bytes = nb * MX_IQ3_XXS_BYTES;
        k->dot_f32 = mx_vec_dot_iq3_xxs_f32;
        return 1;
    case MX_TYPE_IQ3_S:
        k->row_bytes = nb * MX_IQ3_S_BYTES;
        k->dot_f32 = mx_vec_dot_iq3_s_f32;
        return 1;
    case MX_TYPE_IQ1_S:
        k->row_bytes = nb * MX_IQ1_S_BYTES;
        k->dot_f32 = mx_vec_dot_iq1_s_f32;
        return 1;
    case MX_TYPE_IQ1_M:
        k->row_bytes = nb * MX_IQ1_M_BYTES;
        k->dot_f32 = mx_vec_dot_iq1_m_f32;
        return 1;
    default:
        return 0;
    }
}

/* 1 when mx_gemm can handle this (type, in_features) pair - Python checks this once per tensor
 * and keeps the Numba kernel otherwise. */
int mx_supports(int ggml_type, int in_features) {
    mx_kernel k;
    if (!mx_kernel_for(ggml_type, in_features, &k)) {
        return 0;
    }
    return in_features > 0 && in_features % k.block_size == 0;
}

/* w: out_features packed rows; x: (n_tokens, in_features) f32; y: (n_tokens, out_features) f32.
 * Returns 0 on success, -1 unsupported type/shape, -2 out of memory. */
int mx_gemm(int ggml_type, const uint8_t *w, const float *x, float *y, int n_tokens,
            int out_features, int in_features, int n_threads) {
    mx_kernel k;
    if (!mx_supports(ggml_type, in_features)) {
        return -1;
    }
    mx_kernel_for(ggml_type, in_features, &k);

    /* IQ4_NL/IQ4_XS/TQ1_0/TQ2_0: no mx_block_q8_k involved at all - dot_f32 reads the raw
     * activation row directly (see mx_common.h's own comment on why). */
    if (k.dot_f32 != NULL) {
        const int nb_f32 = in_features / k.block_size;
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int o = 0; o < out_features; ++o) {
            const uint8_t *row = w + (size_t)o * k.row_bytes;
            for (int t = 0; t < n_tokens; ++t) {
                y[(size_t)t * out_features + o] =
                    k.dot_f32(row, x + (size_t)t * in_features, nb_f32);
            }
        }
        return 0;
    }

    const int nb = in_features / MX_QK_K;

    mx_block_q8_k *xq = malloc((size_t)n_tokens * nb * sizeof(mx_block_q8_k));
    if (xq == NULL) {
        return -2;
    }
#pragma omp parallel for schedule(static) num_threads(n_threads) if (n_tokens > 1)
    for (int t = 0; t < n_tokens; ++t) {
        mx_quantize_row_q8_k(x + (size_t)t * in_features, xq + (size_t)t * nb, in_features);
    }

    int status = 0;
    const mx_vec_dot_fn dot =
        k.dot != NULL ? k.dot : (n_tokens <= mx_fused_max_tokens ? k.fused : NULL);
    if (dot != NULL) {
#pragma omp parallel for schedule(static) num_threads(n_threads)
        for (int o = 0; o < out_features; ++o) {
            const uint8_t *row = w + (size_t)o * k.row_bytes;
            for (int t = 0; t < n_tokens; ++t) {
                y[(size_t)t * out_features + o] = dot(row, xq + (size_t)t * nb, nb);
            }
        }
    } else {
#pragma omp parallel num_threads(n_threads)
        {
            /* One unpacked row per thread: nb * 328 bytes, e.g. 10 KB for in_features 8192. */
            mx_block_unpacked *wu = malloc((size_t)nb * sizeof(mx_block_unpacked));
            if (wu == NULL) {
#pragma omp atomic write
                status = -2;
            }
#pragma omp for schedule(static)
            for (int o = 0; o < out_features; ++o) {
                if (wu == NULL) {
                    continue;
                }
                k.unpack(w + (size_t)o * k.row_bytes, wu, nb);
                for (int t = 0; t < n_tokens; ++t) {
                    y[(size_t)t * out_features + o] =
                        mx_dot_unpacked(wu, xq + (size_t)t * nb, nb, k.has_min);
                }
            }
            free(wu);
        }
    }

    free(xq);
    return status;
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
