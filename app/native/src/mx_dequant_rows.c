/*
 * Dequantizes specific rows of a packed 2D K-quant tensor (Q3_K/Q4_K/Q5_K/Q6_K) straight to
 * float32 - no activation vector, no dot product, just the real per-element value each weight
 * byte already represents. Used for a real embedding-table lookup
 * (app/architectures/quantized_embedding.py): a forward pass only ever needs a handful of rows
 * (one per input token), so this never has to touch - let alone dequantize - the rest of a real
 * 256K-vocab table.
 *
 * Reuses each type's own existing mx_unpack_* function (mx_unpack_q36k.c/mx_unpack_q45k.c) - the
 * exact same one mx_gemm.c's own prefill path already calls - then reconstructs the real float
 * value per element instead of folding it into a dot product with a quantized activation row.
 * Same per-element math mx_dot_unpacked (mx_common.h) already documents, just never summed:
 * value = d*sc[s]*u - (has_min ? dmin : d)*m[s]. A small, deliberate duplicate of mx_gemm.c's own
 * 4-case K-quant table (row_bytes/unpack/has_min) rather than sharing its private mx_kernel type -
 * mirror any new K-quant case added there here too.
 */
#include <stdlib.h>

#include "mx_common.h"

typedef struct {
    size_t row_bytes;
    mx_unpack_fn unpack;
    int has_min;
} mx_row_kernel;

static int mx_row_kernel_for(int ggml_type, int row_width, mx_row_kernel *k) {
    const size_t nb = (size_t)row_width / MX_QK_K;
    switch (ggml_type) {
    case MX_TYPE_Q3_K:
        k->row_bytes = nb * MX_Q3_K_BYTES;
        k->unpack = mx_unpack_q3_k;
        k->has_min = 0;
        return 1;
    case MX_TYPE_Q4_K:
        k->row_bytes = nb * MX_Q4_K_BYTES;
        k->unpack = mx_unpack_q4_k;
        k->has_min = 1;
        return 1;
    case MX_TYPE_Q5_K:
        k->row_bytes = nb * MX_Q5_K_BYTES;
        k->unpack = mx_unpack_q5_k;
        k->has_min = 1;
        return 1;
    case MX_TYPE_Q6_K:
        k->row_bytes = nb * MX_Q6_K_BYTES;
        k->unpack = mx_unpack_q6_k;
        k->has_min = 0;
        return 1;
    default:
        return 0;
    }
}

/* 1 when mx_dequant_rows can handle this (type, row_width) pair - Python checks this once per
 * tensor and keeps the Python-side per-row dequantize (QuantStrategy.dequantize) otherwise. */
int mx_supports_dequant_rows(int ggml_type, int row_width) {
    mx_row_kernel k;
    if (!mx_row_kernel_for(ggml_type, row_width, &k)) {
        return 0;
    }
    return row_width > 0 && row_width % MX_QK_K == 0;
}

static void mx_dequant_unpacked_to_f32(const mx_block_unpacked *w, int nb, int has_min,
                                       float *out) {
    for (int i = 0; i < nb; ++i) {
        const float min_scale = has_min ? w[i].dmin : w[i].d;
        for (int s = 0; s < MX_QK_K / 16; ++s) {
            const float sc = w[i].d * (float)w[i].sc[s];
            const float mn = min_scale * (float)w[i].m[s];
            for (int l = 0; l < 16; ++l) {
                out[i * MX_QK_K + 16 * s + l] = sc * (float)w[i].u[16 * s + l] - mn;
            }
        }
    }
}

/* w: however many total rows the caller's tensor really has (only row_indices[*] are ever
 * read); row_indices: n_rows indices into it; out: n_rows*row_width contiguous floats, row-major.
 * Returns 0 on success, -1 unsupported type/shape, -2 out of memory. */
int mx_dequant_rows(int ggml_type, const uint8_t *w, const int32_t *row_indices, int n_rows,
                    int row_width, float *out, int n_threads) {
    mx_row_kernel k;
    if (!mx_supports_dequant_rows(ggml_type, row_width)) {
        return -1;
    }
    mx_row_kernel_for(ggml_type, row_width, &k);
    const int nb = row_width / MX_QK_K;

    int status = 0;
#pragma omp parallel num_threads(n_threads)
    {
        mx_block_unpacked *wu = malloc((size_t)nb * sizeof(mx_block_unpacked));
        if (wu == NULL) {
#pragma omp atomic write
            status = -2;
        }
#pragma omp for schedule(static)
        for (int r = 0; r < n_rows; ++r) {
            if (wu == NULL) {
                continue;
            }
            k.unpack(w + (size_t)row_indices[r] * k.row_bytes, wu, nb);
            mx_dequant_unpacked_to_f32(wu, nb, k.has_min, out + (size_t)r * row_width);
        }
        free(wu);
    }
    return status;
}
