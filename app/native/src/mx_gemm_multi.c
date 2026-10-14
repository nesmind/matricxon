/* Several weights that share one input (q/k/v, or gate/up) in one call: the input is quantized to Q8_K once and every output row of every weight is split across threads in one parallel region. Same per-row kernels as mx_gemm, so results are identical. Only for calls that use per-row kernels (decode, small batches); returns -1 otherwise and Python calls mx_gemm per weight. */
#include <stdlib.h>

#include "mx_gemm.h"

#define MX_MULTI_MAX 4

/* ws[i]: packed rows of weight i (outs[i] rows); x: (n_tokens, in_features); y: (n_tokens, sum outs), weight i's columns after weight i-1's. Returns 0, -1 unsupported, -2 out of memory. */
int mx_gemm_multi(int n, const int *types, const uint8_t *const *ws, const int *outs, const float *x,
                  float *y, int n_tokens, int in_features, int n_threads) {
    if (n < 1 || n > MX_MULTI_MAX) {
        return -1;
    }
    mx_kernel k[MX_MULTI_MAX];
    mx_vec_dot_fn dot[MX_MULTI_MAX];
    int start[MX_MULTI_MAX + 1];
    int need_q8 = 0;
    start[0] = 0;
    for (int i = 0; i < n; ++i) {
        if (!mx_supports(types[i], in_features)) {
            return -1;
        }
        mx_kernel_for(types[i], in_features, &k[i]);
        dot[i] = k[i].dot_f32 != NULL ? NULL : mx_row_dot(&k[i], n_tokens);
        if (k[i].dot_f32 == NULL && dot[i] == NULL) {
            return -1;
        }
        need_q8 |= dot[i] != NULL;
        start[i + 1] = start[i] + outs[i];
    }
    const int total = start[n];
    const int nb = in_features / MX_QK_K;

    mx_block_q8_k *xq = NULL;
    if (need_q8) {
        xq = malloc((size_t)n_tokens * nb * sizeof(mx_block_q8_k));
        if (xq == NULL) {
            return -2;
        }
        for (int t = 0; t < n_tokens; ++t) {
            mx_quantize_row_q8_k(x + (size_t)t * in_features, xq + (size_t)t * nb, in_features);
        }
    }

#pragma omp parallel for schedule(static) num_threads(n_threads)
    for (int r = 0; r < total; ++r) {
        int i = 0;
        while (r >= start[i + 1]) {
            ++i;
        }
        const int o = r - start[i];
        const uint8_t *row = ws[i] + (size_t)o * k[i].row_bytes;
        for (int t = 0; t < n_tokens; ++t) {
            y[(size_t)t * total + r] =
                dot[i] != NULL
                    ? dot[i](row, xq + (size_t)t * nb, nb)
                    : k[i].dot_f32(row, x + (size_t)t * in_features, in_features / k[i].block_size);
        }
    }
    free(xq);
    return 0;
}
