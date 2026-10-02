/* Causal attention for a few query tokens against the KV cache (decode, or a short continuation):
 * for each (head, query) a numerically stable softmax over the visible keys, then the weighted sum
 * of the values - what torch's scaled_dot_product_attention plus a boolean mask does in several
 * dispatches. Numba twin: app/native/fused_ops_numba.py (kept in sync, see CLAUDE.md).
 *
 * q is (n_heads, n_q, dim), k/v are (n_kv_heads, n_kv, dim), each with element strides for the
 * head and token axes (last axis contiguous); out is contiguous (n_heads, n_q, dim). Query i sits
 * at absolute position i + offset and sees keys j <= i + offset (and j > i + offset - window when
 * window > 0). GQA: query head h reads key/value head h / (n_heads / n_kv_heads). */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int mx_attention(const float *q, const float *k, const float *v, float *out, int n_heads,
                 int n_kv_heads, int n_q, int n_kv, int dim, int64_t qs_h, int64_t qs_t,
                 int64_t ks_h, int64_t ks_t, int64_t vs_h, int64_t vs_t, float scale, int offset,
                 int window, int n_threads) {
    int group = n_heads / n_kv_heads;
    int n_pairs = n_heads * n_q;
    int failed = 0;
    /* Fork/join costs more than a short context's work: stay serial below ~256K multiply-adds. */
    int parallel = (int64_t)n_pairs * n_kv * dim > (1 << 18) && n_threads > 1;
#pragma omp parallel num_threads(n_threads) if (parallel)
    {
        float *scores = (float *)malloc(sizeof(float) * (size_t)n_kv);
        if (scores == NULL) {
#pragma omp atomic write
            failed = 1;
        }
#pragma omp for schedule(static)
        for (int pair = 0; pair < n_pairs; ++pair) {
            if (scores == NULL) continue;
            int h = pair / n_q, t = pair % n_q;
            int limit = t + offset + 1;
            if (limit > n_kv) limit = n_kv;
            int start = window > 0 ? t + offset - window + 1 : 0;
            if (start < 0) start = 0;
            const float *qr = q + h * qs_h + t * qs_t;
            const float *kh = k + (h / group) * ks_h, *vh = v + (h / group) * vs_h;
            float peak = -INFINITY;
            for (int j = start; j < limit; ++j) {
                const float *kr = kh + j * ks_t;
                float dot = 0.0f;
#pragma omp simd reduction(+ : dot)
                for (int i = 0; i < dim; ++i) dot += qr[i] * kr[i];
                scores[j] = dot * scale;
                if (scores[j] > peak) peak = scores[j];
            }
            float total = 0.0f;
            for (int j = start; j < limit; ++j) {
                scores[j] = expf(scores[j] - peak);
                total += scores[j];
            }
            float *o = out + ((size_t)h * n_q + t) * dim;
            memset(o, 0, sizeof(float) * (size_t)dim);
            for (int j = start; j < limit; ++j) {
                const float *vr = vh + j * vs_t;
                float p = scores[j] / total;
#pragma omp simd
                for (int i = 0; i < dim; ++i) o[i] += p * vr[i];
            }
        }
        free(scores);
    }
    return failed;
}
