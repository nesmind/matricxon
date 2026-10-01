/* Gated DeltaNet recurrence (Qwen3.5 linear attention) - see
 * app/architectures/qwen35_delta_kernels.py for the Numba twin and the exact math.
 * Layouts (float32, contiguous): q/k (T,H,dk), v/out (T,H,dv), g/beta (T,H), state (H,dk,dv),
 * updated in place. Heads are independent over time, so they run in parallel. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mx_common.h"

int mx_gated_delta_rule(const float *q, const float *k, const float *v, const float *g,
                        const float *beta, float *state, float *out, int n_tokens, int n_heads,
                        int dk, int dv, int n_threads) {
    int failed = 0;
#pragma omp parallel num_threads(n_threads)
    {
        float *delta = (float *)malloc(sizeof(float) * (size_t)dv);
        if (delta == NULL) {
#pragma omp atomic write
            failed = 1;
        }
#pragma omp for schedule(static)
        for (int h = 0; h < n_heads; ++h) {
            if (delta == NULL) continue;
            float *s = state + (size_t)h * dk * dv;
            for (int t = 0; t < n_tokens; ++t) {
                size_t qk_off = ((size_t)t * n_heads + h) * dk;
                size_t v_off = ((size_t)t * n_heads + h) * dv;
                const float *kt = k + qk_off, *qt = q + qk_off, *vt = v + v_off;
                float decay = expf(g[(size_t)t * n_heads + h]);
                float b = beta[(size_t)t * n_heads + h];
                memset(delta, 0, sizeof(float) * (size_t)dv);
                /* Decay the state and read what it predicts for k. */
                for (int j = 0; j < dk; ++j) {
                    float *row = s + (size_t)j * dv;
                    float kj = kt[j];
                    for (int i = 0; i < dv; ++i) {
                        row[i] *= decay;
                        delta[i] += row[i] * kj;
                    }
                }
                for (int i = 0; i < dv; ++i) delta[i] = (vt[i] - delta[i]) * b;
                /* Write the error back, then read out with q. */
                float *o = out + v_off;
                memset(o, 0, sizeof(float) * (size_t)dv);
                for (int j = 0; j < dk; ++j) {
                    float *row = s + (size_t)j * dv;
                    float kj = kt[j], qj = qt[j];
                    for (int i = 0; i < dv; ++i) {
                        row[i] += kj * delta[i];
                        o[i] += row[i] * qj;
                    }
                }
            }
        }
        free(delta);
    }
    return failed;
}
