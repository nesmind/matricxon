/* Small per-layer ops that cost far more in PyTorch's per-op dispatch than in arithmetic during a
 * one-token decode step: RMSNorm and rotary embedding. float32, contiguous unless a stride says
 * otherwise. Numba twins: app/native/fused_ops_numba.py (kept in sync, see CLAUDE.md); the exact
 * PyTorch formulas they replace: app/architectures/layers.py RMSNorm, rope.py apply_rotary_pos_emb.
 * Single-threaded on purpose - a few thousand floats is far below what an OpenMP fork/join costs. */
#include <math.h>
#include <stddef.h>
#include <stdint.h>

/* out[r] = x[r] * rsqrt(mean(x[r]^2) + eps) * w, per row of `dim`. */
void mx_rms_norm(const float *x, const float *w, float *out, int rows, int dim, float eps) {
    for (int r = 0; r < rows; ++r) {
        const float *xr = x + (size_t)r * dim;
        float *o = out + (size_t)r * dim;
        float sum = 0.0f;
#pragma omp simd reduction(+ : sum)
        for (int i = 0; i < dim; ++i) sum += xr[i] * xr[i];
        float scale = 1.0f / sqrtf(sum / (float)dim + eps);
        for (int i = 0; i < dim; ++i) o[i] = xr[i] * scale * w[i];
    }
}

/* Rotate-half RoPE, x * cos + rotate_half(x) * sin, for x laid out (heads, tokens, dim) with
 * element strides s_head / s_tok (the last dim contiguous - a q/k projection viewed and
 * transposed). cos/sin are (tokens, dim); out is contiguous (heads, tokens, dim). */
void mx_rope(const float *x, float *out, const float *cos, const float *sin, int n_heads,
             int n_tokens, int dim, int64_t s_head, int64_t s_tok) {
    int half = dim / 2;
    for (int h = 0; h < n_heads; ++h) {
        for (int t = 0; t < n_tokens; ++t) {
            const float *xr = x + h * s_head + t * s_tok;
            const float *c = cos + (size_t)t * dim, *s = sin + (size_t)t * dim;
            float *o = out + ((size_t)h * n_tokens + t) * dim;
            for (int i = 0; i < half; ++i) {
                o[i] = xr[i] * c[i] - xr[i + half] * s[i];
                o[i + half] = xr[i + half] * c[i + half] + xr[i] * s[i + half];
            }
        }
    }
}
