/* Small per-layer ops (RMSNorm, rotary embedding) that cost more in PyTorch dispatch than arithmetic during a one-token decode. float32, contiguous unless a stride says otherwise. Numba twins: app/native/fused_ops_numba.py (keep in sync); the PyTorch originals: layers.py RMSNorm, rope.py apply_rotary_pos_emb. Single-threaded: OpenMP fork/join would cost more than a few thousand floats. */
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

/* Rotate-half RoPE, x * cos + rotate_half(x) * sin, for x laid out (heads, tokens, dim) with strides s_head / s_tok (last dim contiguous). cos/sin are (tokens, dim); out is contiguous. */
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
