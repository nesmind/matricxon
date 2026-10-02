/* One fused pass for the sampler (app/runtime/sampler.py): repetition penalty, temperature, top-k,
 * top-p, softmax and the draw - replacing a chain of full-vocabulary torch ops (a sort and three
 * softmaxes over 128K-250K floats, 18-33 ms per token on the dev laptop) with O(vocab) work.
 * Numba twin: app/native/fused_ops_numba.py (kept in sync, see CLAUDE.md).
 *
 * Same maths as the torch pipeline: top-k keeps every logit >= the k-th largest (ties included),
 * top-p keeps the shortest descending prefix whose probability reaches p (always the best token),
 * the draw is inverse-CDF with the caller's uniform `u` in [0, 1). Requires top_k > 0 or
 * top_p >= 1 (top-p over the whole vocabulary needs a full sort - the caller keeps torch for it).
 * Returns the token id, or -1 if memory ran out. */
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    float value;
    int index;
} Candidate;

static int by_value_desc(const void *a, const void *b) {
    float x = ((const Candidate *)a)->value, y = ((const Candidate *)b)->value;
    return (x < y) - (x > y);
}

/* Smallest of the k largest values: a size-k min-heap over one pass. */
static float kth_largest(const float *values, int n, int k, float *heap) {
    int size = 0;
    for (int i = 0; i < n; ++i) {
        float x = values[i];
        if (size < k) {
            int c = size++;
            while (c > 0 && heap[(c - 1) / 2] > x) { heap[c] = heap[(c - 1) / 2]; c = (c - 1) / 2; }
            heap[c] = x;
        } else if (x > heap[0]) {
            int c = 0;
            for (;;) {
                int l = 2 * c + 1, r = l + 1, m = c;
                float lowest = x;
                if (l < size && heap[l] < lowest) { m = l; lowest = heap[l]; }
                if (r < size && heap[r] < lowest) { m = r; }
                if (m == c) break;
                heap[c] = heap[m];
                c = m;
            }
            heap[c] = x;
        }
    }
    return heap[0];
}

int mx_sample(const float *logits, int n_vocab, const int *penalty_ids, int n_penalty,
              float penalty, float temperature, int top_k, float top_p, float u) {
    float *work = (float *)malloc(sizeof(float) * (size_t)n_vocab);
    if (work == NULL) return -1;
    memcpy(work, logits, sizeof(float) * (size_t)n_vocab);
    for (int i = 0; i < n_penalty; ++i) {
        float x = work[penalty_ids[i]];
        work[penalty_ids[i]] = x > 0 ? x / penalty : x * penalty;
    }
    for (int i = 0; i < n_vocab; ++i) work[i] /= temperature;

    float floor_value = -INFINITY;
    if (top_k > 0 && top_k < n_vocab) {
        float *heap = (float *)malloc(sizeof(float) * (size_t)top_k);
        if (heap == NULL) { free(work); return -1; }
        floor_value = kth_largest(work, n_vocab, top_k, heap);
        free(heap);
    }
    int n_cand = 0;
    for (int i = 0; i < n_vocab; ++i) n_cand += work[i] >= floor_value;
    Candidate *cand = (Candidate *)malloc(sizeof(Candidate) * (size_t)n_cand);
    if (cand == NULL) { free(work); return -1; }
    float peak = -INFINITY;
    for (int i = 0, c = 0; i < n_vocab; ++i) {
        if (work[i] >= floor_value) {
            cand[c].value = work[i];
            cand[c++].index = i;
            if (work[i] > peak) peak = work[i];
        }
    }
    free(work);

    double total = 0.0;
    for (int c = 0; c < n_cand; ++c) {
        cand[c].value = expf(cand[c].value - peak);
        total += cand[c].value;
    }
    if (top_p < 1.0f) {
        qsort(cand, (size_t)n_cand, sizeof(Candidate), by_value_desc);
        double before = 0.0, kept = 0.0;
        int keep = 0;
        while (keep < n_cand && before / total <= top_p) {
            before += cand[keep].value;
            kept = before;
            ++keep;
        }
        n_cand = keep;
        total = kept;
    }
    double target = (double)u * total, running = 0.0;
    int token = cand[n_cand - 1].index;
    for (int c = 0; c < n_cand; ++c) {
        running += cand[c].value;
        if (running > target) { token = cand[c].index; break; }
    }
    free(cand);
    return token;
}
