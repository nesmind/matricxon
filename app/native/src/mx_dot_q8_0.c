/* Integer dot product for Q8_0 weight rows against a Q8_K activation row. Q8_0 has a float
 * scale per 32 values (not an integer sub-block scale), so it doesn't fit mx_block_unpacked and
 * stays a direct per-token kernel - its values are already plain int8, nothing to unpack. */
#include "mx_common.h"

/* Q8_0 block: {f16 d; i8 qs[32]} - 34 bytes, eight per 256-value activation block. */
float mx_vec_dot_q8_0(const uint8_t *row, const mx_block_q8_k *y, int nb) {
    float sumf = 0.0f;
    for (int i = 0; i < nb; ++i) {
        float block_sum = 0.0f;
        for (int k = 0; k < MX_QK_K / MX_QK8_0; ++k) {
            const uint8_t *b = row + (size_t)(i * 8 + k) * MX_Q8_0_BYTES;
            const int8_t *qs = (const int8_t *)(b + 2);
            const int8_t *q8 = y[i].qs + k * MX_QK8_0;
            int32_t acc = 0;
            for (int l = 0; l < MX_QK8_0; ++l) {
                acc += (int16_t)qs[l] * (int16_t)q8[l];
            }
            block_sum += mx_fp16(b) * (float)acc;
        }
        sumf += y[i].d * block_sum;
    }
    return sumf;
}
