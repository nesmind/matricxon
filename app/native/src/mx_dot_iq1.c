/*
 * IQ1_S, IQ1_M - ported directly from their own already-oracle-validated Numba kernels
 * (app/gguf/dequant/quantized_gemv_iq1.py). No per-lane sign flip like IQ2/IQ3: each grid value
 * gets one shared additive delta per group instead. IQ1_M has no `d` field at all - its scale is
 * a synthetic fp16 word reassembled from 4 fragments of `sc` (see mx_fp16 reuse below).
 */
#include "mx_common.h"
#include "mx_iq_grids.h"

#define MX_IQ1_DELTA 0.125f

float mx_vec_dot_iq1_s_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ1_S_BYTES;
        const float d = mx_fp16(blk);
        const uint8_t *qs = blk + 2;
        const uint8_t *qh_bytes = blk + 34;
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int ib = 0; ib < 8; ++ib) {
            uint16_t qh;
            memcpy(&qh, qh_bytes + ib * 2, sizeof(qh));
            const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
            const float delta = (qh & 0x8000) ? -MX_IQ1_DELTA : MX_IQ1_DELTA;
            const float *base = xb + ib * 32;
            for (int group = 0; group < 4; ++group) {
                const int idx = qs[ib * 4 + group] | (((qh >> (3 * group)) & 7) << 8);
                const int8_t *grid = MX_IQ1S_GRID + (size_t)idx * 8;
                const float *lane = base + group * 8;
                for (int j = 0; j < 8; ++j) {
                    acc += lane[j] * (dl * ((float)grid[j] + delta));
                }
            }
        }
    }
    return acc;
}

float mx_vec_dot_iq1_m_f32(const uint8_t *row, const float *x, int nb) {
    float acc = 0.0f;
    for (int sb = 0; sb < nb; ++sb) {
        const uint8_t *blk = row + (size_t)sb * MX_IQ1_M_BYTES;
        const uint8_t *qs = blk;
        const uint8_t *qh = blk + 32;
        const uint8_t *sc_bytes = blk + 48;
        uint16_t sc[4];
        memcpy(sc, sc_bytes, sizeof(sc));
        const uint16_t scale_u16 = (uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) |
                                              ((sc[2] >> 4) & 0xF00) | (sc[3] & 0xF000));
        uint8_t scale_bytes[2];
        memcpy(scale_bytes, &scale_u16, sizeof(scale_u16));
        const float d = mx_fp16(scale_bytes);
        const float *xb = x + (size_t)sb * MX_QK_K;
        for (int ib = 0; ib < 8; ++ib) {
            const int i2 = ib / 2;
            const int parity = ib % 2;
            const uint16_t sc_i2 = sc[i2];
            const float d1 = d * (float)(2 * ((sc_i2 >> (6 * parity)) & 7) + 1);
            const float d2 = d * (float)(2 * ((sc_i2 >> (6 * parity + 3)) & 7) + 1);
            const int qh0 = qh[ib * 2];
            const int qh1 = qh[ib * 2 + 1];
            const int idx0 = qs[ib * 4 + 0] | ((qh0 << 8) & 0x700);
            const int idx1 = qs[ib * 4 + 1] | ((qh0 << 4) & 0x700);
            const int idx2 = qs[ib * 4 + 2] | ((qh1 << 8) & 0x700);
            const int idx3 = qs[ib * 4 + 3] | ((qh1 << 4) & 0x700);
            const float delta0 = (qh0 & 0x08) ? -MX_IQ1_DELTA : MX_IQ1_DELTA;
            const float delta1 = (qh0 & 0x80) ? -MX_IQ1_DELTA : MX_IQ1_DELTA;
            const float delta2 = (qh1 & 0x08) ? -MX_IQ1_DELTA : MX_IQ1_DELTA;
            const float delta3 = (qh1 & 0x80) ? -MX_IQ1_DELTA : MX_IQ1_DELTA;
            const int8_t *g0 = MX_IQ1S_GRID + (size_t)idx0 * 8;
            const int8_t *g1 = MX_IQ1S_GRID + (size_t)idx1 * 8;
            const int8_t *g2 = MX_IQ1S_GRID + (size_t)idx2 * 8;
            const int8_t *g3 = MX_IQ1S_GRID + (size_t)idx3 * 8;
            const float *base = xb + ib * 32;
            for (int j = 0; j < 8; ++j) {
                acc += base[j] * (d1 * ((float)g0[j] + delta0));
            }
            for (int j = 0; j < 8; ++j) {
                acc += base[8 + j] * (d1 * ((float)g1[j] + delta1));
            }
            for (int j = 0; j < 8; ++j) {
                acc += base[16 + j] * (d2 * ((float)g2[j] + delta2));
            }
            for (int j = 0; j < 8; ++j) {
                acc += base[24 + j] * (d2 * ((float)g3[j] + delta3));
            }
        }
    }
    return acc;
}
