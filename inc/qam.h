#ifndef QAM_H
#define QAM_H

#include "types.h"

#define _16QAM_AMP_00 -0.75f
#define _16QAM_AMP_01 -0.25f
#define _16QAM_AMP_10  0.25f
#define _16QAM_AMP_11  0.75f

#define _16QAM_PHASE_00 315
#define _16QAM_PHASE_01  45
#define _16QAM_PHASE_10 225
#define _16QAM_PHASE_11 135

void modulate_16qam(int32_t n_cos_samples,
                    int32_t n_bits,
                    const int32_t* bit_stream,
                    const float* signal_data,
                    float*  modulated_signal);

void modulate_16qam_cuda(int32_t n_cos_samples,
                         int32_t n_bits,
                         const int32_t* bit_stream,
                         const float* signal_data,
                         float*  modulated_signal);

#endif
