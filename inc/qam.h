#ifndef QAM_H
#define QAM_H

#include "types.h"

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
