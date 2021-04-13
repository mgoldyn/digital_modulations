#ifndef QPSK_H
#define QPSK_H

#include "types.h"

CUDA_DELLEXPORT
void modulate_qpsk(int32_t n_cos_samples,
                   int32_t n_bits,
                   const int32_t*  bit_stream,
                   const float*  signal_data,
                   float*  modulated_signal);
#endif
