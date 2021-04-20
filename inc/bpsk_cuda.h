#ifndef BPSK_CUDA_H
#define BPSK_CUDA_H

#include "types.h"

void modulate_bpsk_cuda(int32_t n_cos_samples,
                   int32_t n_bits,
                   const int32_t*  bit_stream,
                   const float*  signal_data,
                   float*  modulated_signal);
#endif