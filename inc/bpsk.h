#ifndef BPSK_H
#define BPSK_H

#include "types.h"

void init_bpsk_signal_lut(float* restrict signal_lut);

void modulate_bpsk(int32_t n_bits,
                   const int32_t* restrict bit_stream,
                   const float* restrict signal_data,
                   float* restrict modulated_signal);
#endif