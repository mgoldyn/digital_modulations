#ifndef QPSK_H
#define QPSK_H

#include "types.h"

void init_qpsk_signal_lut(float* restrict signal_lut);

void modulate_qpsk(int32_t n_bits,
                   const int32_t* restrict bit_stream,
                   const float* restrict signal_data,
                   float* restrict modulated_signal);
#endif
