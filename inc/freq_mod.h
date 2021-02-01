#ifndef FREQ_MOD_H
#define FREQ_MOD_H

#include "types.h"
#include "psk_common.h"

void init_fm_cos_lut(const psk_params* restrict params, float* restrict signal_lut);

void modulate_fm(int32_t n_cos_samples,
                 int32_t n_bits,
                 const int32_t* restrict bit_stream,
                 const float* restrict signal_data,
                 float* restrict modulated_signal);

#endif