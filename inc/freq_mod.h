#ifndef FREQ_MOD_H
#define FREQ_MOD_H

#include "types.h"
#include "mod_common.h"

void init_fm_cos_lut(const mod_params* params, float* signal_lut);

void modulate_fm(int32_t n_cos_samples,
                 int32_t n_bits,
                 const int32_t*  bit_stream,
                 const float*  signal_data,
                 float*  modulated_signal);

void modulate_fm_cuda(int32_t n_cos_samples,
                      int32_t n_bits,
                      const int32_t*  bit_stream,
                      const float*  signal_data,
                      float*  modulated_signal);

#endif
