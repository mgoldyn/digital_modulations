#ifndef DEMODULATE_H
#define DEMODULATE_H

#include "types.h"

void demodulate(char* mod,
                float amp,
                float freq,
                const float* signal_data,
                const float* modulated_signal,
                int32_t n_bits,
                int32_t n_cos_samples,
                int32_t* demodulated_bits);

#endif