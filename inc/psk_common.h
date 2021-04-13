#ifndef PSK_COMMON_H
#define PSK_COMMON_H
#include "types.h"

typedef struct psk_params
{
    float amplitude;
    float freq;
    int32_t cos_factor_idx; 
}psk_params;

int32_t get_n_cos_samples(int32_t factor_idx);

void init_psk_cos_lut(const psk_params*  params, float*  signal_lut);

void set_phase_shift(int32_t n_cos_samples, int32_t phase_shift, const float*  signal_data, float*  modulated_signal);

#endif