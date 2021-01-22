#ifndef PSK_COMMON_H
#define PSK_COMMON_H
#include "types.h"

typedef struct psk_params
{
    float amplitude;
    float freq;
    int32_t cos_factor_idx; 
}psk_params;

const int32_t get_n_cos_samples(int32_t factor_idx);
void init_psk_cos_lut(const psk_params* restrict params, float* restrict signal_lut);

#endif