#ifndef MOD_COMMON_H
#define MOD_COMMON_H
#include "types.h"
#include "consts.h"
#include "math.h"

typedef struct mod_params
{
    float amplitude;
    float freq;
    int32_t cos_factor_idx;
}mod_params;


const int32_t n_cos_samples[] = {90, 180, 360};

int32_t get_n_cos_samples(int32_t factor_idx)
{
    return n_cos_samples[factor_idx];
}

void init_cos_lut(const mod_params* params, float* signal_lut)
{
    const float amplitude = params->amplitude;
    const float freq = params->freq;
    const float samples_factor = get_n_cos_samples(params->cos_factor_idx);

    const float period = 1 / freq;
    const float cos_elem_part = period / samples_factor;


    int32_t sig_idx = 0;
    int32_t half_sig_idx = samples_factor;
    float cos_value;
    for(; sig_idx <= samples_factor; ++sig_idx)
    {
        cos_value = amplitude * cos(2 * PI * freq * sig_idx * cos_elem_part);
        signal_lut[sig_idx] = cos_value;
        signal_lut[half_sig_idx++] = cos_value;
    }
}

#endif
