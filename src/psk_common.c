#include "../inc/psk_common.h"
#include "../inc/consts.h"
#include "math.h"

const int32_t n_cos_samples[] = {90, 180, 360};

const int32_t get_n_cos_samples(int32_t factor_idx)
{
    return n_cos_samples[factor_idx];
}

void init_psk_cos_lut(const psk_params* restrict params, float* restrict signal_lut)
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

void set_phase_shift(int32_t n_cos_samples, int32_t phase_shift, const float* restrict signal_data, float* restrict modulated_signal)
{
    const int32_t scaled_phase_shift = (int32_t)((float)phase_shift * ((float)n_cos_samples)/ N_MAX_DEGREE);

    int32_t sig_idx = 0;
    int32_t sig_lut_idx = 0;
    for(; sig_idx < n_cos_samples; ++sig_idx)
    {
        modulated_signal[sig_idx] = signal_data[scaled_phase_shift + sig_idx];
    }
}