#include "../inc/freq_mod.h"
#include "../inc/consts.h"

#include "math.h"

#define FM_FREQ_0 0
#define FM_FREQ_1 1

void init_fm_cos_lut(const psk_params* restrict params, float* restrict signal_lut)
{
    const float amplitude = params->amplitude;
    const float freq = params->freq;
    const float samples_factor = get_n_cos_samples(params->cos_factor_idx);

    const float period = 1 / freq;
    const float cos_elem_part = period / samples_factor;

    const float freq_0 = freq * 2;
    const float period_0 = 1 / freq_0;
    const float cos_elem_part_0 = period_0 / samples_factor;
    
    int32_t sig_idx = 0;
    for(; sig_idx <= samples_factor; ++sig_idx)
    {
        signal_lut[sig_idx] = amplitude * cos(2 * PI * freq_0 * sig_idx * cos_elem_part_0);
        printf("c0[%d] = %f\n", sig_idx, signal_lut[sig_idx]);
        
    }
    for(; sig_idx <= samples_factor * 2; ++sig_idx)
    {
        signal_lut[sig_idx] = amplitude * cos(2 * PI * freq * sig_idx * cos_elem_part);
        printf("c1[%d] = %f\n", sig_idx, signal_lut[sig_idx]);
    }

    
}

static void set_freq(int32_t n_cos_samples, int32_t freq_sig_idx, const float* restrict signal_data, float* restrict modulated_signal)
{
    int32_t sig_idx = 0;
    const int32_t car_signal_offset = n_cos_samples * freq_sig_idx;
    for(; sig_idx < n_cos_samples; ++sig_idx)
    {
        modulated_signal[sig_idx] = signal_data[car_signal_offset + sig_idx];
    }
}

void modulate_fm(int32_t n_cos_samples,
                 int32_t n_bits,
                 const int32_t* restrict bit_stream,
                 const float* restrict signal_data,
                 float* restrict modulated_signal)
{
    int32_t bit_idx = 0;
    int32_t freq_sig_idx;
    for(; bit_idx < n_bits; ++bit_idx)
    {
        if(!bit_stream[bit_idx])
        {
            freq_sig_idx = FM_FREQ_0;
        }
        else
        {
            freq_sig_idx = FM_FREQ_1;
        }
        set_freq(n_cos_samples, freq_sig_idx, signal_data, &modulated_signal[bit_idx * n_cos_samples]);
    }
}