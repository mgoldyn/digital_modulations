#include "..\inc\amp_mod.h"

#define AM_AMP_0 0.3f
#define AM_AMP_1 1.f

static void set_amplitude(int32_t n_cos_samples, float amp_factor, const float*  signal_data, float*  modulated_signal)
{
    int32_t sig_idx = 0;
    for(; sig_idx < n_cos_samples; ++sig_idx)
    {
        modulated_signal[sig_idx] = amp_factor * signal_data[sig_idx];
    }
}

void modulate_am(int32_t n_cos_samples,
                 int32_t n_bits,
                 const int32_t*  bit_stream,
                 const float*  signal_data,
                 float*  modulated_signal)
{
    int32_t bit_idx = 0;
    float amp_factor;
    for(; bit_idx < n_bits; ++bit_idx)
    {
        if(!bit_stream[bit_idx])
        {
            amp_factor = AM_AMP_0;
        }
        else
        {
            amp_factor = AM_AMP_1;
        }
        set_amplitude(n_cos_samples, amp_factor, signal_data, &modulated_signal[bit_idx * n_cos_samples]);
    }
}