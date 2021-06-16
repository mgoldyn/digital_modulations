#include "..\inc\psk_common.h"

void set_phase_shift(int32_t n_cos_samples, int32_t phase_shift, const float* signal_data, float* modulated_signal)
{
    const int32_t scaled_phase_shift = (int32_t)((float)phase_shift * ((float)n_cos_samples)/ N_MAX_DEGREE);

    int32_t sig_idx = 0;
    for(; sig_idx < n_cos_samples; ++sig_idx)
    {
        modulated_signal[sig_idx] = signal_data[scaled_phase_shift + sig_idx];
    }
}
