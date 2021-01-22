#include "../inc/qpsk.h"
#include "../inc/consts.h"
#include <stdio.h>
#include <math.h>

#define QPSK_PHASE_01  45
#define QPSK_PHASE_11 135
#define QPSK_PHASE_10 225
#define QPSK_PHASE_00 315

static inline
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

C_DELLEXPORT
void modulate_qpsk(int32_t n_cos_samples,
                   int32_t n_bits,
                   const int32_t* restrict bit_stream,
                   const float* restrict signal_data,
                   float* restrict modulated_signal)
{
    int32_t bit_idx = 0, data_idx = 0;
    int32_t phase_shift;
    for(; bit_idx < n_bits; bit_idx += 2, ++data_idx)
    {
        if(!bit_stream[bit_idx])
        {
            if(!bit_stream[bit_idx + 1])
            {
                phase_shift = QPSK_PHASE_00;
            }
            else
            {
                phase_shift = QPSK_PHASE_01;
            }
        }
        else
        {
            if(!bit_stream[bit_idx + 1])
            {
                phase_shift = QPSK_PHASE_10;
            }
            else
            {
                phase_shift = QPSK_PHASE_11;
            }
        }
        printf("\n bit idx = %d, data_idx = %d, shift = %d\n", bit_idx, data_idx, phase_shift);
        set_phase_shift(n_cos_samples, phase_shift, signal_data, &modulated_signal[data_idx * n_cos_samples]);
    }
}
