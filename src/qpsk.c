#include "/home/mgoldyn/git/magisterka/inc/qpsk.h"
#include "/home/mgoldyn/git/magisterka/inc/consts.h"
#include <stdio.h>
#include <math.h>

#define QPSK_PHASE_01  45
#define QPSK_PHASE_11 135
#define QPSK_PHASE_10 225
#define QPSK_PHASE_00 315

void init_qpsk_signal_lut(float* restrict signal_lut)
{
    int32_t sig_idx = 0, half_sig_idx = N_MAX_DEGREE;
    for(; sig_idx < N_MAX_DEGREE; ++sig_idx)
    {
        signal_lut[sig_idx] = cos(((float)sig_idx * 2 * PI )/ N_MAX_DEGREE);
        signal_lut[half_sig_idx++] = - signal_lut[sig_idx];
    }
}

void set_phase_shift(int32_t phase_shift, const float* restrict signal_data, float* restrict modulated_signal)
{
    int32_t sig_idx = 0;
    int32_t sig_lut_idx = 0;
    for(; sig_idx < N_MAX_DEGREE; ++sig_idx)
    {
        modulated_signal[sig_idx] = signal_data[phase_shift + sig_idx];
    }
}

void modulate_qpsk(int32_t n_bits,
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

        set_phase_shift(phase_shift, signal_data, &modulated_signal[data_idx * N_MAX_DEGREE]);
    }
}
