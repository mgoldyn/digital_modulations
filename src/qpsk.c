#include "../inc/qpsk.h"
#include "../inc/consts.h"
#include <stdio.h>
#include <math.h>

#define QPSK_PHASE_01  45
#define QPSK_PHASE_11 135
#define QPSK_PHASE_10 225
#define QPSK_PHASE_00 315

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
        
        set_phase_shift(n_cos_samples, phase_shift, signal_data, &modulated_signal[data_idx * n_cos_samples]);
    }
}
