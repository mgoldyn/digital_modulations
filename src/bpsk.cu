#include "..\inc\bpsk.h"
#include "..\inc\consts.h"
#include "..\inc\psk_common.h"

#define BSPK_PHASE_0 90
#define BPSK_PHASE_1 270

void modulate_bpsk(int32_t n_cos_samples,
                   int32_t n_bits,
                   const int32_t*  bit_stream,
                   const float*  signal_data,
                   float*  modulated_signal)
{
    int32_t bit_idx = 0;
    int32_t phase_shift;
    for(; bit_idx < n_bits; ++bit_idx)
    {
        if(!bit_stream[bit_idx])
        {
            phase_shift = BSPK_PHASE_0;
        }
        else
        {
            phase_shift = BPSK_PHASE_1;
        }
        set_phase_shift(n_cos_samples, phase_shift, signal_data, &modulated_signal[bit_idx * n_cos_samples]);
    }
}