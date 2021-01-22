#include "inc/qpsk.h"
// #include "inc/bpsk.h"
#include "inc/types.h"
#include "inc/consts.h"
#include "inc/psk_common.h"

#include <stdio.h>

C_DELLEXPORT float psk_cos_lut[N_DEGREE * 2];
C_DELLEXPORT float modulated_data[N_MAX_DEGREE * 8];

C_DELLEXPORT init_func(float amplitude,
                       float freq,
                       int32_t cos_factor_idx)
{
    int32_t bit_stream[] = {0, 0,
                            0, 1,
                            1, 1, 
                            1, 0,
                            1, 0,
                            1, 1,
                            0, 1,
                            0, 0};
    const psk_params params = {amplitude, freq, cos_factor_idx};
    int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);
    init_psk_cos_lut(&params, psk_cos_lut);
    modulate_qpsk(n_cos_samples, 16, bit_stream, psk_cos_lut, modulated_data);
}

int main()
{
    int32_t bit_stream[] = {0, 1, 1, 0, 0, 0, 1, 1};
    psk_params params = {1, 5, 0};
    int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);
    init_psk_cos_lut(&params, psk_cos_lut);
    modulate_qpsk(n_cos_samples, 8, bit_stream, psk_cos_lut, modulated_data);
    int32_t i = 0;
    for(; i < 4* n_cos_samples; ++i)
    {
        printf("mod[%d] = %f \n", i, modulated_data[i]);
    }
    return 0;
}
