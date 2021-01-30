#include "inc/qpsk.h"
// #include "inc/bpsk.h"
#include "inc/types.h"
#include "inc/consts.h"
#include "inc/psk_common.h"

#include <stdio.h>
#include <stdlib.h>

C_DELLEXPORT float* psk_cos_lut;
C_DELLEXPORT float* modulated_data;
C_DELLEXPORT float* dynamic_data;

C_DELLEXPORT int32_t init_func(float amplitude,
                               float freq,
                               int32_t cos_factor_idx,
                               int32_t n_bits,
                               int32_t* bit_stream)
{
    const psk_params params = {amplitude, freq, cos_factor_idx};
    int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

    psk_cos_lut    = malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
    modulated_data = malloc(sizeof(float) * n_cos_samples * n_bits);
    if(!psk_cos_lut || !modulated_data)
    {
        return 0;
    }

    init_psk_cos_lut(&params, psk_cos_lut);
    modulate_qpsk(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    
    return 0;
}

C_DELLEXPORT void memory_free()
{
   free(dynamic_data);
   free(psk_cos_lut);
   free(modulated_data);
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
