#include "inc\qpsk.h"
#include "inc\bpsk.h"
#include "inc\types.h"
#include "inc\consts.h"
#include "inc\psk_common.h"
#include "inc\amp_mod.h"
#include "inc\freq_mod.h"
#include "inc\bpsk_cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern "C"
{
C_DELLEXPORT float* psk_cos_lut;
C_DELLEXPORT float* modulated_data;
C_DELLEXPORT float* dynamic_data;

C_DELLEXPORT int32_t init_func(float amplitude,
                               float freq,
                               int32_t cos_factor_idx,
                               int32_t n_bits,
                               int32_t* bit_stream,
                               char* mod)
{
    char bps[] = "bpsk";
    char bpsc[] = "bpskc";
    char qps[] = "qpsk";
    char qpsc[] = "qpskc";
    char am[]  = "am";
    char amc[]  = "amc";
    char fm[]  = "fm";
    char fmc[]  = "fmc";
    
    if(!strcmp(mod, bps))
    {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);
        clock_t t; t = clock();
        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        
        modulate_bpsk(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("mod C = %s, time = %.30lf\n", mod, time_taken);
    }
    else if(!strcmp(mod, qps))
    {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits / 2);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_qpsk(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        printf("mod C = %s\n", mod);
    }
    else if(!strcmp(mod, qpsc))
    {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits / 2);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_qpsk_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        printf("mod C = %s\n", mod);
    }
    else if(!strcmp(mod, am))
    {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_am(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        printf("mod C = %s\n", mod);
    }
    else if(!strcmp(mod, amc))
    {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_am_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        printf("mod C = %s\n", mod);
    }
    else if(!strcmp(mod, fm))
    {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_fm_cos_lut(&params, psk_cos_lut);
        modulate_fm(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        printf("mod C = %s\n", mod);
    }
    else if(!strcmp(mod, fmc))
    {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_fm_cos_lut(&params, psk_cos_lut);
        modulate_fm_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        printf("mod C = %s\n", mod);
    }
    else if(!strcmp(mod, bpsc))
    {
        cudaFree(0);
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples   = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float*)malloc(sizeof(float) * n_cos_samples * n_bits);
        if(!psk_cos_lut || !modulated_data)
        {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        clock_t t = clock();
        modulate_bpsk_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
        t = clock() - t;
        double time_taken = ((double)t)/CLOCKS_PER_SEC;
        printf("mod C = %s, time = %.30lf\n", mod, time_taken);
        
    }
    else
    {
        return 1;
    }
    
    return 0;
}

C_DELLEXPORT void memory_free()
{
   free(dynamic_data);
   free(psk_cos_lut);
   free(modulated_data);
}
}

int main(void)
{
    // printf("dupa");
    // int32_t bit_stream[] = {0, 1, 1, 0, 0, 0, 1, 1};
    // psk_params params = {1, 5, 2};
    // int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);
    // psk_cos_lut    = (float*)malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
    // modulated_data =  (float*)malloc(sizeof(float) * n_cos_samples * 8);
    // init_psk_cos_lut(&params, psk_cos_lut);
    // modulate_bpsk_cuda(n_cos_samples, 8, bit_stream, psk_cos_lut, modulated_data);
    // printf("dupa");
    // int32_t i = 0;

    int32_t bit_stream[] = {0,1,0,1};//,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
    char bps[] = "fmc";
    cudaFree(0);
    init_func(1,
        1,
        2,
        4,
        bit_stream,
        bps);
         int i = 0;
             for(; i < 360; ++i)
     {
         printf("mod[%d] = %f \n", i, modulated_data[i]);
     }
    memory_free();
//    int32_t bit_stream[] = {0,1,0,1};//,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
    char bps2[] = "fm";
    cudaFree(0);
    init_func(1,
              1,
              2,
              4,
              bit_stream,
              bps2);
    i = 0;
    for(; i < 360; ++i)
    {
        printf("mod[%d] = %f \n", i, modulated_data[i]);
    }

    memory_free();
    return 0;
}
