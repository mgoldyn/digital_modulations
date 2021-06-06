#include "inc\qpsk.h"
#include "inc\bpsk.h"
#include "inc\types.h"
#include "inc\consts.h"
#include "inc\psk_common.h"
#include "inc\amp_mod.h"
#include "inc\freq_mod.h"
#include "inc\bpsk_cuda.h"
#include "inc\demodulate.h"
#include "inc\qam.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern "C"
{
C_DELLEXPORT float *psk_cos_lut = NULL;
C_DELLEXPORT float *modulated_data = NULL;
C_DELLEXPORT int32_t *demodulated_bits = NULL;

C_DELLEXPORT void memory_free()
{
    if (demodulated_bits != NULL) {
        free(demodulated_bits);
        demodulated_bits = NULL;
    }
    if (psk_cos_lut != NULL) {
        free(psk_cos_lut);
        psk_cos_lut = NULL;
    }
    if (modulated_data != NULL) {
        free(modulated_data);
        modulated_data = NULL;
    }
}


C_DELLEXPORT int32_t modulate(float amplitude,
                              float freq,
                              int32_t cos_factor_idx,
                              int32_t n_bits,
                              int32_t *bit_stream,
                              char *mod) {
    char bps[] = "bpsk";
    char bpsc[] = "bpskc";
    char qps[] = "qpsk";
    char qpsc[] = "qpskc";
    char am[] = "am";
    char amc[] = "amc";
    char fm[] = "fm";
    char fmc[] = "fmc";
    char _16_qam[] = "16qam";
    char _16_qamc[] = "16qamc";

    memory_free();
    demodulated_bits = (int32_t*)malloc(sizeof(int32_t) * n_bits);

    if (!strcmp(mod, bps)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);
        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);

        modulate_bpsk(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, qps)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits / 2);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_qpsk(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, qpsc)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits / 2);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_qpsk_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, am)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_am(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, amc)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_am_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, fm)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_fm_cos_lut(&params, psk_cos_lut);
        modulate_fm(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, fmc)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_fm_cos_lut(&params, psk_cos_lut);
        modulate_fm_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, bpsc)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_bpsk_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, _16_qam)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_16qam(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, _16_qamc)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_16qam_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    }
    else
    {
        memory_free();
        return 1;
    }
    demodulate(mod,
               amplitude,
               freq,
               psk_cos_lut,
               modulated_data,
               n_bits,
               cos_factor_idx,
               demodulated_bits);

    return 0;
}

C_DELLEXPORT void cuda_dummy_free() {
    cudaFree(0);
}
}

int main(void) {
//    int32_t bit_stream[] = {1,1, 0, 1};//,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
//    char bps[] = "qpsk";
//    int32_t n_bits = 4;
//
//    init_func(1,
//        1,
//        2,
//              n_bits,
//              bit_stream,
//              bps);
//    int i = 0;
////    for(; i < 360; ++i)
////    {
////        printf("mod[%d] = %f \n", i, modulated_data[i]);
////    }
//
//             int32_t demodulated_bits[4];
//    demodulate(bps,
//               0,
//               0,
//               psk_cos_lut,
//               modulated_data,
//               n_bits,
//               360,
//               &demodulated_bits[0]);
//
//    int32_t dupsko = 0;
//    for(i = 0; i < n_bits; ++i)
//    {
//        if(demodulated_bits[i] != bit_stream[i])
//        {dupsko = 1;
//            printf("nie działa idx = %d out = %d, in = %d\n",i, demodulated_bits[i] ,bit_stream[i]);
//        }
//    }
//    if(dupsko)
//    {
//        printf("nie działa\n");
//    }
//    else
//    {
//        printf("dziala\n");
//    }
    int32_t bit_stream[] = {0, 0, 1,1};//,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1};
    char bps[] = "bpskc";
    float amp = 1;
    float freq = 1;
    int32_t cos_factor = 2;
    int32_t n_bits = 4;
    cudaFree(0);
    modulate(amp,
              freq,
              cos_factor,
              n_bits,
              bit_stream,
              bps);
    int32_t demodulated_bits[64];
    demodulate(bps,
               amp,
               freq,
               psk_cos_lut,
               modulated_data,
               n_bits,
               cos_factor,
               demodulated_bits);

    int i = 0;
    for (; i < n_bits; i++) {
        if (demodulated_bits[i] != bit_stream[i]) {
            printf(" %d ||| %d != %d\n",i,  demodulated_bits[i], bit_stream[i]);
        }
    }

    memory_free();
    return 0;
}
