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
#include "inc\cuda_common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <time.h>

extern "C"
{
C_DELLEXPORT float *psk_cos_lut = NULL;
C_DELLEXPORT float *modulated_data = NULL;
C_DELLEXPORT int32_t *demodulated_bits = NULL;

C_DELLEXPORT void alloc_memory(int32_t n_bits)
{
    alloc_cuda_memory(n_bits);
    cudaMallocHost(&demodulated_bits, sizeof(int32_t) * n_bits);
    cudaMallocHost(&psk_cos_lut, sizeof(float) * 360 * N_SIGNAL_PERIODS);
    cudaMallocHost(&modulated_data, sizeof(float) * 360 * n_bits);
}

C_DELLEXPORT void memory_free()
{
    free_cuda_memory();
    cudaFreeHost(demodulated_bits);
    cudaFreeHost(psk_cos_lut);
    cudaFreeHost(modulated_data);
}


C_DELLEXPORT int32_t modulate(float amplitude,
                              float freq,
                              int32_t cos_factor_idx,
                              int32_t n_bits,
                              int32_t *bit_stream,
                              char *mod) {
    char bpsk_c[] = "bpsk_c";
    char bpsk_cuda[] = "bpsk_cuda";
    char qpsk_c[] = "qpsk_c";
    char qpsk_cuda[] = "qpsk_cuda";
    char bask_c[] = "bask_c";
    char bask_cuda[] = "bask_cuda";
    char bfsk_c[] = "bfsk_c";
    char bfsk_cuda[] = "bfsk_cuda";
    char _16_qam_c[] = "16qam_c";
    char _16_qam_cuda[] = "16qam_cuda";

    if (!strcmp(mod, bpsk_c)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_psk_cos_lut(&params, psk_cos_lut);

        modulate_bpsk(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, qpsk_c)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_qpsk(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, qpsk_cuda)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        psk_cos_lut = (float *) malloc(sizeof(float) * n_cos_samples * N_SIGNAL_PERIODS);
        modulated_data = (float *) malloc(sizeof(float) * n_cos_samples * n_bits / 2);
        if (!psk_cos_lut || !modulated_data) {
            return 1;
        }

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_qpsk_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, bask_c)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_am(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, bask_cuda)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_am_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, bfsk_c)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_fm_cos_lut(&params, psk_cos_lut);
        modulate_fm(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, bfsk_cuda)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_fm_cos_lut(&params, psk_cos_lut);
        modulate_fm_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, bpsk_cuda)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_bpsk_cuda(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, _16_qam_c)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

        init_psk_cos_lut(&params, psk_cos_lut);
        modulate_16qam(n_cos_samples, n_bits, bit_stream, psk_cos_lut, modulated_data);
    } else if (!strcmp(mod, _16_qam_cuda)) {
        const psk_params params = {amplitude, freq, cos_factor_idx};
        int32_t n_cos_samples = get_n_cos_samples(params.cos_factor_idx);

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

void prepare_ref_data(int32_t n_bits, int32_t* reference_data)
{
    int32_t bit_idx = 0;
    for(; bit_idx < n_bits; ++bit_idx)
    {
        reference_data[bit_idx] = rand() % 2;
    }
}

int32_t compare_results(int32_t n_bits, int32_t* reference_data, int32_t* result_data)
{
    int32_t flag = 0;
    int32_t bit_idx = 0;
    for (; bit_idx < n_bits; bit_idx++)
    {
        if (result_data[bit_idx] != reference_data[bit_idx])
        {
            flag = 1;
        }
    }
    return flag;
}

void test_digital_modulation(float amp,
                             float freq,
                             int32_t cos_factor,
                             int32_t n_bits,
                             int32_t* bit_stream,
                             char* mod)
{
    modulate(amp,
             freq,
             cos_factor,
             n_bits,
             bit_stream,
             mod);

    demodulate(mod,
               amp,
               freq,
               psk_cos_lut,
               modulated_data,
               n_bits,
               cos_factor,
               demodulated_bits);

    printf("Test %s modulation and demodulation.\n",mod);
    if(!compare_results(n_bits, bit_stream, demodulated_bits))
    {
        printf("    Result ==== PASSED\n");
    }
    else
    {
        printf("    Result ==== FAILED\n");
    }
}

int main(void)
{
    char bask_c[] = "bask_c";
    char bask_cuda[] = "bask_cuda";
    char bfsk_c[] = "bfsk_c";
    char bfsk_cuda[] = "bfsk_cuda";
    char bpsk_c[] = "bpsk_c";
    char bpsk_cuda[] = "bpsk_cuda";
    char qpsk_c[] = "qpsk_c";
    char qpsk_cuda[] = "qpsk_cuda";
    char _16_qam_c[] = "16qam_c";
    char _16_qam_cuda[] = "16qam_cuda";
    const int32_t n_bits = 512;
    int32_t bit_stream[n_bits];

    float amp = 3;
    float freq = 10000;
    int32_t cos_factor = 2;

    prepare_ref_data(n_bits, bit_stream);

    alloc_memory(n_bits);


    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, bask_c);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, bask_cuda);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, bfsk_c);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, bfsk_cuda);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, bpsk_c);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, bpsk_cuda);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, qpsk_c);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, qpsk_cuda);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, _16_qam_c);
    test_digital_modulation(amp, freq, cos_factor, n_bits, bit_stream, _16_qam_cuda);

    memory_free();
    return 0;
}
