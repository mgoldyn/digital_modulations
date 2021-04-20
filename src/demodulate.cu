#include "..\inc\demodulate.h"
#include "..\inc\consts.h"

#include <string.h>
#include <stdio.h>

void demodulate(char* mod,
                float amp,
                float freq,
                const float* signal_data,
                const float* modulated_signal,
                int32_t n_bits,
                int32_t n_cos_samples,
                int32_t* demodulated_bits)
{
    char bps[] = "bpsk";
    char bpsc[] = "bpskc";
    char qps[] = "qpsk";
    char qpsc[] = "qpskc";
    char am[]  = "am";
    char amc[]  = "amc";
    char fm[]  = "fm";
    char fmc[]  = "fmc";

    if((strcmp(mod, bps) == 0) || (strcmp(mod, bpsc) == 0))
    {
            int32_t i = 0;
            for (; i < n_bits; ++i)
            {
                int32_t bit_idx = i * n_cos_samples;
                int32_t bpsk_0_counter = 0;
                int32_t bpsk_1_counter = 0;
                int32_t j = 0;
                for (; j < n_cos_samples; ++j)
                {
                    const float* modulated_signal2 = &modulated_signal[bit_idx];
                    if (modulated_signal2[j] == signal_data[(n_cos_samples * 90 / N_MAX_DEGREE) + j])
                    {
                        bpsk_0_counter++;
                    }
                    else if (modulated_signal2[j] == signal_data[(n_cos_samples * 180 / N_MAX_DEGREE) + j])
                    {
                        bpsk_1_counter++;
                    }
                }
                demodulated_bits[i] = (int32_t) (bpsk_0_counter < bpsk_1_counter);
            }
    }
    else if((strcmp(mod, qps) == 0) || (strcmp(mod, qpsc) == 0))
    {
            int32_t i = 0, k = 0;
            for(; i < n_bits/2; ++i)
            {
                int32_t bit_idx = (i) * n_cos_samples;
                int32_t qpsk_00_counter = 0;
                int32_t qpsk_01_counter = 0;
                int32_t qpsk_10_counter = 0;
                int32_t qpsk_11_counter = 0;
                int32_t j = 0;
                const float* modulated_signal2 = &modulated_signal[bit_idx];
                for(; j < n_cos_samples; ++j)
                {
                    printf("mgoldyn [%d][%d] %e/%e/%e/%e/%e\n", i,
                           j,
                           modulated_signal2[j],
                           signal_data[315 + j],
                           signal_data[45 + j],
                           signal_data[225 + j],
                           signal_data[135 + j]);
                    if(modulated_signal2[j] == signal_data[315 + j])
                    {
                        qpsk_00_counter++;
                    }
                    else if(modulated_signal2[j] == signal_data[45 + j])
                    {
                        qpsk_01_counter++;
                    }
                    else if(modulated_signal2[j] == signal_data[225  + j])
                    {
                        qpsk_10_counter++;
                    }
                    else if(modulated_signal2[j] == signal_data[135  + j])
                    {
                        qpsk_11_counter++;
                    }
                }
                printf("mgoldyn 0 = %d, 1 = %d",  (int32_t)(qpsk_10_counter + qpsk_11_counter >  qpsk_00_counter + qpsk_01_counter),
                       (int32_t)(qpsk_10_counter + qpsk_00_counter < qpsk_11_counter + qpsk_01_counter));
                demodulated_bits[k] = (int32_t)((qpsk_10_counter + qpsk_11_counter) >  (qpsk_00_counter + qpsk_01_counter));
                demodulated_bits[k + 1] = (int32_t)((qpsk_10_counter + qpsk_00_counter) < (qpsk_11_counter + qpsk_01_counter));
                k += 2;
            }
    }
    else if((strcmp(mod, fm) == 0) || (strcmp(mod, fmc) == 0))
    {
        int32_t i = 0;
        for (; i < n_bits; ++i) {
            int32_t bit_idx = i * n_cos_samples;
            int32_t fm_0_counter = 0;
            int32_t fm_1_counter = 0;
            int32_t j = 0;
            for (; j < n_cos_samples; ++j) {
                if (modulated_signal[bit_idx + j] == signal_data[j]) {
                    fm_0_counter++;
                } else if (modulated_signal[bit_idx + j] == signal_data[n_cos_samples + j]) {
                    fm_1_counter++;
                }
            }
            demodulated_bits[i] = (int32_t)(fm_0_counter < fm_1_counter);
        }
    }
    else if((strcmp(mod, am) == 0) || (strcmp(mod, amc) == 0))
    {
        int32_t i = 0;
        for (; i < n_bits; ++i) {
            int32_t bit_idx = i * n_cos_samples;
            int32_t am_0_counter = 0;
            int32_t am_1_counter = 0;
            int32_t j = 0;
            for (; j < n_cos_samples; ++j) {
                if (modulated_signal[bit_idx + j] == 0.3f * signal_data[j]) {
                    am_0_counter++;
                } else if (modulated_signal[bit_idx + j] == signal_data[j]) {
                    am_1_counter++;
                }
            }
            demodulated_bits[i] = (int32_t)(am_0_counter < am_1_counter);
        }
    }
}
