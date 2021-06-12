#include "..\inc\demodulate.h"
#include "..\inc\consts.h"
#include "..\inc\qam.h"
#include "..\inc\psk_common.h"

#include <string.h>
#include <stdio.h>

enum _16_qam_bits
{
    _16_qam_00 = 0,
    _16_qam_01 = 1,
    _16_qam_10 = 2,
    _16_qam_11 = 3,
    _16_qam_size = 4
};

static inline void demodulate_16_qam(float modulated_signal,
                                     const float* ref_signal,
                                     int32_t* phase_ctr,
                                     int32_t* amp_ctr)
{
    if (modulated_signal == _16QAM_AMP_00 * ref_signal[_16QAM_PHASE_00])
    {
        amp_ctr[_16_qam_00]++;
        phase_ctr[_16_qam_00]++;
    }
    else if (modulated_signal == _16QAM_AMP_01 * ref_signal[_16QAM_PHASE_00])
    {
        amp_ctr[_16_qam_01]++;
        phase_ctr[_16_qam_00]++;
    }
    else if (modulated_signal == _16QAM_AMP_10 * ref_signal[_16QAM_PHASE_00])
    {
        amp_ctr[_16_qam_10]++;
        phase_ctr[_16_qam_00]++;
    }
    else if (modulated_signal == _16QAM_AMP_11 * ref_signal[_16QAM_PHASE_00])
    {
        amp_ctr[_16_qam_11]++;
        phase_ctr[_16_qam_00]++;
    }
    else if (modulated_signal == _16QAM_AMP_00 * ref_signal[_16QAM_PHASE_01])
    {
        amp_ctr[_16_qam_00]++;
        phase_ctr[_16_qam_01]++;
    }
    else if (modulated_signal == _16QAM_AMP_01 * ref_signal[_16QAM_PHASE_01])
    {
        amp_ctr[_16_qam_01]++;
        phase_ctr[_16_qam_01]++;
    }
    else if (modulated_signal == _16QAM_AMP_10 * ref_signal[_16QAM_PHASE_01])
    {
        amp_ctr[_16_qam_10]++;
        phase_ctr[_16_qam_01]++;
    }

    else if (modulated_signal == _16QAM_AMP_11 * ref_signal[_16QAM_PHASE_01])
    {
        amp_ctr[_16_qam_11]++;
        phase_ctr[_16_qam_01]++;
    }
    else if (modulated_signal == _16QAM_AMP_00 * ref_signal[_16QAM_PHASE_10])
    {
        amp_ctr[_16_qam_00]++;
        phase_ctr[_16_qam_10]++;
    }
    else if (modulated_signal == _16QAM_AMP_01 * ref_signal[_16QAM_PHASE_10])
    {
        amp_ctr[_16_qam_01]++;
        phase_ctr[_16_qam_10]++;
    }
    else if (modulated_signal == _16QAM_AMP_10 * ref_signal[_16QAM_PHASE_10])
    {
        amp_ctr[_16_qam_10]++;
        phase_ctr[_16_qam_10]++;
    }
    else if (modulated_signal == _16QAM_AMP_11 * ref_signal[_16QAM_PHASE_10])
    {
        amp_ctr[_16_qam_11]++;
        phase_ctr[_16_qam_10]++;
    }
    else if (modulated_signal == _16QAM_AMP_00 * ref_signal[_16QAM_PHASE_11])
    {
        amp_ctr[_16_qam_00]++;
        phase_ctr[_16_qam_11]++;
    }
    else if (modulated_signal == _16QAM_AMP_01 * ref_signal[_16QAM_PHASE_11])
    {
        amp_ctr[_16_qam_01]++;
        phase_ctr[_16_qam_11]++;
    }
    else if (modulated_signal == _16QAM_AMP_10 * ref_signal[_16QAM_PHASE_11])
    {
        amp_ctr[_16_qam_10]++;
        phase_ctr[_16_qam_11]++;
    }
    else
    {
        amp_ctr[_16_qam_11]++;
        phase_ctr[_16_qam_11]++;
    }
}

void demodulate(char* mod,
                float amp,
                float freq,
                const float* signal_data,
                const float* modulated_signal,
                int32_t n_bits,
                int32_t cos_factor,
                int32_t* demodulated_bits){
    int32_t n_cos_samples = get_n_cos_samples(cos_factor);
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

    if (!strcmp(mod, bpsk_c) || !strcmp(mod, bpsk_cuda)) {
        int32_t i = 0;
        for (; i < n_bits; ++i) {
            int32_t bit_idx = i * n_cos_samples;
            int32_t bpsk_0_counter = 0;
            int32_t bpsk_1_counter = 0;
            int32_t j = 0;
            for (; j < n_cos_samples; ++j) {
                const float *modulated_signal2 = &modulated_signal[bit_idx];
                if (modulated_signal2[j] == signal_data[(n_cos_samples * 90 / N_MAX_DEGREE) + j]) {
                    bpsk_0_counter++;
                } else if (modulated_signal2[j] == signal_data[(n_cos_samples * 270 / N_MAX_DEGREE) + j]) {
                    bpsk_1_counter++;
                }
            }
            demodulated_bits[i] = (int32_t) (bpsk_0_counter < bpsk_1_counter);
        }
    } else if (!strcmp(mod, qpsk_c) || !strcmp(mod, qpsk_cuda)) {
        int32_t i = 0, k = 0;
        for (; i < n_bits / 2; ++i) {
            int32_t bit_idx = (i) * n_cos_samples;
            int32_t qpsk_00_counter = 0;
            int32_t qpsk_01_counter = 0;
            int32_t qpsk_10_counter = 0;
            int32_t qpsk_11_counter = 0;
            int32_t j = 0;
            const float *modulated_signal2 = &modulated_signal[bit_idx];
            for (; j < n_cos_samples; ++j) {
                if (modulated_signal2[j] == signal_data[315 + j]) {
                    qpsk_00_counter++;
                } else if (modulated_signal2[j] == signal_data[45 + j]) {
                    qpsk_01_counter++;
                } else if (modulated_signal2[j] == signal_data[225 + j]) {
                    qpsk_10_counter++;
                } else if (modulated_signal2[j] == signal_data[135 + j]) {
                    qpsk_11_counter++;
                }
            }
            demodulated_bits[k] = (int32_t) ((qpsk_10_counter + qpsk_11_counter) > (qpsk_00_counter + qpsk_01_counter));
            demodulated_bits[k + 1] = (int32_t) ((qpsk_10_counter + qpsk_00_counter) <
                                                 (qpsk_11_counter + qpsk_01_counter));
            k += 2;
        }
    } else if (!strcmp(mod, bfsk_c) || !strcmp(mod, bfsk_cuda)) {
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
            demodulated_bits[i] = (int32_t) (fm_0_counter < fm_1_counter);
        }
    }
    else if (!strcmp(mod, bask_c) || !strcmp(mod, bask_cuda)) {
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
            demodulated_bits[i] = (int32_t) (am_0_counter < am_1_counter);
        }
    }
    else if (!strcmp(mod, _16_qam_c)  || !strcmp(mod, _16_qam_cuda))
    {
        int32_t i = 0, k = 0;
        for (; i < n_bits / 4; ++i) {
            int32_t bit_idx = i * n_cos_samples;
            int32_t phase_counter[_16_qam_size] = {0};
            int32_t amp_counter[_16_qam_size]   = {0};
            int32_t j = 0;
            const float *modulated_signal_bit = &modulated_signal[bit_idx];
            for (; j < n_cos_samples; ++j)
            {
                demodulate_16_qam(modulated_signal_bit[j], &signal_data[j], phase_counter, amp_counter);
            }
            //phase bits
            demodulated_bits[k + 0] = (int32_t) ((phase_counter[_16_qam_10] + phase_counter[_16_qam_11]) >
                    (phase_counter[_16_qam_01] + phase_counter[_16_qam_00]));
            demodulated_bits[k + 1] = (int32_t) ((phase_counter[_16_qam_11] + phase_counter[_16_qam_01]) >
                    (phase_counter[_16_qam_10] + phase_counter[_16_qam_00]));
            //amp bits
            demodulated_bits[k + 2] = (int32_t) ((amp_counter[_16_qam_10] + amp_counter[_16_qam_11]) >
                    (amp_counter[_16_qam_01] + amp_counter[_16_qam_00]));
            demodulated_bits[k + 3] = (int32_t) ((amp_counter[_16_qam_01] + amp_counter[_16_qam_11]) >
                    (amp_counter[_16_qam_00] + amp_counter[_16_qam_10]));
            k += 4;
        }
    }
    else
    {
        printf("dupa\n");
    }
}
