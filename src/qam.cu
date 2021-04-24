#include "..\inc\qam.h"
#include "..\inc\consts.h"

#include <cuda_runtime.h>

#define _16QAM_PHASE_00 315
#define _16QAM_PHASE_01  45
#define _16QAM_PHASE_10 225
#define _16QAM_PHASE_11 135

#define _16QAM_AMP_00 -0.75f
#define _16QAM_AMP_01 -0.25f
#define _16QAM_AMP_10  0.25f
#define _16QAM_AMP_11  0.75f

void set_phase_and_amplitude(int32_t n_cos_samples,
                             int32_t phase_shift,
                             float amp_factor,
                             const float* signal_data,
                             float* modulated_signal)
{
    const int32_t scaled_phase_shift = (int32_t)((float)phase_shift * (((float)n_cos_samples)/ N_MAX_DEGREE));

    int32_t sample_idx = 0;
    for(; sample_idx < n_cos_samples; ++sample_idx)
    {
        modulated_signal[sample_idx] = amp_factor * signal_data[scaled_phase_shift + sample_idx];
    }
}

void modulate_16qam(int32_t n_cos_samples,
                    int32_t n_bits,
                    const int32_t* bit_stream,
                    const float* signal_data,
                    float*  modulated_signal)
{
    int32_t bit_idx = 0, data_idx = 0;
    int32_t phase_shift;
    float amp_factor;
    for(; bit_idx < n_bits; bit_idx += 4, ++data_idx)
    {
        if(!bit_stream[bit_idx])
        {
            if(!bit_stream[bit_idx + 1])
            {
                phase_shift = _16QAM_PHASE_00;
            }
            else
            {
                phase_shift =_16QAM_PHASE_01;
            }
        }
        else
        {
            if(!bit_stream[bit_idx + 1])
            {
                phase_shift = _16QAM_PHASE_10;
            }
            else
            {
                phase_shift = _16QAM_PHASE_11;
            }
        }

        if(!bit_stream[bit_idx + 2])
        {
            if(!bit_stream[bit_idx + 3])
            {
                amp_factor = _16QAM_AMP_00;
            }
            else
            {
                amp_factor =_16QAM_AMP_01;
            }
        }
        else
        {
            if(!bit_stream[bit_idx + 3])
            {
                amp_factor = _16QAM_AMP_10;
            }
            else
            {
                amp_factor = _16QAM_AMP_11;
            }
        }

        set_phase_and_amplitude(n_cos_samples, phase_shift, amp_factor, signal_data, &modulated_signal[data_idx * n_cos_samples]);
    }
}
__global__ void
calc_mod_sig(int32_t n_cos_samples,
             int32_t scaled_phase_shift,
             float amp_factor,
             const float* d_signal_data,
             float* d_modulated_signal)
{
    int32_t sig_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(sig_idx < n_cos_samples)
    {
        d_modulated_signal[sig_idx] = amp_factor * d_signal_data[scaled_phase_shift + sig_idx];
    }
}

//__global__ void
//set_phase_and_amp_16qam(
//        int32_t n_cos_samples,
//                        int32_t* bit_stream,
//                        float* d_signal_data,
//                        float* d_modulated_signal)
//{
//    int32_t sig_idx = 0;
//    for(; sig_idx < n_bits; sig_idx += 4;)
//    {
//        int32_t bit_idx = sig_idx;
//
//        int32_t phase_shift;
//        float amp_factor;
//
//        if (!bit_stream[bit_idx]) {
//            if (!bit_stream[bit_idx + 1]) {
//                phase_shift = _16QAM_PHASE_00;
//            } else {
//                phase_shift = _16QAM_PHASE_01;
//            }
//        } else {
//            if (!bit_stream[bit_idx + 1]) {
//                phase_shift = _16QAM_PHASE_10;
//            } else {
//                phase_shift = _16QAM_PHASE_11;
//            }
//        }
//
//        if (!bit_stream[bit_idx + 2]) {
//            if (!bit_stream[bit_idx + 3]) {
//                amp_factor = _16QAM_AMP_00;
//            } else {
//                amp_factor = _16QAM_AMP_01;
//            }
//        } else {
//            if (!bit_stream[bit_idx + 3]) {
//                amp_factor = _16QAM_AMP_10;
//            } else {
//                amp_factor = _16QAM_AMP_11;
//            }
//        }
//        const int32_t scaled_phase_shift = (int32_t)((float)phase_shift * (((float)n_cos_samples)/ N_MAX_DEGREE));
//
//        int32_t threadsPerBlock = 8;
//        int32_t blocksPerGrid   = ( n_cos_samples + threadsPerBlock - 1) / threadsPerBlock;
//
//        calc_mod_sig<<<blocksPerGrid, threadsPerBlock>>>(n_cos_samples, scaled_phase_shift, amp_factor, d_signal_data, &d_modulated_signal[sig_idx * n_cos_samples]);
//    }
//}

void modulate_16qam_cuda(int32_t n_cos_samples,
                         int32_t n_bits,
                         const int32_t* bit_stream,
                         const float* signal_data,
                         float*  modulated_signal)
{
    float* d_modulated_signal;
    float* d_signal_data;

    cudaMalloc((void**)&d_modulated_signal, sizeof(float) * n_cos_samples * n_bits/4);
    cudaMalloc((void**)&d_signal_data, sizeof(float) * n_cos_samples * 2);
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples * 2, cudaMemcpyHostToDevice);

    int32_t sig_idx = 0;
    for(; sig_idx < n_bits; sig_idx += 4)
    {
        int32_t bit_idx = sig_idx;

        int32_t phase_shift;
        float amp_factor;

        if (!bit_stream[bit_idx]) {
            if (!bit_stream[bit_idx + 1]) {
                phase_shift = _16QAM_PHASE_00;
            } else {
                phase_shift = _16QAM_PHASE_01;
            }
        } else {
            if (!bit_stream[bit_idx + 1]) {
                phase_shift = _16QAM_PHASE_10;
            } else {
                phase_shift = _16QAM_PHASE_11;
            }
        }

        if (!bit_stream[bit_idx + 2]) {
            if (!bit_stream[bit_idx + 3]) {
                amp_factor = _16QAM_AMP_00;
            } else {
                amp_factor = _16QAM_AMP_01;
            }
        } else {
            if (!bit_stream[bit_idx + 3]) {
                amp_factor = _16QAM_AMP_10;
            } else {
                amp_factor = _16QAM_AMP_11;
            }
        }
        const int32_t scaled_phase_shift = (int32_t)((float)phase_shift * (((float)n_cos_samples)/ N_MAX_DEGREE));

        int32_t threadsPerBlock = 8;
        int32_t blocksPerGrid   = ( n_cos_samples + threadsPerBlock - 1) / threadsPerBlock;

        calc_mod_sig<<<blocksPerGrid, threadsPerBlock>>>(n_cos_samples, scaled_phase_shift, amp_factor, d_signal_data, &d_modulated_signal[sig_idx * n_cos_samples]);
        cudaMemcpy(&modulated_signal[(sig_idx/4)*n_cos_samples], d_modulated_signal, sizeof(float) * n_cos_samples, cudaMemcpyDeviceToHost);
    }


    cudaFree((void*)d_modulated_signal);
    cudaFree((void*)d_signal_data);
}