#include "..\inc\qpsk.h"
#include "..\inc\consts.h"
#include "..\inc\psk_common.h"
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define QPSK_PHASE_01  45
#define QPSK_PHASE_11 135
#define QPSK_PHASE_10 225
#define QPSK_PHASE_00 315

void modulate_qpsk(int32_t n_cos_samples,
                   int32_t n_bits,
                   const int32_t*  bit_stream,
                   const float*  signal_data,
                   float*  modulated_signal)
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

__global__ void
set_phase_shift_cuda_qpsk(int32_t scaled_phase_shift, int32_t n_cos_samples, int32_t phase_shift, const float* signal_data, float* modulated_signal)
{

    int32_t sig_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(sig_idx < (n_cos_samples))
    {
        modulated_signal[sig_idx] = signal_data[scaled_phase_shift + sig_idx];
    }
    else
    {
        printf("dupa");
    }
}

void modulate_qpsk_cuda(int32_t n_cos_samples,
                        int32_t n_bits,
                        const int32_t*  bit_stream,
                        const float*  signal_data,
                        float*  modulated_signal)
{
    int32_t bit_idx = 0, data_idx = 0;
    int32_t phase_shift;
    float* d_modulated_signal;
    float* d_signal_data;

    cudaMalloc((void**)&d_modulated_signal, sizeof(float) * n_cos_samples * (n_bits/2));
    cudaMalloc((void**)&d_signal_data, sizeof(float) * n_cos_samples * 2);
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples * 2, cudaMemcpyHostToDevice);

    int threadsPerBlock = 8;
    int blocksPerGrid   = ( n_cos_samples + threadsPerBlock - 1) / threadsPerBlock;

    int32_t bit_idx_for_cuda = 0;
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
        int32_t scaled_phase_shift = (int32_t)((float)phase_shift * ((float)n_cos_samples)/ N_MAX_DEGREE);
        set_phase_shift_cuda_qpsk<<<blocksPerGrid, threadsPerBlock>>>(scaled_phase_shift, n_cos_samples, phase_shift, d_signal_data, &d_modulated_signal[bit_idx_for_cuda * n_cos_samples]);
        bit_idx_for_cuda++;
    }
    cudaMemcpy(modulated_signal, d_modulated_signal, sizeof(float) * n_cos_samples * n_bits / 2, cudaMemcpyDeviceToHost);
    cudaFree((void*)d_modulated_signal);
    cudaFree((void*)d_signal_data);
}
