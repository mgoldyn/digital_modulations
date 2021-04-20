#include "..\inc\bpsk_cuda.h"
#include "..\inc\consts.h"
#include "..\inc\psk_common.h"
#include <cuda_runtime.h>

#define BSPK_PHASE_0 90
#define BPSK_PHASE_1 180

__global__ void
set_phase_shift_cuda(int32_t scaled_phase_shift, int32_t n_cos_samples, int32_t phase_shift, const float* signal_data, float* modulated_signal)
{

    int32_t sig_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(sig_idx < n_cos_samples)
    {
        modulated_signal[sig_idx] = signal_data[scaled_phase_shift + sig_idx];
    }
}

void modulate_bpsk_cuda(int32_t n_cos_samples,
    int32_t n_bits,
    const int32_t* bit_stream,
    const float* signal_data,
    float* modulated_signal)
{
    int32_t bit_idx = 0;
    int32_t phase_shift;
    float* d_modulated_signal;
    float* d_signal_data;

    cudaMalloc((void**)&d_modulated_signal, sizeof(float) * n_cos_samples * n_bits);
    cudaMalloc((void**)&d_signal_data, sizeof(float) * n_cos_samples * 2);
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples * 2, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_bits * n_cos_samples + threadsPerBlock - 1) / threadsPerBlock;

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
        int32_t scaled_phase_shift = (int32_t)((float)phase_shift * ((float)n_cos_samples)/ N_MAX_DEGREE);
        set_phase_shift_cuda<<<blocksPerGrid, threadsPerBlock>>>(scaled_phase_shift, n_cos_samples, phase_shift, d_signal_data, &d_modulated_signal[bit_idx * n_cos_samples]);
    }
    cudaMemcpy(modulated_signal, d_modulated_signal, sizeof(float) * n_cos_samples * n_bits, cudaMemcpyDeviceToHost);
    cudaFree((void*)d_modulated_signal);
    cudaFree((void*)d_signal_data);
}