#include "..\inc\bpsk_cuda.h"
#include "..\inc\consts.h"
#include "..\inc\psk_common.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdio.h>

#define BSPK_PHASE_0 90
#define BPSK_PHASE_1 180

#define N_CUDA_ELEM 128

__global__ void
set_phase_shift_cuda(const int32_t* bit_stream,
                     int32_t n_cos_samples,
                     int32_t n_bits,
                     const float* signal_data,
                     float* modulated_signal)
{

    int32_t bit_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(bit_idx < n_bits)
    {
        int32_t phase_shift;
        if(!bit_stream[bit_idx])
        {
            phase_shift = BSPK_PHASE_0;
        }
        else
        {
            phase_shift = BPSK_PHASE_1;
        }
        int32_t scaled_phase_shift = (int32_t)((float)phase_shift * ((float)360)/ N_MAX_DEGREE);

        float* modulated_signal_ptr = &modulated_signal[bit_idx * n_cos_samples];
        int32_t sig_idx = 0;
        for(; sig_idx < n_cos_samples; ++sig_idx)
        {
            modulated_signal_ptr[sig_idx] = signal_data[scaled_phase_shift + sig_idx];
        }
//        modulated_signal[sig_idx] = signal_data[scaled_phase_shift + sig_idx];
    }
}

void modulate_bpsk_cuda(int32_t n_cos_samples,
                        int32_t n_bits,
                        const int32_t* bit_stream,
                        const float* signal_data,
                        float* modulated_signal)
{
    int32_t bit_idx = 0;
    float* d_modulated_signal;
    float* d_signal_data;
    int32_t* d_bit_stream;

    cudaMalloc((void**)&d_modulated_signal, sizeof(float) * n_cos_samples * N_CUDA_ELEM);
    cudaMalloc((void**)&d_signal_data, sizeof(float) * n_cos_samples * 2);
    cudaMalloc((void**)&d_bit_stream, sizeof(int32_t) * n_bits);
    cudaMemcpy(d_bit_stream, bit_stream, sizeof(int32_t) * n_bits, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples * 2, cudaMemcpyHostToDevice);

    int threadsPerBlock = 16;
//    int blocksPerGrid = (n_bits * n_cos_samples + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid = (N_CUDA_ELEM + threadsPerBlock - 1) / threadsPerBlock;
//    printf("mgoldyn blocks = %d", blocksPerGrid);
    for(; bit_idx < n_bits; bit_idx += N_CUDA_ELEM)
    {
        int32_t n_cuda_bits = (bit_idx + N_CUDA_ELEM) < n_bits ? N_CUDA_ELEM : (n_bits % N_CUDA_ELEM) == 0 ? N_CUDA_ELEM : n_bits - ((n_bits % N_CUDA_ELEM) * N_CUDA_ELEM);
        set_phase_shift_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_bit_stream, n_cos_samples, n_cuda_bits, d_signal_data, d_modulated_signal);
        cudaMemcpy(&modulated_signal[bit_idx * n_cos_samples], d_modulated_signal, sizeof(float) * n_cos_samples * N_CUDA_ELEM, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
//        for(int32_t i = 0; i < 360; ++i)
//        {
//            printf("mgoldyn out[%d] = %e\n", i, modulated_signal[i]);
//        }
    }
    cudaFree((void*)d_modulated_signal);
    cudaFree((void*)d_signal_data);
    cudaFree((void*)d_bit_stream);

}