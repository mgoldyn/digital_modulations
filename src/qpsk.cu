#include "..\inc\qpsk.h"
#include "..\inc\consts.h"
#include "..\inc\psk_common.h"
#include "..\inc\cuda_common.h"
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define QPSK_PHASE_01  45
#define QPSK_PHASE_11 135
#define QPSK_PHASE_10 225
#define QPSK_PHASE_00 315

#define N_CUDA_ELEM 128

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
set_phase_offset_qpsk_cuda(const int32_t* bit_stream,
                           int32_t n_bits,
                           int32_t* phase_offset)
{
    int32_t bit_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(bit_idx < n_bits)
    {
        if(!bit_stream[bit_idx * 2])
        {
            if(!bit_stream[(bit_idx * 2) + 1])
            {
                phase_offset[bit_idx] = QPSK_PHASE_00;
            }
            else
            {
                phase_offset[bit_idx] = QPSK_PHASE_01;
            }
        }
        else
        {
            if(!bit_stream[(bit_idx * 2) + 1])
            {
                phase_offset[bit_idx] = QPSK_PHASE_10;
            }
            else
            {
                phase_offset[bit_idx] = QPSK_PHASE_11;
            }
        }
    }
}

__global__ void
set_phase_shift_qspk_cuda(int32_t* phase_offset,
                          int32_t n_cos_samples,
                          int32_t n_bits,
                          const float* signal_data,
                          float* modulated_signal)
{

    int32_t bit_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(bit_idx < n_bits)
    {
        float* modulated_signal_ptr = &modulated_signal[bit_idx * n_cos_samples];
        int32_t sig_idx = 0;
        for(; sig_idx < n_cos_samples; ++sig_idx)
        {
            modulated_signal_ptr[sig_idx] = signal_data[phase_offset[bit_idx] + sig_idx];
        }
    }
}

void modulate_qpsk_cuda(int32_t n_cos_samples,
                        int32_t n_bits,
                        const int32_t*  bit_stream,
                        const float*  signal_data,
                        float*  modulated_signal)
{
    float* d_modulated_signal = get_modulated_signal();
    float* d_signal_data = get_signal_data();
    int32_t* d_bit_stream = get_bit_stream();
    int32_t* d_phase_offset;
    int32_t n_elem = n_bits / 2 < N_CUDA_ELEM ? n_bits / 2 : N_CUDA_ELEM;

    cudaMalloc((void**)&d_phase_offset, sizeof(int32_t) * n_bits / 2);
    cudaMemcpy(d_bit_stream, bit_stream, sizeof(int32_t) * n_bits, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples * 2, cudaMemcpyHostToDevice);

    int threadsPerBlock = 16;
    int blocksPerGrid   = ( N_CUDA_ELEM + threadsPerBlock - 1) / threadsPerBlock;

    int32_t n_data = n_bits / 2;
    int32_t data_idx = 0;
    for(; data_idx < n_data; data_idx += N_CUDA_ELEM)
    {
        int32_t n_cuda_bits = n_data < N_CUDA_ELEM ? n_data : data_idx + N_CUDA_ELEM > n_data ? n_data - data_idx: N_CUDA_ELEM;
        set_phase_offset_qpsk_cuda<<<blocksPerGrid, threadsPerBlock>>>(&d_bit_stream[data_idx],
                                                                       n_cuda_bits,
                                                                       &d_phase_offset[data_idx]);
    }


    for(data_idx = 0; data_idx < n_data; data_idx += N_CUDA_ELEM)
    {
        int32_t n_cuda_bits = n_data < N_CUDA_ELEM ? n_data : data_idx + N_CUDA_ELEM > n_data ? n_data - data_idx : N_CUDA_ELEM;
        set_phase_shift_qspk_cuda<<<blocksPerGrid, threadsPerBlock>>>(&d_phase_offset[data_idx],
                                                                      n_cos_samples,
                                                                      n_cuda_bits,
                                                                      d_signal_data,
                                                                      d_modulated_signal);
        cudaMemcpy(&modulated_signal[data_idx * n_cos_samples],
                   d_modulated_signal,
                   sizeof(float) * n_cos_samples * n_cuda_bits,
                   cudaMemcpyDeviceToHost);
    }

    cudaFree((void*)d_phase_offset);
}
