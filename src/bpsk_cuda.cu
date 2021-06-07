#include "..\inc\bpsk_cuda.h"
#include "..\inc\consts.h"
#include "..\inc\psk_common.h"
#include "..\inc\cuda_common.h"
#include <cuda_runtime.h>
#include <cstring>
#include <stdio.h>

#define BSPK_PHASE_0 90
#define BPSK_PHASE_1 270

#define N_CUDA_ELEM 512

__global__ void
set_phase_shift_cuda(int32_t* bit_stream,
                     int32_t n_cos_samples,
                     int32_t n_bits,
                     const float* signal_data,
                     float* modulated_signal)
{

    int32_t bit_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(bit_idx < n_bits)
    {
        int32_t phase_offset;
        if(!bit_stream[bit_idx])
        {
            phase_offset = (int32_t)((float)BSPK_PHASE_0 * ((float)360)/ N_MAX_DEGREE);
        }
        else
        {
            phase_offset = (int32_t)((float)BPSK_PHASE_1 * ((float)360)/ N_MAX_DEGREE);
        }
        float* modulated_signal_ptr = &modulated_signal[bit_idx * n_cos_samples];
        int32_t sig_idx = 0;
        for(; sig_idx < n_cos_samples; ++sig_idx)
        {
            modulated_signal_ptr[sig_idx] = signal_data[phase_offset + sig_idx];
        }
    }
}

void modulate_bpsk_cuda(int32_t n_cos_samples,
                        int32_t n_bits,
                        const int32_t* bit_stream,
                        const float* signal_data,
                        float* modulated_signal)
{
    float* d_modulated_signal = get_modulated_signal();
    float* d_signal_data = get_signal_data();
    int32_t* d_bit_stream = get_bit_stream();
    int32_t n_elem = n_bits < N_CUDA_ELEM ? n_bits : N_CUDA_ELEM;

//    cudaMalloc((void**)&d_modulated_signal, sizeof(float) * n_cos_samples * n_bits);
//    cudaMalloc((void**)&d_signal_data, sizeof(float) * n_cos_samples * 2);
//    cudaMalloc((void**)&d_bit_stream, sizeof(int32_t) * n_bits);
//    cudaMemcpy(d_bit_stream, bit_stream, sizeof(int32_t) * n_bits, cudaMemcpyHostToDevice);
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples * 2, cudaMemcpyHostToDevice);

    int threadsPerBlock = 16;
    int blocksPerGrid = (N_CUDA_ELEM + threadsPerBlock - 1) / threadsPerBlock;
    cudaStream_t prolog_stream;
    cudaStream_t main_stream[8];
    cudaStream_t epilog_stream;
    cudaStreamCreate(&prolog_stream);
    for(int32_t i  = 0; i < (n_bits - N_CUDA_ELEM) / N_CUDA_ELEM; ++i)
    {
        cudaStreamCreate(&main_stream[i]);
    }
//    prolog
    int32_t bit_idx = 0;
    int32_t n_cuda_prolog_bits = n_bits < N_CUDA_ELEM ? n_bits : bit_idx + N_CUDA_ELEM > n_bits ? n_bits - bit_idx: N_CUDA_ELEM;
    cudaMemcpyAsync(&d_bit_stream[bit_idx],
                    bit_stream,
                    sizeof(int32_t) * n_cuda_prolog_bits,
                    cudaMemcpyHostToDevice,
                    prolog_stream);

    set_phase_shift_cuda<<<blocksPerGrid, threadsPerBlock, 0, prolog_stream>>>(&d_bit_stream[bit_idx],
                                                                               n_cos_samples,
                                                                               n_cuda_prolog_bits,
                                                                               d_signal_data,
                                                                               &d_modulated_signal[bit_idx * n_cos_samples]);
    cudaMemcpyAsync(&modulated_signal[bit_idx * n_cos_samples],
                    &d_modulated_signal[bit_idx * n_cos_samples],
                    sizeof(float) * n_cos_samples * n_cuda_prolog_bits,
                    cudaMemcpyDeviceToHost,
                    prolog_stream);
//  main loop
    int32_t n_main_loop_bits = n_bits - N_CUDA_ELEM;
    bit_idx = N_CUDA_ELEM;
    for(; bit_idx < n_main_loop_bits; bit_idx += N_CUDA_ELEM)
    {
        cudaMemcpyAsync(&d_bit_stream[bit_idx],
                        bit_stream,
                        sizeof(int32_t) * N_CUDA_ELEM,
                        cudaMemcpyHostToDevice,
                        main_stream[(bit_idx / N_CUDA_ELEM) - 1]);

        set_phase_shift_cuda<<<blocksPerGrid, threadsPerBlock, 0, main_stream[(bit_idx / N_CUDA_ELEM) - 1]>>>(&d_bit_stream[bit_idx],
                                                                                                               n_cos_samples,
                                                                                                               N_CUDA_ELEM,
                                                                                                               d_signal_data,
                                                                                                               &d_modulated_signal[bit_idx * n_cos_samples]);
        cudaMemcpyAsync(&modulated_signal[bit_idx * n_cos_samples],
                        &d_modulated_signal[bit_idx * n_cos_samples],
                        sizeof(float) * n_cos_samples * N_CUDA_ELEM,
                        cudaMemcpyDeviceToHost,
                        main_stream[(bit_idx / N_CUDA_ELEM) - 1]);
    }

//    epilog
//printf("mgoldyn bit_idx = %d\n", bit_idx);
    int32_t n_epilog_bits = n_bits - bit_idx;
    if(n_epilog_bits > 0)
    {
        cudaStreamCreate(&epilog_stream);
        cudaMemcpyAsync(&d_bit_stream[bit_idx],
                        bit_stream,
                        sizeof(int32_t) * n_epilog_bits,
                        cudaMemcpyHostToDevice,
                        epilog_stream);

        set_phase_shift_cuda<<<blocksPerGrid, threadsPerBlock, 0, epilog_stream>>>(&d_bit_stream[bit_idx],
                                                                                   n_cos_samples,
                                                                                   n_epilog_bits,
                                                                                   d_signal_data,
                                                                                   &d_modulated_signal[bit_idx * n_cos_samples]);
        cudaMemcpyAsync(&modulated_signal[bit_idx * n_cos_samples],
                        &d_modulated_signal[bit_idx * n_cos_samples],
                        sizeof(float) * n_cos_samples * n_epilog_bits,
                        cudaMemcpyDeviceToHost,
                        epilog_stream);
        cudaStreamDestroy(epilog_stream);
    }
    for(int32_t i  = 0; i < (n_bits - N_CUDA_ELEM) / N_CUDA_ELEM; ++i)
    {
        cudaStreamDestroy(main_stream[i]);
    }
    cudaStreamDestroy(prolog_stream);

//    cudaFree((void*)d_modulated_signal);
//    cudaFree((void*)d_signal_data);
//    cudaFree((void*)d_bit_stream);

}