#include "..\inc\amp_mod.h"
#include "..\inc\cuda_common.h"
#include <cuda_runtime.h>

#define AM_AMP_0 0.3f
#define AM_AMP_1 1.f

#define N_CUDA_ELEM 512
static void set_amplitude(int32_t n_cos_samples, float amp_factor, const float*  signal_data, float*  modulated_signal)
{
    int32_t sig_idx = 0;
    for(; sig_idx < n_cos_samples; ++sig_idx)
    {
        modulated_signal[sig_idx] = amp_factor * signal_data[sig_idx];
    }
}

void modulate_am(int32_t n_cos_samples,
                 int32_t n_bits,
                 const int32_t*  bit_stream,
                 const float*  signal_data,
                 float*  modulated_signal)
{
    int32_t bit_idx = 0;
    float amp_factor;
    for(; bit_idx < n_bits; ++bit_idx)
    {
        if(!bit_stream[bit_idx])
        {
            amp_factor = AM_AMP_0;
        }
        else
        {
            amp_factor = AM_AMP_1;
        }
        set_amplitude(n_cos_samples, amp_factor, signal_data, &modulated_signal[bit_idx * n_cos_samples]);
    }
}

__global__ void set_amplitude_cuda(int32_t* bit_stream,
                                   int32_t n_bits,
                                   int32_t n_cos_samples,
                                   const float* signal_data,
                                   float* modulated_signal)
{
    int32_t bit_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(bit_idx < n_bits)
    {
        const float amp_factor = !bit_stream[bit_idx] ? AM_AMP_0 : AM_AMP_1;
        float* mod_signal_ptr = &modulated_signal[bit_idx * n_cos_samples];
        int32_t sig_idx = 0;
        for(; sig_idx < n_cos_samples; ++sig_idx)
        {
            mod_signal_ptr[sig_idx] = amp_factor * signal_data[sig_idx];
        }
    }
}

void modulate_am_cuda(int32_t n_cos_samples,
                      int32_t n_bits,
                      const int32_t*  bit_stream,
                      const float*  signal_data,
                      float*  modulated_signal)
{
    float* d_modulated_signal = get_modulated_signal();
    float* d_signal_data = get_signal_data();
    int32_t* d_bit_stream = get_bit_stream();
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples, cudaMemcpyHostToDevice);

    cudaStream_t stream[20];
    for(int32_t i  = 0; i < (n_bits + N_CUDA_ELEM + 1) / N_CUDA_ELEM; ++i)
    {
        cudaStreamCreate(&stream[i]);
    }

    int32_t threadsPerBlock = 16;
    int32_t blocksPerGrid = (N_CUDA_ELEM + threadsPerBlock - 1) / threadsPerBlock;

    int32_t bit_idx = 0;
    for(; bit_idx < n_bits; bit_idx += N_CUDA_ELEM)
    {
        int32_t n_cuda_bits = n_bits < N_CUDA_ELEM ? n_bits : bit_idx + N_CUDA_ELEM > n_bits ? n_bits - bit_idx: N_CUDA_ELEM;
        int32_t stream_idx = bit_idx / N_CUDA_ELEM;

        cudaMemcpyAsync(&d_bit_stream[bit_idx],
                        bit_stream,
                        sizeof(int32_t) * n_cuda_bits,
                        cudaMemcpyHostToDevice,
                        stream[stream_idx]);

        set_amplitude_cuda<<<blocksPerGrid, threadsPerBlock, 0, stream[stream_idx]>>>(&d_bit_stream[bit_idx],
                                                                                      n_cuda_bits,
                                                                                      n_cos_samples,
                                                                                      d_signal_data,
                                                                                      &d_modulated_signal[bit_idx * n_cos_samples]);
        cudaMemcpyAsync(&modulated_signal[bit_idx * n_cos_samples],
                        &d_modulated_signal[bit_idx * n_cos_samples],
                        sizeof(float) * n_cos_samples * n_cuda_bits,
                        cudaMemcpyDeviceToHost,
                        stream[stream_idx]);
    }
    for(int32_t i  = 0; i < (n_bits + N_CUDA_ELEM + 1) / N_CUDA_ELEM; ++i)
    {
        cudaStreamDestroy(stream[i]);
    }
}