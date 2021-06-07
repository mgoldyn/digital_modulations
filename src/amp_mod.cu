#include "..\inc\amp_mod.h"
#include "..\inc\cuda_common.h"
#include <cuda_runtime.h>

#define AM_AMP_0 0.3f
#define AM_AMP_1 1.f

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

__global__ void set_amplitude_cuda(int32_t n_cos_samples, float amp_factor, const float*  signal_data, float*  modulated_signal)
{
    int32_t sig_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(sig_idx < n_cos_samples)
    {
        modulated_signal[sig_idx] = amp_factor * signal_data[sig_idx];
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
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n_bits * n_cos_samples + threadsPerBlock - 1) / threadsPerBlock;

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
        set_amplitude_cuda<<<blocksPerGrid, threadsPerBlock>>>(n_cos_samples, amp_factor, d_signal_data, &d_modulated_signal[bit_idx * n_cos_samples]);
    }
    cudaMemcpy(modulated_signal, d_modulated_signal, sizeof(float) * n_cos_samples * n_bits, cudaMemcpyDeviceToHost);
}