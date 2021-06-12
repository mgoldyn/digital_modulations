#include "..\inc\freq_mod.h"
#include "..\inc\consts.h"
#include "..\inc\cuda_common.h"

#include "math.h"
#include <cuda_runtime.h>

#define FM_FREQ_0 0
#define FM_FREQ_1 1

#define N_CUDA_ELEM 512

void init_fm_cos_lut(const psk_params*  params, float*  signal_lut)
{
    const float amplitude = params->amplitude;
    const float freq = params->freq;
    const float samples_factor = get_n_cos_samples(params->cos_factor_idx);

    const float period = 1 / freq;
    const float cos_elem_part = period / samples_factor;
    
    int32_t sig_idx = 0;
    for(; sig_idx <= samples_factor; ++sig_idx)
    {
        signal_lut[sig_idx] = amplitude * cos(2 * PI * freq * sig_idx * cos_elem_part * 4);
        
    }
    for(; sig_idx <= samples_factor * 2; ++sig_idx)
    {
        signal_lut[sig_idx] = amplitude * cos(2 * PI * freq * sig_idx * cos_elem_part);
    }
}

inline void set_freq(int32_t n_cos_samples, int32_t freq_sig_idx, const float*  signal_data, float*  modulated_signal)
{
    int32_t sig_idx = 0;
    const int32_t car_signal_offset = n_cos_samples * freq_sig_idx;
    for(; sig_idx < n_cos_samples; ++sig_idx)
    {
        modulated_signal[sig_idx] = signal_data[car_signal_offset + sig_idx];
    }
}

void modulate_fm(int32_t n_cos_samples,
                 int32_t n_bits,
                 const int32_t*  bit_stream,
                 const float*  signal_data,
                 float*  modulated_signal)
{
    int32_t bit_idx = 0;
    int32_t freq_sig_idx;
    for(; bit_idx < n_bits; ++bit_idx)
    {
        if(!bit_stream[bit_idx])
        {
            freq_sig_idx = FM_FREQ_0;
        }
        else
        {
            freq_sig_idx = FM_FREQ_1;
        }
        set_freq(n_cos_samples, freq_sig_idx, signal_data, &modulated_signal[bit_idx * n_cos_samples]);
    }
}

__global__ void set_freq_cuda(int32_t* bit_stream,
                              int32_t n_bits,
                              int32_t n_cos_samples,
                              const float* signal_data,
                              float* modulated_signal)
{
    int32_t bit_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(bit_idx < n_bits)
    {
        float* mod_signal_ptr = &modulated_signal[bit_idx * n_cos_samples];

        int32_t freq_offset = n_cos_samples * (!bit_stream[bit_idx] ? FM_FREQ_0 : FM_FREQ_1);
        int32_t n_sigs = n_cos_samples + freq_offset;
//        int32_t freq_offset = !bit_stream[bit_idx] ? FM_FREQ_0 : FM_FREQ_1;

//        int32_t n_signals = freq_offset * n_cos_samples;
        int32_t sig_idx = freq_offset;
        int32_t mod_idx = 0;
        for(; sig_idx < n_sigs; ++sig_idx, ++mod_idx)
        {
            mod_signal_ptr[mod_idx] = signal_data[sig_idx];
        }
    }
}

void modulate_fm_cuda(int32_t n_cos_samples,
                      int32_t n_bits,
                      const int32_t*  bit_stream,
                      const float*  signal_data,
                      float*  modulated_signal)
{
    float* d_modulated_signal = get_modulated_signal();
    float* d_signal_data = get_signal_data();
    int32_t* d_bit_stream = get_bit_stream();
    cudaMemcpy(d_signal_data, signal_data, sizeof(float) * n_cos_samples * 2, cudaMemcpyHostToDevice);

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

        set_freq_cuda<<<blocksPerGrid, threadsPerBlock, 0, stream[stream_idx]>>>(&d_bit_stream[bit_idx],
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