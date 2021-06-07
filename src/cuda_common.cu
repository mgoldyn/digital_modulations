#include "..\inc\cuda_common.h"
#include <cuda_runtime.h>

static float* d_modulated_signal;
static float* d_signal_data;
static int32_t* d_bit_stream;
#define N_COS_SAMPLES 360
void alloc_cuda_memory(int32_t n_bits)
{
    cudaMalloc((void**)&d_modulated_signal, sizeof(float) * N_COS_SAMPLES * n_bits);
    cudaMalloc((void**)&d_signal_data, sizeof(float) * N_COS_SAMPLES * 2);
    cudaMalloc((void**)&d_bit_stream, sizeof(int32_t) * n_bits);
}

void free_cuda_memory()
{
    cudaFree((void*)d_modulated_signal);
    cudaFree((void*)d_signal_data);
    cudaFree((void*)d_bit_stream);
}

float* get_modulated_signal()
{
    return d_modulated_signal;
}
float* get_signal_data()
{
    return d_signal_data;
}

int32_t* get_bit_stream()
{
    return d_bit_stream;
}