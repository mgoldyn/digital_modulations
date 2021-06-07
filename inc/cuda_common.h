#ifndef DIGITAL_MODULATIONS_CUDA_COMMON_H
#define DIGITAL_MODULATIONS_CUDA_COMMON_H

#include "types.h"
void alloc_cuda_memory(int32_t n_bits);

void free_cuda_memory();

float* get_modulated_signal();

float* get_signal_data();

int32_t* get_bit_stream();
#endif //DIGITAL_MODULATIONS_CUDA_COMMON_H
