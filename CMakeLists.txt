cmake_minimum_required(VERSION 3.19)
project(digital_modulations CUDA)

set(CMAKE_CUDA_STANDARD 98)


include_directories(inc)
include_directories(D:/cuda/include)
add_executable(digital_modulations
               main.cu
               src/amp_mod.cu
               src/bpsk.cu
               src/bpsk_cuda.cu
               src/demodulate.cu
               src/freq_mod.cu
               src/psk_common.cu
               src/qam.cu
               src/qpsk.cu
#        inc/cuda_common.h src/cuda_common.cu
        )
#-gencode=arch=compute_75,code=compute_75 -shared --output-file=main2.dll
set_property(TARGET digital_modulations PROPERTY CUDA_ARCHITECTURES 75)

set_target_properties(
        digital_modulations
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(digital_modulations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_75,code=compute_75
        -shared
        -o main2.dll
        >)