cmake_minimum_required(VERSION 3.19)
project(digital_modulations CUDA)
include_directories(D:/magisterka/git/digital_modulations/inc/)
include_directories(D:/cuda/include/)
set(CMAKE_CUDA_STANDARD 17)
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
               src/cuda_common.cu)

add_library(digital_modulations_lib SHARED
            main.cu
            src/amp_mod.cu
            src/bpsk.cu
            src/bpsk_cuda.cu
            src/demodulate.cu
            src/freq_mod.cu
            src/psk_common.cu
            src/qam.cu
            src/qpsk.cu
            src/cuda_common.cu)
set_target_properties(digital_modulations
                      PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(digital_modulations PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -gencode=arch=compute_75,code=compute_75
                       >)
target_compile_options(digital_modulations_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -gencode=arch=compute_75,code=compute_75
                       >)
