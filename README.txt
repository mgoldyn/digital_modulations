nvcc src\qam.cu main.cu src\psk_common.cu src\qpsk.cu src\bpsk_cuda.cu src\amp_mod.cu src\freq_mod.cu src\bpsk.cu src\demodulate.cu -gencode=arch=compute_75,code=compute_75 -shared -o main2.dll

profiler gui nvvp -vm "C:\Program Files\Java\jdk1.8.0_291\jre\bin\java"

profiler commands:
 set PATH=%PATH%;D:\cuda\extras\CUPTI\lib64 
 nvvp -vm "C:\Program Files\Java\jdk1.8.0_291\jre\bin\java"
