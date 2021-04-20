#ifndef TYPES_H
#define TYPES_H

#define C_DELLEXPORT __declspec(dllexport)
#define CUDA_DELLEXPORT extern "C" __declspec(dllexport)

#define C_EXTERN extern "C"

typedef int int32_t;
typedef unsigned int uint32_t;

typedef struct cint32_t
{
    int32_t re;
    int32_t im;
}cint32_t;

typedef struct cfloat_t
{
    float re;
    float im;
}cfloat_t;

#endif
