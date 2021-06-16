#ifndef PSK_COMMON_H
#define PSK_COMMON_H
#include "types.h"

void set_phase_shift(int32_t n_cos_samples, int32_t phase_shift, const float*  signal_data, float*  modulated_signal);

#endif
