#include "inc/qpsk.h"
#include "inc/types.h"
#include "inc/consts.h"

#include <stdio.h>

float qpsk_lut[N_DEGREE * 2];
float modulated_data[N_MAX_DEGREE * 4];
int main()
{
    int32_t bit_stream[] = {0, 1, 1, 0, 0, 0, 1, 1};
    init_qpsk_signal_lut(qpsk_lut);
    modulate_qpsk(8, bit_stream, qpsk_lut, modulated_data);
    int32_t i = 0;
    for(; i < 4 *  N_DEGREE; ++i)
    {
        printf("mod[%d] = %f \n", i, modulated_data[i]);
    }
    return 0;
}
