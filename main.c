#include "inc/qpsk.h"
#include "inc/types.h"
#include "inc/consts.h"

float qpsk_lut[N_MAX_DEGREE * 2];
float modulated_data[N_MAX_DEGREE * 8];
int main()
{
    int32_t bit_stream[] = {0, 1, 1, 0, 0, 0, 1, 1};
    init_qpsk_signal_lut(qpsk_lut);
    modulate_qpsk(8, bit_stream, qpsk_lut, modulated_data);
    return 0;
}
