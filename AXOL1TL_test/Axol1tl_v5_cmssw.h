#ifndef AXOL1TL_V5_DA_H_
#define AXOL1TL_V5_DA_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

namespace GTADModel_v5 {

// Prototype of top level function for C-synthesis
void Axol1tl_v5_da(
    input_2_t input_2[N_INPUT_1_1],
    result_t layer32_out[OUT_DOT_32]
);

}

#endif
