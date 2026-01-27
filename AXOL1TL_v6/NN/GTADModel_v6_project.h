#ifndef GTADMODEL_V6_PROJECT_H_
#define GTADMODEL_V6_PROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

namespace hls4ml_axol1tl_v6 {

// Prototype of top level function for C-synthesis
void GTADModel_v6_project(
    input_t input_1[57],
    result_t layer19_out[1]
);

// hls-fpga-machine-learning insert emulator-defines

}

#endif
