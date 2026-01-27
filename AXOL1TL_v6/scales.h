#ifndef __ADT_SCALES_H
#define __ADT_SCALES_H

#include "NN/GTADModel_v6_project.h"

namespace hls4ml_axol1tl_v6 {

typedef ap_fixed<5,5> ad_shift_t;
typedef ap_fixed<10,10> ad_offset_t;

const ad_shift_t ad_shift[57] = {
    28, 0, 72, 14, 0, 71, 9, 2, 71, 8, 2, 71, 6, 0, 72, 10, -2, 286, 6, 0, 284, 6, 0, 286, 6, -2, 285, 62, 0, 71, 48, 0, 71, 42, 0, 71, 37, -2, 72, 32, 0, 72, 24, -2, 72, 22, 0, 72, 21, 0, 71, 20, -2, 72, 20, 0, 71
};

const ad_offset_t ad_offsets[57] = {
    16, 1, 64, 8, 32, 64, 2, 32, 64, 2, 32, 64, 1, 32, 64, 4, 128, 256, 2, 128, 256, 2, 128, 256, 2, 128, 256, 16, 64, 64, 16, 64, 64, 16, 64, 64, 8, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64
};

} // namespace hls4ml_axol1tl_v6
#endif