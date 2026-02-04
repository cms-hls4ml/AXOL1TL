#ifndef __ADT_SCALES_H
#define __ADT_SCALES_H

#include "NN/GTADModel_v6_project.h"

namespace hls4ml_axol1tl_v6 {

typedef ap_fixed<5,5> ad_shift_t;
typedef ap_fixed<10,10> ad_offset_t;

const ad_shift_t ad_shift[57] = {
    3, 0, 6, 2, 5, 6, 0, 5, 6, 0, 5, 6, -1, 5, 6, 1, 7, 8, 0, 7, 8, 0, 7, 8, 0, 7, 8, 3, 6, 6, 3, 6, 6, 3, 6, 6, 2, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6
};

const ad_offset_t ad_offsets[57] = {
14, 0, 72, 7, 0, 71, 4, 2, 71, 4, 2, 71, 3, 0, 72, 5, -2, 286, 3, -0, 284, 3, -0, 286, 3, -2, 285, 31, 0, 71, 24, 0, 71, 21, 0, 71, 18, -2, 72, 16, 0, 72, 12, -2, 72, 11, 0, 72, 10, 0, 71, 10, -2, 72, 10, 0, 71
};

} // namespace hls4ml_axol1tl_v6
#endif

