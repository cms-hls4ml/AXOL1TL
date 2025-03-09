#ifndef __ADT_SCALES_H
#define __ADT_SCALES_H

#include "NN/GTADModel_v5.h"

namespace hls4ml_axol1tl_v5{

typedef ap_fixed<5,5> ad_shift_t;
typedef ap_fixed<10,10> ad_offset_t;

const ad_shift_t ad_shift[57] = {3, 0, 6, 2, 5, 6, 0, 5, 6, 0, 5, 6, -1, 5, 6, 2, 7, 8, 0, 7, 8, 0, 7, 8, 0, 7, 8, 4, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 3, 6, 6, 2, 6, 6};
const ad_offset_t ad_offsets[57] = {18, 0, 72, 7, 0, 73, 4, 0, 73, 4, 0, 72, 3, 0, 72, 6, -0, 286, 3, -2, 285, 3, -2, 282, 3, -2, 286, 29, 0, 72, 22, 0, 72, 18, 0, 72, 14, 0, 72, 11, 0, 72, 10, 0, 72, 10, 0, 73, 9, 0, 73, 9, 0, 72, 8, -2, 72};

} // namespace hls4ml_axol1tl_v5
#endif
