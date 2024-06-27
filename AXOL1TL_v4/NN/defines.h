#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

namespace hls4ml_axol1tl_v4 {

static const int N_INPUT_1_1=57;
static const int N_LAYER_2=28;
static const int N_LAYER_4=15;
static const int N_LAYER_6=8;

//typedef ap_fixed<5,5> ad_shift_t;
//typedef ap_fixed<10,10> ad_offset_t;
typedef ap_fixed<8,6,AP_RND_CONV,AP_SAT> input_t;
typedef ap_fixed<29,16> q_dense_accum_t;
typedef ap_fixed<29,16> layer2_t;
typedef ap_fixed<6,3> weight2_t;
typedef ap_fixed<10,7> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer3_t;
typedef ap_fixed<18,8> q_dense_quantized_relu_table_t;
typedef ap_fixed<22,15> q_dense_1_accum_t;
typedef ap_fixed<22,15> layer4_t;
typedef ap_fixed<6,3> weight4_t;
typedef ap_fixed<10,7> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer5_t;
typedef ap_fixed<18,8> q_dense_1_quantized_relu_table_t;
typedef ap_fixed<21,14> mu_accum_t;
typedef ap_fixed<21,14> layer6_t;
typedef ap_fixed<6,3> weight6_t;
typedef ap_fixed<10,7> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<10,7,AP_RND_CONV,AP_SAT> result_t;
typedef ap_fixed<18,8> mu_quantized_bits_table_t;

typedef ap_ufixed<18,14> resultsq_t; //from AD_NN_OUT_SQ_T
typedef ap_fixed<18,13> unscaled_t; //from AD_NNNOUTPUTS

extern weight2_t w2[1596];
extern bias2_t b2[28];
extern weight4_t w4[420];
extern bias4_t b4[15];
extern weight6_t w6[120];
extern bias6_t b6[8];

} // namespace hls4ml_axol1tl_v4
#endif