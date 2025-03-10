#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

namespace hls4ml_axol1tl_v5 {

static const int N_INPUT_1_1=57;
static const int N_LAYER_2=29;
static const int N_LAYER_4=10;
static const int N_LAYER_6=9;
static const int N_LAYER_8=6;
static const int N_LAYER_10=4;
static const int N_LAYER_12=6;
static const int N_LAYER_14=9;
static const int N_LAYER_16=10;
static const int OUT_DOT_19=1;

typedef ap_fixed<14,6,AP_RND_CONV,AP_SAT> input_t; //from AD_NN_IN_T
typedef ap_fixed<27,16> q_dense_accum_t;
typedef ap_fixed<27,16> layer2_t;
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
typedef ap_fixed<21,14> q_dense_2_accum_t;
typedef ap_fixed<21,14> layer6_t;
typedef ap_fixed<6,3> weight6_t;
typedef ap_fixed<10,7> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer7_t;
typedef ap_fixed<18,8> q_dense_2_quantized_relu_table_t;
typedef ap_fixed<21,14> q_dense_3_accum_t;
typedef ap_fixed<21,14> layer8_t;
typedef ap_fixed<6,3> weight8_t;
typedef ap_fixed<10,7> bias8_t;
typedef ap_uint<1> layer8_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer9_t;
typedef ap_fixed<18,8> q_dense_3_quantized_relu_table_t;
typedef ap_fixed<20,13> q_dense_4_accum_t;
typedef ap_fixed<20,13> layer10_t;
typedef ap_fixed<6,3> weight10_t;
typedef ap_fixed<10,7> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<18,8> q_dense_4_linear_table_t;
typedef ap_fixed<25,12> q_dense_6_accum_t;
typedef ap_fixed<25,12> layer12_t;
typedef ap_fixed<6,3> weight12_t;
typedef ap_fixed<10,7> bias12_t;
typedef ap_uint<1> layer12_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer13_t;
typedef ap_fixed<18,8> q_dense_6_quantized_relu_table_t;
typedef ap_fixed<20,13> q_dense_7_accum_t;
typedef ap_fixed<20,13> layer14_t;
typedef ap_fixed<6,3> weight14_t;
typedef ap_fixed<10,7> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer15_t;
typedef ap_fixed<18,8> q_dense_7_quantized_relu_table_t;
typedef ap_fixed<21,14> q_dense_8_accum_t;
typedef ap_fixed<21,14> layer16_t;
typedef ap_fixed<6,3> weight16_t;
typedef ap_fixed<10,7> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<16,6> layer17_t;
typedef ap_fixed<18,8> q_dense_8_linear_table_t;
typedef ap_fixed<16,6> layer18_t;
typedef ap_fixed<18,14,AP_RND_CONV,AP_SAT> dot_default_t;
typedef ap_fixed<18,14,AP_RND_CONV,AP_SAT> result_t; //from AD_NN_OUT_T;

//for emulator
typedef ap_ufixed<18, 14> resultsq_t; //from AD_NN_OUT_SQ_T;
typedef ap_fixed<18,13> unscaled_t; //from AD_NNNOUTPUTS

extern weight2_t w2[1653];
extern bias2_t b2[29];
extern weight4_t w4[290];
extern bias4_t b4[10];
extern weight6_t w6[90];
extern bias6_t b6[9];
extern weight8_t w8[54];
extern bias8_t b8[6];
extern weight10_t w10[24];
extern bias10_t b10[4];
extern weight12_t w12[24];
extern bias12_t b12[6];
extern weight14_t w14[54];
extern bias14_t b14[9];
extern weight16_t w16[90];
extern bias16_t b16[10];

} // namespace hls4ml_axol1tl_v5
#endif
