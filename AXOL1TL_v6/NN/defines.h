#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

namespace hls4ml_axol1tl_v6 {

static const int N_INPUT_1_1=57;
static const int N_LAYER_3=29;
static const int N_LAYER_7=10;
static const int N_LAYER_11=9;
static const int N_LAYER_15=6;
static const int N_LAYER_19=4;
static const int N_LAYER_21=6;
static const int N_LAYER_25=9;
static const int N_LAYER_29=10;
static const int OUT_DOT_32=1;

//for emulator //unchanged from v5
typedef ap_fixed<14,6,AP_RND_CONV,AP_SAT,0> input_t; //input_2_t; //from AD_NN_IN_T
typedef ap_fixed<18,14,AP_RND_CONV,AP_SAT,0> result_t; //from AD_NN_OUT_T;
typedef ap_ufixed<18,14> resultsq_t; //from AD_NN_OUT_SQ_T;
typedef ap_fixed<18,13> unscaled_t; //from AD_NNNOUTPUTS

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<20,9> q_dense_accum_t;
typedef ap_fixed<20,9> q_dense_t;
typedef ap_uint<1> layer3_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> activation_t;
typedef ap_fixed<18,8> activation_table_t;
typedef ap_fixed<17,10> q_dense_1_accum_t;
typedef ap_fixed<17,10> q_dense_1_t;
typedef ap_uint<1> layer7_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> activation_1_t;
typedef ap_fixed<18,8> activation_1_table_t;
typedef ap_fixed<16,9> q_dense_2_accum_t;
typedef ap_fixed<16,9> q_dense_2_t;
typedef ap_uint<1> layer11_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> activation_2_t;
typedef ap_fixed<18,8> activation_2_table_t;
typedef ap_fixed<16,9> q_dense_3_accum_t;
typedef ap_fixed<16,9> q_dense_3_t;
typedef ap_uint<1> layer15_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> activation_3_t;
typedef ap_fixed<18,8> activation_3_table_t;
typedef ap_fixed<15,8> q_dense_4_accum_t;
typedef ap_fixed<15,8> q_dense_4_t;
typedef ap_uint<1> layer19_index;
typedef ap_fixed<20,10> q_dense_6_accum_t;
typedef ap_fixed<20,10> q_dense_6_t;
typedef ap_uint<1> layer21_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> activation_4_t;
typedef ap_fixed<18,8> activation_4_table_t;
typedef ap_fixed<16,9> q_dense_7_accum_t;
typedef ap_fixed<16,9> q_dense_7_t;
typedef ap_uint<1> layer25_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> activation_5_t;
typedef ap_fixed<18,8> activation_5_table_t;
typedef ap_fixed<17,10> q_dense_8_accum_t;
typedef ap_fixed<17,10> q_dense_8_t;
typedef ap_uint<1> layer29_index;
typedef ap_fixed<18,11> subtract_t;
typedef ap_fixed<37,23> dot_accum_t;

} // namespace hls4ml_axol1tl_v6
#endif
