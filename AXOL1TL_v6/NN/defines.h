#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>

namespace hls4ml_axol1tl_v6 {

// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<14,6,AP_RND_CONV,AP_SAT,0> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<27,16> q_dense_result_t;
typedef ap_fixed<6,3> weight2_t;
typedef ap_fixed<10,7> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> layer3_t;
typedef ap_fixed<18,8> q_dense_quantized_relu_table_t;
typedef ap_fixed<22,15> q_dense_1_result_t;
typedef ap_fixed<6,3> weight4_t;
typedef ap_fixed<10,7> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> layer5_t;
typedef ap_fixed<18,8> q_dense_1_quantized_relu_table_t;
typedef ap_fixed<21,14> q_dense_2_result_t;
typedef ap_fixed<6,3> weight6_t;
typedef ap_fixed<10,7> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> layer7_t;
typedef ap_fixed<18,8> q_dense_2_quantized_relu_table_t;
typedef ap_fixed<21,14> q_dense_3_result_t;
typedef ap_fixed<6,3> weight8_t;
typedef ap_fixed<10,7> bias8_t;
typedef ap_uint<1> layer8_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> layer9_t;
typedef ap_fixed<18,8> q_dense_3_quantized_relu_table_t;
typedef ap_fixed<20,13> q_dense_4_result_t;
typedef ap_fixed<6,3> weight10_t;
typedef ap_fixed<10,7> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<29,19> q_dense_6_result_t;
typedef ap_fixed<6,3> weight12_t;
typedef ap_fixed<10,7> bias12_t;
typedef ap_uint<1> layer12_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> layer13_t;
typedef ap_fixed<18,8> q_dense_6_quantized_relu_table_t;
typedef ap_fixed<20,13> q_dense_7_result_t;
typedef ap_fixed<6,3> weight14_t;
typedef ap_fixed<10,7> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT,0> layer15_t;
typedef ap_fixed<18,8> q_dense_7_quantized_relu_table_t;
typedef ap_fixed<21,14> q_dense_8_result_t;
typedef ap_fixed<6,3> weight16_t;
typedef ap_fixed<10,7> bias16_t;
typedef ap_uint<1> layer16_index;
typedef ap_fixed<22,15> subtract_result_t;
typedef ap_fixed<18,14,AP_RND_CONV,AP_SAT,0> result_t;

// hls-fpga-machine-learning insert emulator-defines

}

#endif
