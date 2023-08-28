#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//from https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/master/ugt_hls/src/anomaly_detection/NN/defines.h
//https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/axol1tl1_v2/ugt_hls/src/anomaly_detection/Axol1tl_v2.h

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 57
#define N_LAYER_2 32
#define N_LAYER_4 16
#define N_LAYER_6 8 //changed v2

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<5,5> ad_shift_t;
typedef ap_fixed<10,9> ad_offset_t;
typedef ap_fixed<8,6,AP_RND_CONV,AP_SAT> input_t;
typedef ap_fixed<16,6> model_default_t;
typedef ap_fixed<16,6> layer2_t;
typedef ap_fixed<6,3> weight2_t;
typedef ap_fixed<10,7> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer3_t;
typedef ap_fixed<18,8> q_dense_quantized_relu_table_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<6,3> weight4_t;
typedef ap_fixed<10,7> bias4_t;
typedef ap_uint<1> layer4_index;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer5_t;
typedef ap_fixed<18,8> q_dense_1_quantized_relu_table_t;
typedef ap_fixed<16,6> layer6_t;
typedef ap_fixed<6,3> weight6_t;
typedef ap_fixed<10,7> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<10,7,AP_RND_CONV,AP_SAT> result_t;
typedef ap_fixed<18,8> mu_quantized_bits_table_t;
typedef ap_ufixed<18,14> resultsq_t;
typedef ap_fixed<18,13> unscaled_t;

#endif
