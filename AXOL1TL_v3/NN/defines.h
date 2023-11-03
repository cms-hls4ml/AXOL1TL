#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>


// hls-fpga-machine-learning insert numbers
//copied from https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/axol1tl_v3/ugt_hls/src/anomaly_detection/Axol1tl_v3.h
#define N_INPUT_1_1 57
#define N_LAYER_2 32
#define N_LAYER_4 16
#define N_LAYER_6 8 //changed v3

//hls-fpga-machine-learning insert layer-precision
//copied from https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/axol1tl_v3/ugt_hls/src/anomaly_detection/Axol1tl_v3.h
typedef ap_fixed<8,6,AP_RND_CONV,AP_SAT> input_t;
typedef ap_fixed<21,16> q_dense_accum_t;
typedef ap_fixed<21,16> layer2_t;
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

//check definitions
//from https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/axol1tl_v3/ugt_hls/src/anomaly_detection/Axol1tl_v3.h
typedef ap_ufixed<18,14> resultsq_t; //from AD_NN_OUT_SQ_T
typedef ap_fixed<18,13> unscaled_t; //from AD_NNNOUTPUTS

//define weights here instead of in weight/ files
//filled in weights.cpp
extern weight2_t w2[1824];
extern bias2_t b2[32];
extern weight4_t w4[512];
extern bias4_t b4[16];
extern weight6_t w6[128];
extern bias6_t b6[8];

#endif
