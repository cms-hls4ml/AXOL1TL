#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_6 32
#define N_LAYER_10 32
#define N_LAYER_14 5

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,8> model_default_t;
typedef ap_fixed<16,8> input_t;
typedef ap_fixed<16,8> layer2_t;
typedef ap_fixed<6,1> weight2_t;
typedef ap_fixed<6,1> bias2_t;
typedef ap_uint<1> layer2_index;
typedef ap_fixed<16,8> layer4_t;
typedef ap_fixed<16,8> layer5_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<16,8> layer6_t;
typedef ap_fixed<6,1> weight6_t;
typedef ap_fixed<6,1> bias6_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<16,8> layer8_t;
typedef ap_fixed<16,8> layer9_t;
typedef ap_fixed<18,8> relu2_table_t;
typedef ap_fixed<16,8> layer10_t;
typedef ap_fixed<6,1> weight10_t;
typedef ap_fixed<6,1> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,8> layer12_t;
typedef ap_fixed<16,8> layer13_t;
typedef ap_fixed<18,8> relu3_table_t;
typedef ap_fixed<16,8> layer14_t;
typedef ap_fixed<6,1> weight14_t;
typedef ap_fixed<6,1> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_fixed<16,8> result_t;
typedef ap_fixed<18,8> softmax_table_t;

#endif
