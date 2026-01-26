#include <iostream>

#include "GTADModel_v6_project.h"
#include "parameters.h"

namespace hls4ml_axol1tl_v6 {

void GTADModel_v6_project(
    input_t input_1[57],
    result_t layer19_out[1]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer19_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    q_dense_result_t layer2_out[29];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[29];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    q_dense_1_result_t layer4_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    layer5_t layer5_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    q_dense_2_result_t layer6_out[9];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    layer7_t layer7_out[9];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    q_dense_3_result_t layer8_out[6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    layer9_t layer9_out[6];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    q_dense_4_result_t layer10_out[4];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0

    q_dense_6_result_t layer12_out[6];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    layer13_t layer13_out[6];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    q_dense_7_result_t layer14_out[9];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    layer15_t layer15_out[9];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    q_dense_8_result_t layer16_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0

    subtract_result_t layer18_out[10];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0

    nnet::dense<input_t, q_dense_result_t, config2>(input_1, layer2_out, w2, b2); // q_dense

    nnet::relu<q_dense_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // q_dense_quantized_relu

    nnet::dense<layer3_t, q_dense_1_result_t, config4>(layer3_out, layer4_out, w4, b4); // q_dense_1

    nnet::relu<q_dense_1_result_t, layer5_t, relu_config5>(layer4_out, layer5_out); // q_dense_1_quantized_relu

    nnet::dense<layer5_t, q_dense_2_result_t, config6>(layer5_out, layer6_out, w6, b6); // q_dense_2

    nnet::relu<q_dense_2_result_t, layer7_t, relu_config7>(layer6_out, layer7_out); // q_dense_2_quantized_relu

    nnet::dense<layer7_t, q_dense_3_result_t, config8>(layer7_out, layer8_out, w8, b8); // q_dense_3

    nnet::relu<q_dense_3_result_t, layer9_t, relu_config9>(layer8_out, layer9_out); // q_dense_3_quantized_relu

    nnet::dense<layer9_t, q_dense_4_result_t, config10>(layer9_out, layer10_out, w10, b10); // q_dense_4

    nnet::dense<q_dense_4_result_t, q_dense_6_result_t, config12>(layer10_out, layer12_out, w12, b12); // q_dense_6

    nnet::relu<q_dense_6_result_t, layer13_t, relu_config13>(layer12_out, layer13_out); // q_dense_6_quantized_relu

    nnet::dense<layer13_t, q_dense_7_result_t, config14>(layer13_out, layer14_out, w14, b14); // q_dense_7

    nnet::relu<q_dense_7_result_t, layer15_t, relu_config15>(layer14_out, layer15_out); // q_dense_7_quantized_relu

    nnet::dense<layer15_t, q_dense_8_result_t, config16>(layer15_out, layer16_out, w16, b16); // q_dense_8

    nnet::subtract<q_dense_8_result_t, layer5_t, subtract_result_t, config18>(layer16_out, layer5_out, layer18_out); // subtract

    nnet::dot1d<subtract_result_t, subtract_result_t, result_t, config19>(layer18_out, layer18_out, layer19_out); // dot

}

}
