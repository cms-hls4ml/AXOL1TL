#include <iostream>

#include "GTADModel_v5.h"
#include "parameters.h"
#include "defines.h"

namespace hls4ml_axol1tl_v5 {

void GTADModel_v5(
    input_t input_2[N_INPUT_1_1],
    result_t layer32_out[OUT_DOT_32]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer32_out complete dim=0
    #pragma HLS PIPELINE 

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    q_dense_t layer3_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::dense<input_t, q_dense_t, config3>(input_2, layer3_out, w3, b3); // q_dense

    activation_t layer5_out[N_LAYER_3];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<q_dense_t, activation_t, relu_config5>(layer3_out, layer5_out); // activation

    q_dense_1_t layer7_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::dense<activation_t, q_dense_1_t, config7>(layer5_out, layer7_out, w7, b7); // q_dense_1

    activation_1_t layer9_out[N_LAYER_7];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0
    nnet::relu<q_dense_1_t, activation_1_t, relu_config9>(layer7_out, layer9_out); // activation_1

    q_dense_2_t layer11_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::dense<activation_1_t, q_dense_2_t, config11>(layer9_out, layer11_out, w11, b11); // q_dense_2

    activation_2_t layer13_out[N_LAYER_11];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::relu<q_dense_2_t, activation_2_t, relu_config13>(layer11_out, layer13_out); // activation_2

    q_dense_3_t layer15_out[N_LAYER_15];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    nnet::dense<activation_2_t, q_dense_3_t, config15>(layer13_out, layer15_out, w15, b15); // q_dense_3

    activation_3_t layer17_out[N_LAYER_15];
    #pragma HLS ARRAY_PARTITION variable=layer17_out complete dim=0
    nnet::relu<q_dense_3_t, activation_3_t, relu_config17>(layer15_out, layer17_out); // activation_3

    q_dense_4_t layer19_out[N_LAYER_19];
    #pragma HLS ARRAY_PARTITION variable=layer19_out complete dim=0
    nnet::dense<activation_3_t, q_dense_4_t, config19>(layer17_out, layer19_out, w19, b19); // q_dense_4

    q_dense_6_t layer21_out[N_LAYER_21];
    #pragma HLS ARRAY_PARTITION variable=layer21_out complete dim=0
    nnet::dense<q_dense_4_t, q_dense_6_t, config21>(layer19_out, layer21_out, w21, b21); // q_dense_6

    activation_4_t layer23_out[N_LAYER_21];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0
    nnet::relu<q_dense_6_t, activation_4_t, relu_config23>(layer21_out, layer23_out); // activation_4

    q_dense_7_t layer25_out[N_LAYER_25];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0
    nnet::dense<activation_4_t, q_dense_7_t, config25>(layer23_out, layer25_out, w25, b25); // q_dense_7

    activation_5_t layer27_out[N_LAYER_25];
    #pragma HLS ARRAY_PARTITION variable=layer27_out complete dim=0
    nnet::relu<q_dense_7_t, activation_5_t, relu_config27>(layer25_out, layer27_out); // activation_5

    q_dense_8_t layer29_out[N_LAYER_29];
    #pragma HLS ARRAY_PARTITION variable=layer29_out complete dim=0
    nnet::dense<activation_5_t, q_dense_8_t, config29>(layer27_out, layer29_out, w29, b29); // q_dense_8

    subtract_t layer31_out[N_LAYER_29];
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0
    nnet::subtract<q_dense_8_t, activation_1_t, subtract_t, config31>(layer29_out, layer9_out, layer31_out); // subtract

    nnet::dot1d<subtract_t, subtract_t, result_t, config32>(layer31_out, layer31_out, layer32_out); // dot

}

} // namespace hls4ml_axol1tl_v5
