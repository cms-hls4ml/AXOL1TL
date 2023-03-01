//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "GTADModel.h"
#include "parameters.h"

void GTADModel(
    input_t _input[N_INPUT_1_1],
    result_t _result[N_LAYER_6]
) {

    //hls-fpga-machine-learning insert IO
    // #pragma HLS ARRAY_RESHAPE variable=fc1_input complete dim=0
    // #pragma HLS ARRAY_PARTITION variable=layer16_out complete dim=0
    // #pragma HLS INTERFACE ap_vld port=fc1_input,layer16_out 
    // #pragma HLS PIPELINE 
    #pragma HLS ARRAY_RESHAPE variable=_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=_result complete dim=0
    #pragma HLS INTERFACE ap_vld port=_input,_result
    #pragma HLS PIPELINE
  
#ifdef LOAD_WEIGHTS_FROM_TXT
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 1824>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight4_t, 512>(w4, "w4.txt");
        nnet::load_weights_from_txt<bias4_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<weight6_t, 208>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 13>(b6, "b6.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers
 
    layer2_t layer2_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0
    nnet::dense<input_t, layer2_t, config2>(_input, layer2_out, w2, b2); // q_dense

    layer3_t layer3_out[N_LAYER_2];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::relu<layer2_t, layer3_t, relu_config3>(layer2_out, layer3_out); // q_dense_quantized_relu

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4); // q_dense_1

    layer5_t layer5_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // q_dense_1_quantized_relu

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // mu

    nnet::linear<layer6_t, result_t, linear_config7>(layer6_out, _result); // mu_quantized_bits

}
