#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"
 
//hls-fpga-machine-learning insert weights
#include "defines.h"
#include "weights.h"

namespace hls4ml_axol1tl_v4 {

// q_dense
struct config2 : nnet::dense_config {
    static const unsigned n_in = 57;
    static const unsigned n_out = 28;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 766;
    static const unsigned n_nonzeros = 830;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_accum_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef layer2_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_quantized_relu
struct relu_config3 : nnet::activ_config {
    static const unsigned n_in = 28;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_quantized_relu_table_t table_t;
};

// q_dense_1
struct config4 : nnet::dense_config {
    static const unsigned n_in = 28;
    static const unsigned n_out = 15;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 59;
    static const unsigned n_nonzeros = 361;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_1_accum_t accum_t;
    typedef bias4_t bias_t;
    typedef weight4_t weight_t;
    typedef layer4_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_1_quantized_relu
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_1_quantized_relu_table_t table_t;
};

// mu
struct config6 : nnet::dense_config {
    static const unsigned n_in = 15;
    static const unsigned n_out = 8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 70;
    static const unsigned n_nonzeros = 50;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef mu_accum_t accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    typedef layer6_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// mu_quantized_bits
struct linear_config7 : nnet::activ_config {
    static const unsigned n_in = 8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef mu_quantized_bits_table_t table_t;
};

} // namespace hls4ml_axol1tl_v4

#endif