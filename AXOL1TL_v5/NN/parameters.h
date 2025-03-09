#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_merge.h"
 
//hls-fpga-machine-learning insert weights
#include "defines.h"
#include "weights.h"

namespace hls4ml_axol1tl_v5 {

// q_dense
struct config2 : nnet::dense_config {
    static const unsigned n_in = 57;
    static const unsigned n_out = 29;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 547;
    static const unsigned n_nonzeros = 1106;
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
    static const unsigned n_in = 29;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_quantized_relu_table_t table_t;
};

// q_dense_1
struct config4 : nnet::dense_config {
    static const unsigned n_in = 29;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 55;
    static const unsigned n_nonzeros = 235;
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
    static const unsigned n_in = 10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_1_quantized_relu_table_t table_t;
};

// q_dense_2
struct config6 : nnet::dense_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 9;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 13;
    static const unsigned n_nonzeros = 77;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_2_accum_t accum_t;
    typedef bias6_t bias_t;
    typedef weight6_t weight_t;
    typedef layer6_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_2_quantized_relu
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 9;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_2_quantized_relu_table_t table_t;
};

// q_dense_3
struct config8 : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 52;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_3_accum_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef layer8_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_3_quantized_relu
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = 6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_3_quantized_relu_table_t table_t;
};

// q_dense_4
struct config10 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 4;
    static const unsigned n_nonzeros = 20;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_4_accum_t accum_t;
    typedef bias10_t bias_t;
    typedef weight10_t weight_t;
    typedef layer10_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_4_linear
struct linear_config11 : nnet::activ_config {
    static const unsigned n_in = 4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_4_linear_table_t table_t;
};

// q_dense_6
struct config12 : nnet::dense_config {
    static const unsigned n_in = 4;
    static const unsigned n_out = 6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2;
    static const unsigned n_nonzeros = 22;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_6_accum_t accum_t;
    typedef bias12_t bias_t;
    typedef weight12_t weight_t;
    typedef layer12_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_6_quantized_relu
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_6_quantized_relu_table_t table_t;
};

// q_dense_7
struct config14 : nnet::dense_config {
    static const unsigned n_in = 6;
    static const unsigned n_out = 9;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 8;
    static const unsigned n_nonzeros = 46;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_7_accum_t accum_t;
    typedef bias14_t bias_t;
    typedef weight14_t weight_t;
    typedef layer14_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_7_quantized_relu
struct relu_config15 : nnet::activ_config {
    static const unsigned n_in = 9;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_7_quantized_relu_table_t table_t;
};

// q_dense_8
struct config16 : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 18;
    static const unsigned n_nonzeros = 72;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef q_dense_8_accum_t accum_t;
    typedef bias16_t bias_t;
    typedef weight16_t weight_t;
    typedef layer16_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_8_linear
struct linear_config17 : nnet::activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef q_dense_8_linear_table_t table_t;
};

// subtract
struct config18 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_16;
};

// dot
struct config19 : nnet::dot_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    typedef dot_default_t accum_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

} // namespace hls4ml_axol1tl_v5

#endif
