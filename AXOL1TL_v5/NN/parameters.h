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
struct config3 : nnet::dense_config {
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
    typedef q_dense_bias_t bias_t;
    typedef q_dense_weight_t weight_t;
    typedef layer3_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 29;
    static const unsigned table_size = 2097152;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_table_t table_t;
};

// q_dense_1
struct config7 : nnet::dense_config {
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
    typedef q_dense_1_bias_t bias_t;
    typedef q_dense_1_weight_t weight_t;
    typedef layer7_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation_1
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 131072;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_1_table_t table_t;
};

// q_dense_2
struct config11 : nnet::dense_config {
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
    typedef q_dense_2_bias_t bias_t;
    typedef q_dense_2_weight_t weight_t;
    typedef layer11_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation_2
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 9;
    static const unsigned table_size = 65536;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_2_table_t table_t;
};

// q_dense_3
struct config15 : nnet::dense_config {
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
    typedef q_dense_3_bias_t bias_t;
    typedef q_dense_3_weight_t weight_t;
    typedef layer15_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation_3
struct relu_config17 : nnet::activ_config {
    static const unsigned n_in = 6;
    static const unsigned table_size = 65536;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_3_table_t table_t;
};

// q_dense_4
struct config19 : nnet::dense_config {
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
    typedef q_dense_4_bias_t bias_t;
    typedef q_dense_4_weight_t weight_t;
    typedef layer19_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// q_dense_6
struct config21 : nnet::dense_config {
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
    typedef q_dense_6_bias_t bias_t;
    typedef q_dense_6_weight_t weight_t;
    typedef layer21_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation_4
struct relu_config23 : nnet::activ_config {
    static const unsigned n_in = 6;
    static const unsigned table_size = 1048576;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_4_table_t table_t;
};

// q_dense_7
struct config25 : nnet::dense_config {
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
    typedef q_dense_7_bias_t bias_t;
    typedef q_dense_7_weight_t weight_t;
    typedef layer25_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// activation_5
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 9;
    static const unsigned table_size = 65536;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_5_table_t table_t;
};

// q_dense_8
struct config29 : nnet::dense_config {
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
    typedef q_dense_8_bias_t bias_t;
    typedef q_dense_8_weight_t weight_t;
    typedef layer29_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// subtract
struct config31 : nnet::merge_config {
    static const unsigned n_elem = N_LAYER_29;
    static const unsigned reuse_factor = 1;
};

// dot
struct config32 : nnet::dot_config {
    static const unsigned n_in = 10;
    static const unsigned n_out = 1;
    static const unsigned reuse_factor = 1;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in, reuse_factor);
    typedef dot_accum_t accum_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

} // namespace hls4ml_axol1tl_v5

#endif
