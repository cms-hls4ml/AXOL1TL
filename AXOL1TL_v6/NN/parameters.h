#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "defines.h"
#include "weights.h"

namespace hls4ml_axol1tl_v6 {

// hls-fpga-machine-learning insert layer-config
// activation
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = 29;
    static const unsigned table_size = 1048576;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_table_t table_t;
};

// activation_1
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = 10;
    static const unsigned table_size = 131072;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_1_table_t table_t;
};

// activation_2
struct relu_config13 : nnet::activ_config {
    static const unsigned n_in = 9;
    static const unsigned table_size = 65536;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_2_table_t table_t;
};

// activation_3
struct relu_config17 : nnet::activ_config {
    static const unsigned n_in = 6;
    static const unsigned table_size = 65536;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_3_table_t table_t;
};

// activation_4
struct relu_config23 : nnet::activ_config {
    static const unsigned n_in = 6;
    static const unsigned table_size = 1048576;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_4_table_t table_t;
};

// activation_5
struct relu_config27 : nnet::activ_config {
    static const unsigned n_in = 9;
    static const unsigned table_size = 65536;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef activation_5_table_t table_t;
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


} // namespace hls4ml_axol1tl_v6

#endif
