#include "GTADModel_v6_project.h"
#include "data_types.h"

namespace GTADModel_v6 {

typedef ap_fixed<10,10> offset_t;
typedef ap_fixed<5,5>  shift_t;

static const offset_t NN_OFFSETS[57] = { 16, 1, 64, 8, 32, 64, 2, 32, 64, 2, 32, 64, 1, 32, 64, 4, 128, 256, 2, 128, 256, 2, 128, 256, 2, 128, 256, 16, 64, 64, 16, 64, 64, 16, 64, 64, 8, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64, 16, 64, 64 };
static const shift_t  NN_SHIFTS[57]  = { 28, 0, 72, 14, 0, 71, 9, 2, 71, 8, 2, 71, 6, 0, 72, 10, -2, 286, 6, 0, 284, 6, 0, 286, 6, -2, 285, 62, 0, 71, 48, 0, 71, 42, 0, 71, 37, -2, 72, 32, 0, 72, 24, -2, 72, 22, 0, 72, 21, 0, 71, 20, -2, 72, 20, 0, 71 };

static void scaleNNInputs(
    input_t unscaled[57],
    input_t scaled[57]
) {
    #pragma HLS pipeline
    for (int i = 0; i < 57; i++) {
        #pragma HLS unroll
        input_t tmp0 = unscaled[i] - NN_OFFSETS[i];
        input_t tmp1 = tmp0 >> NN_SHIFTS[i];
        scaled[i] = tmp1;
    }
}

void GTADModel_v6_GT(
    Muon      muons[4],
    Jet       jets[10],
    EGamma    egammas[4],
    Tau       taus[0],
    ET        et,
    HT        ht,
    ETMiss    etmiss,
    HTMiss    htmiss,
    ETHFMiss  ethfmiss,
    HTHFMiss  hthfmiss,
    result_t layer19_out[1]
) {
    #pragma HLS aggregate variable=muons compact=bit
    #pragma HLS aggregate variable=jets compact=bit
    #pragma HLS aggregate variable=egammas compact=bit
    #pragma HLS aggregate variable=taus compact=bit
    #pragma HLS aggregate variable=et compact=bit
    #pragma HLS aggregate variable=ht compact=bit
    #pragma HLS aggregate variable=etmiss compact=bit
    #pragma HLS aggregate variable=htmiss compact=bit
    #pragma HLS aggregate variable=ethfmiss compact=bit
    #pragma HLS aggregate variable=hthfmiss compact=bit

    #pragma HLS array_partition variable=muons complete
    #pragma HLS array_partition variable=jets complete
    #pragma HLS array_partition variable=egammas complete
    #pragma HLS array_partition variable=taus complete

    #pragma HLS pipeline II=1
    #pragma HLS latency min=2 max=2
    #pragma HLS inline recursive

    input_t input_unscaled[57];
    input_t input_scaled[57];
    int idx = 0;

    // Scalars / global objects FIRST
    
    input_unscaled[idx++] = etmiss.et;
    
    input_unscaled[idx++] = etmiss.phi;
    

    // EGammas
    for (int i = 0; i < 4; i++) {
        #pragma HLS unroll
        
        input_unscaled[idx++] = egammas[i].et;
        
        input_unscaled[idx++] = egammas[i].eta;
        
        input_unscaled[idx++] = egammas[i].phi;
        
    }

    // Muons
    for (int i = 0; i < 4; i++) {
        #pragma HLS unroll
        
        input_unscaled[idx++] = muons[i].pt;
        
        input_unscaled[idx++] = muons[i].eta_extrapolated;
        
        input_unscaled[idx++] = muons[i].phi_extrapolated;
        
    }

    // Taus
    for (int i = 0; i < 0; i++) {
        #pragma HLS unroll
        
        input_unscaled[idx++] = taus[i].et;
        
        input_unscaled[idx++] = taus[i].eta;
        
        input_unscaled[idx++] = taus[i].phi;
        
    }

    // Jets
    for (int i = 0; i < 10; i++) {
        #pragma HLS unroll
        
        input_unscaled[idx++] = jets[i].et;
        
        input_unscaled[idx++] = jets[i].eta;
        
        input_unscaled[idx++] = jets[i].phi;
        
    }

    scaleNNInputs(input_unscaled, input_scaled);

    GTADModel_v6_project(input_scaled, layer19_out);
}

} // namespace GTADModel_v6