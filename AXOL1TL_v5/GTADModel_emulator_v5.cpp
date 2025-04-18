#include "NN/GTADModel_v5.h"
#include "emulator.h"
#include "NN/nnet_utils/nnet_common.h"
#include <any>
#include <array>
#include <utility>
#include "ap_fixed.h"
#include "ap_int.h"
#include "scales.h"

using namespace hls4ml_axol1tl_v5;

class GTADModel_emulator_v5 : public hls4mlEmulator::Model {

private:
    unscaled_t _unscaled_input[N_INPUT_1_1];
    input_t _scaled_input[N_INPUT_1_1];
    result_t _result[OUT_DOT_32];
    resultsq_t _loss; //identical to result in v5

    // scaleNNInputs function from
  // https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/anomaly_detection_trigger/firmware/hls/anomaly_detection/anomaly_detection.cpp#L28
  //now in https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/master/ugt_hls/src/anomaly_detection/anomaly_detection.cpp
  //unchanged in v5
  virtual void _scaleNNInputs(unscaled_t unscaled[N_INPUT_1_1], input_t scaled[N_INPUT_1_1])
  {
    for (int i = 0; i < N_INPUT_1_1; i++)
    {
      unscaled_t tmp0 = unscaled[i] - hls4ml_axol1tl_v5::ad_offsets[i];
      input_t tmp1 = tmp0 >> hls4ml_axol1tl_v5::ad_shift[i];
      scaled[i] = tmp1;
    }
  }

  // computeLoss function from
  // https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/anomaly_detection_trigger/firmware/hls/anomaly_detection/anomaly_detection.cpp#L7
  // now in https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/master/ugt_hls/src/anomaly_detection/anomaly_detection.cpp#L7
  virtual resultsq_t _computeLoss(result_t result_p[OUT_DOT_19]) {
    //now loss output directly, no mu^2 computed
    resultsq_t loss;
    loss = result_p[0];
    return loss;
  }

public: 
  virtual void prepare_input(std::any input) {
    unscaled_t *unscaled_input_p = std::any_cast<unscaled_t*>(input);
    
    // first get unscaled inputs
    for (int i = 0; i < N_INPUT_1_1; i++) {
      _unscaled_input[i] = std::any_cast<unscaled_t>(unscaled_input_p[i]);
    }

    // scale inputs
    _scaleNNInputs(_unscaled_input, _scaled_input);
  }

  virtual void predict() {
    GTADModel_v5(_scaled_input, _result);
    _loss = _computeLoss(_result);
  }
  
  virtual void read_result(std::any result) {
    // return results as an std::pair
    // first = reconstructed output
    // second = loss
    /* std::pair<std::array<result_t, OUT_DOT_19>, resultsq_t> *result_p = std::any_cast<std::pair<std::array<result_t, OUT_DOT_19>, resultsq_t>*>(result);
    for (int i = 0; i < OUT_DOT_19; i++) {
      result_p->first[i] = _result[i];
     */
    std::pair<result_t, resultsq_t> *result_p = std::any_cast<std::pair<result_t, resultsq_t>*>(result);
    result_p->first = _result[0]; //only 1 result this time, 1d array
    result_p->second = _loss;
  }
};

extern "C" hls4mlEmulator::Model* create_model() {
    return new GTADModel_emulator_v5;
}

extern "C" void destroy_model(hls4mlEmulator::Model* m) {
    delete m;
}
