#include "NN/GTADModel_v1.h"
#include "emulator.h"
#include "NN/nnet_utils/nnet_common.h"
#include <any>
#include <array>
#include <utility>
#include "ap_fixed.h"
#include "ap_int.h"
#include "scales.h"

class GTADModel_emulator_v1 : public hls4mlEmulator::Model {

private:
    unscaled_t _unscaled_input[N_INPUT_1_1];
    input_t _scaled_input[N_INPUT_1_1];
    result_t _result[N_LAYER_6];
    resultsq_t _loss;

    // scaleNNInputs function from
  // https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/anomaly_detection_trigger/firmware/hls/anomaly_detection/anomaly_detection.cpp#L28
  virtual void _scaleNNInputs(unscaled_t unscaled[N_INPUT_1_1], input_t scaled[N_INPUT_1_1])
  {
    for (int i = 0; i < N_INPUT_1_1; i++)
      {
	unscaled_t tmp0 = unscaled[i] - ad_offsets[i];
	input_t tmp1 = tmp0 >> ad_shift[i];
	scaled[i] = tmp1;
      }
  }

  // computeLoss function from
  // https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/anomaly_detection_trigger/firmware/hls/anomaly_detection/anomaly_detection.cpp#L7
  virtual resultsq_t _computeLoss(result_t result_p[N_LAYER_6]) {
      resultsq_t squares[N_LAYER_6];
      resultsq_t square_sum;

      for (int i = 0; i < N_LAYER_6; i++) {
	squares[i]  = result_p[i] * result_p[i];
      }
      nnet::Op_add<resultsq_t> op;
      square_sum = nnet::reduce<resultsq_t, N_LAYER_6, nnet::Op_add<resultsq_t>>(squares, op);
      return square_sum;
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
    GTADModel_v1(_scaled_input, _result);
    _loss = _computeLoss(_result);
  }
  
  virtual void read_result(std::any result) {
    // return results as an std::pair
    // first = reconstructed output
    // second = loss
    std::pair<std::array<result_t, N_LAYER_6>, resultsq_t> *result_p = std::any_cast<std::pair<std::array<result_t, N_LAYER_6>, resultsq_t>*>(result);
    for (int i = 0; i < N_LAYER_6; i++) {
      result_p->first[i] = _result[i];
    }
    result_p->second = _loss;
  }
  
};

extern "C" hls4mlEmulator::Model* create_model() {
    return new GTADModel_emulator_v1;
}

extern "C" void destroy_model(hls4mlEmulator::Model* m) {
    delete m;
}

