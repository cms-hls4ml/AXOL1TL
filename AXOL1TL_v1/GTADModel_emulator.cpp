#include "firmware/GTADModel.h"
#include "../emulator.h"
#include "firmware/nnet_utils/nnet_common.h"
#include <any>
#include "ap_fixed.h"

//type notes
//AD_NN_OUT_SQ_T = resultsq_t = typedef ap_ufixed<18,14> resultsq_t;
//AD_NN_IN_T = input_t = typedef ap_fixed<8, 6, AP_RND_CONV, AP_SAT> input_t;
//AD_NN_OUT_T = result_t = typedef ap_fixed<10,7> result_t;
//unscaled_t = typedef ap_fixed<18,13> unscaled_t;
//resultsq_t = typedef ap_ufixed<18,14> resultsq_t;

//AD_NNNOUTPUTS = N_LAYER_6 = 13
//AD_NNNINPUTS = N_INPUT_1_1 = 57

class GTADModel_model : public HLS4MLModel {

private:
    input_t _input[N_INPUT_1_1];
    result_t _result[N_LAYER_6];

public: 
  virtual void prepare_input(std::any input) {
    // input_t *input_p = std::any_cast<input_t*>(input);
    unscaled_t *unscaled_input_p = std::any_cast<unscaled_t*>(input);
    unscaled_t unscaled_input[N_INPUT_1_1];
    input_t scaled_input[N_INPUT_1_1];
    
    //first get unscaled inputs
    for (int i = 0; i < N_INPUT_1_1; i++) {
      // _input[i] = std::any_cast<input_t>(input_p[i]);
      unscaled_input[i] = std::any_cast<unscaled_t>(unscaled_input_p[i]);
    }

    //scale inputs
    scaleNNInputs(unscaled_input, scaled_input);

    //then fill scaled inputs
    for (int i = 0; i < N_INPUT_1_1; i++) { 
      // _input[i] = std::any_cast<input_t>(scaled_input[i]) 
      _input[i] = scaled_input[i] //dont think any_cast needed here but correct me if I'm wrong
    }
    
  }
    
  virtual void predict() {
    GTADModel(_input, _result);
  }
  
  virtual void read_result(std::any result) {
    result_t *result_p = std::any_cast<result_t*>(result);
    for (int i = 0; i < N_LAYER_6; i++) {
      result_p[i] = _result[i];
    }
  }

  //scaleNNInputs function from https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/anomaly_detection_trigger/firmware/hls/anomaly_detection/anomaly_detection.cpp#L28
  virtual void scaleNNInputs(unscaled_t unscaled[N_INPUT_1_1], input_t scaled[N_INPUT_1_1])
  {
    for (int i = 0; i < N_INPUT_1_1; i++)
      {
	unscaled_t tmp0 = unscaled[i] - ad_offsets[i];
	input_t tmp1 = tmp0 >> ad_shift[i];
	scaled[i] = tmp1;
      }
  }


  // computeLoss function from https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/anomaly_detection_trigger/firmware/hls/anomaly_detection/anomaly_detection.cpp#L7
  virtual resultsq_t computeLoss(std::any result) { 
      result_t *result_p = std::any_cast<result_t*>(result); 
      resultsq_t squares[N_LAYER_6];
      resultsq_t square_sum;

      for (int i = 0; i < N_LAYER_6; i++) { 
	result_p[i] = _result[i];
	resultsq_t sq  = result_p[i] * result_p[i];
	squares[i] = sq;
      }
      nnet::Op_add<resultsq_t> op;
      square_sum = nnet::reduce<resultsq_t, N_LAYER_6, nnet::Op_add<resultsq_t>>(squares, op);
      return square_sum;
  }
};

extern "C" hls4mlEmulator::Model* create_model() {
    return new GTADModel_model;
}

extern "C" void destroy_model(hls4mlEmulator::Model* m) {
    delete m;
}

