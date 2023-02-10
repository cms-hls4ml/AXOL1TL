#include "firmware/anomaly_detection.h"
#include "../emulator.h"
#include "firmware/nnet_utils/nnet_common.h"
#include <any>
#include "ap_fixed.h"

class anomaly_detection_model : public HLS4MLModel {
private:
    input_t _input[N_INPUT_1_1];
    result_t _result[N_LAYER_6];
public:
  virtual void prepare_input(std::any input) {
    input_t *input_p = std::any_cast<input_t*>(input);
    for (int i = 0; i < N_INPUT_1_1; i++) {
      _input[i] = std::any_cast<input_t>(input_p[i]);
    }
  }
    
  virtual void predict() {
    anomaly_detection(_input, _result);
  }
  
  virtual void read_result(std::any result) {
    result_t *result_p = std::any_cast<result_t*>(result);
    for (int i = 0; i < N_LAYER_6; i++) {
      result_p[i] = _result[i];
    }
  }


  // virtual void computeLoss(std::any result, std::any square_sum) { 
  //     result_t *squares_p = std::any_cast<result_t*>(result); 
  //     resultsq_t *square_sum_p = std::any_cast<resultsq_t*>(square_sum); 

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
      // square_sum = square_sum_p;
  }
};

extern "C" HLS4MLModel* create_model() {
    return new anomaly_detection_model;
}

extern "C" void destroy_model(HLS4MLModel* m) {
    delete m;
}

