#include "emulator.h"
#include <iostream>
#include "ap_fixed.h"
//#include "path/to/anomaly_detection_defs.h" // Note that this is optional if you don't know the types (unlikely)

int main() {
    
    ModelLoader loader = ModelLoader("./anomaly_detection");
    // ModelLoader loader = ModelLoader("./jet_tagger");
    
    HLS4MLModel* model = loader.load_model();
    
    ap_fixed<8, 6, AP_RND_CONV, AP_SAT> input[57] = {
        -0.12011719, 0.40527344, -1.04101562, -0.82519531, -0.75585938, -0.58105469, 1.98632812, 1.53710938, 1.98632812, 0.63085938, 0.3828125, -0.20214844, 1.05957031, 0.40820312, -1.02050781, -0.18066406
    };
    ap_fixed<10,7> result[13];
    model->prepare_input(input);
    model->predict();
    model->read_result(result);
    std::cout << "Emulated predictions: [ ";
    for (int i = 0; i < 5; i++) {
        std::cout << result[i].to_float() << ", ";
    }
    std::cout << "]" << std::endl;
    
    loader.destroy_model();

}
