//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

//from https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/master/ugt_hls/src/anomaly_detection/NN/VAE_HLS.h

#ifndef GTADMODEL_V4_H_
#define GTADMODEL_V4_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#include "defines.h"

namespace hls4ml_axol1tl_v4 {

//from https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/axol1tl_v4/ugt_hls/src/anomaly_detection/Axol1tl_v4.h
void GTADModel_v4(
	       hls4ml_axol1tl_v4::input_t input_3[hls4ml_axol1tl_v4::N_INPUT_1_1],
	       hls4ml_axol1tl_v4::result_t layer7_out[hls4ml_axol1tl_v4::N_LAYER_6]
	       );

} // namespace hls4ml_axol1tl_v4

#endif
