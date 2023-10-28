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

#ifndef GTADMODEL_V3_H_
#define GTADMODEL_V3_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

#include "defines.h"

//from https://gitlab.cern.ch/ssummers/run3_ugt_ml/-/blob/axol1tl_v3/ugt_hls/src/anomaly_detection/Axol1tl_v3.h
void GTADModel_v3(
	       input_t input_3[N_INPUT_1_1],
	       result_t layer7_out[N_LAYER_6]
	       );

#endif
