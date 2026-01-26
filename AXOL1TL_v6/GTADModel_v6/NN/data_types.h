#ifndef L1GT_DATATYPES
#define L1GT_DATATYPES

#include "ap_fixed.h"
#include <cmath>

// Author: sioni@cern.ch August 2022
// Object data types defined in https://github.com/cms-l1-globaltrigger/mp7_ugt_legacy/blob/master/firmware/hdl/packages/gtl_pkg.vhd
// Code mimics Phase 2 L1Trigger Particle Flow data types HLS:
// https://github.com/cms-sw/cmssw/tree/master/DataFormats/L1TParticleFlow/interface

typedef ap_fixed<9,2> cossin_t;
typedef ap_fixed<13,7> sinh_t;

static constexpr int N_TABLE = 2048;

/* ---
 * Constants useful for converting physical units to hardware integers 
 * --- */
namespace Scales{
  static const double MUON_PHI_LSB = 2 * M_PI / 576;
  static const double CALO_PHI_LSB = 2 * M_PI / 144;

  static const double MUON_ETA_LSB = 0.0870 / 8;
  static const double CALO_ETA_LSB = 0.0870 / 2;

  static const double MUON_PT_LSB = 0.5;
  static const double CALO_PT_LSB = 0.5;
  
  const int INTPHI_PI = 720;

  static const int MUON_HALF_PI = 144;
  static const int CALO_HALF_PI = 36;
}; // namespace Scales

/* ---
 * Functions for packing and unpacking ap_ objects 
 * --- */
template <typename U, typename T>
inline void pack_into_bits(U& u, unsigned int& start, const T& data) {
  const unsigned int w = T::width;
  u(start + w - 1, start) = data(w - 1, 0);
  start += w;
}

template <typename U, typename T>
inline void unpack_from_bits(const U& u, unsigned int& start, T& data) {
  const unsigned int w = T::width;
  data(w - 1, 0) = u(start + w - 1, start);
  start += w;
}

/* ---
 * Definitions of the objects received by the GT
 * --- */
/* ---
 * Muon 
 * --- */
struct Muon{
  ap_uint<10> phi_extrapolated;
  ap_ufixed<9,8> pt;
  ap_uint<4> quality;
  ap_int<9> eta_extrapolated;
  ap_uint<2> iso;
  ap_uint<1> charge_sign;
  ap_uint<1> charge_valid;
  ap_uint<7> index_bits;
  ap_uint<10> phi_out;
  ap_uint<8> pt_unconstrained;
  ap_uint<1> hadronic_shower_trigger;
  ap_uint<2> impact_parameter;

  static const int BITWIDTH = 64;

  inline void clear(){
    phi_extrapolated = 0;
    pt = 0;
    quality = 0;
    eta_extrapolated = 0;
    iso = 0;
    charge_sign = 0;
    charge_valid = 0;
    index_bits = 0;
    phi_out = 0;
    pt_unconstrained = 0;
    hadronic_shower_trigger = 0;
    impact_parameter = 0;
  }

  inline ap_uint<BITWIDTH> pack() const{
    ap_uint<BITWIDTH> ret;
    unsigned int start = 0;
    pack_into_bits(ret, start, phi_extrapolated);
    pack_into_bits(ret, start, pt);
    pack_into_bits(ret, start, quality);
    pack_into_bits(ret, start, eta_extrapolated);
    pack_into_bits(ret, start, iso);
    pack_into_bits(ret, start, charge_sign);
    pack_into_bits(ret, start, charge_valid);
    pack_into_bits(ret, start, index_bits);
    pack_into_bits(ret, start, phi_out);
    pack_into_bits(ret, start, pt_unconstrained);
    pack_into_bits(ret, start, hadronic_shower_trigger);
    pack_into_bits(ret, start, impact_parameter);
    return ret;
  }

  inline void initFromBits(const ap_uint<BITWIDTH> &src){
    unsigned int start = 0;
    unpack_from_bits(src, start, phi_extrapolated);
    unpack_from_bits(src, start, pt);
    unpack_from_bits(src, start, quality);
    unpack_from_bits(src, start, eta_extrapolated);
    unpack_from_bits(src, start, iso);
    unpack_from_bits(src, start, charge_sign);
    unpack_from_bits(src, start, charge_valid);
    unpack_from_bits(src, start, index_bits);
    unpack_from_bits(src, start, phi_out);
    unpack_from_bits(src, start, pt_unconstrained);
    unpack_from_bits(src, start, hadronic_shower_trigger);
    unpack_from_bits(src, start, impact_parameter);
  }

  inline static Muon unpack(const ap_uint<BITWIDTH> &src){
    Muon ret;
    ret.initFromBits(src);
    return ret;
  }

  inline static Muon initFromPhysicalDoubles(const double pt, const double eta, const double phi){
    Muon ret;
    ret.clear();
    ret.pt = pt;
    ret.eta_extrapolated = round(eta / Scales::MUON_ETA_LSB);
    ret.phi_extrapolated = round(phi / Scales::MUON_PHI_LSB);
    return ret;
  }

  inline static Muon initFromHWInt(int pt, int eta, int phi){
      Muon muon;
      muon.clear();
      muon.pt.V = pt;
      muon.eta_extrapolated.V = eta;
      muon.phi_extrapolated.V = phi;
      return muon;
  }

}; // struct Muon

/* ---
 * Jet 
 * --- */
struct Jet{
  ap_ufixed<11,10> et;
  ap_int<8> eta;
  ap_uint<8> phi;
  ap_uint<1> disp;
  ap_uint<2> quality;
  ap_uint<2> spare;

  static const int BITWIDTH = 32;

  inline void clear(){
    et = 0;
    eta = 0;
    phi = 0;
    disp = 0;
    quality = 0;
    spare = 0;
  }

  inline ap_uint<BITWIDTH> pack() const{
    ap_uint<BITWIDTH> ret;
    unsigned int start = 0;
    pack_into_bits(ret, start, et);
    pack_into_bits(ret, start, eta);
    pack_into_bits(ret, start, phi);
    pack_into_bits(ret, start, disp);
    pack_into_bits(ret, start, quality);
    pack_into_bits(ret, start, spare);
    return ret;
  }

  inline void initFromBits(const ap_uint<BITWIDTH> &src){
    unsigned int start = 0;
    unpack_from_bits(src, start, et);
    unpack_from_bits(src, start, eta);
    unpack_from_bits(src, start, phi);
    unpack_from_bits(src, start, disp);
    unpack_from_bits(src, start, quality);
    unpack_from_bits(src, start, spare);
  }

  inline static Jet unpack(const ap_uint<BITWIDTH> &src){
    Jet ret;
    ret.initFromBits(src);
    return ret;
  }

  inline static Jet initFromPhysicalDoubles(const double et, const double eta, const double phi){
    Jet ret;
    ret.clear();
    ret.et = et;
    ret.eta = round(eta / Scales::CALO_ETA_LSB);
    ret.phi = round(phi / Scales::CALO_PHI_LSB);
    return ret;
  }

  inline static Jet initFromHWInt(const int et, const int eta, const int phi){
    Jet ret;
    ret.clear();
    ret.et.V = et;
    ret.eta.V = eta;
    ret.phi.V = phi;
    return ret;
  }

}; // struct Jet

/* ---
 * e / gamma or tau (same format)
 * --- */
struct CaloCommon{
  ap_ufixed<9,8> et;
  ap_int<8> eta;
  ap_uint<8> phi;
  ap_uint<2> iso;
  ap_uint<5> spare;

  static const int BITWIDTH = 32;

  inline void clear(){
    et = 0;
    eta = 0;
    phi = 0;
    iso = 0;
    spare = 0;
  }

  inline ap_uint<BITWIDTH> pack() const{
    ap_uint<BITWIDTH> ret;
    unsigned int start = 0;
    pack_into_bits(ret, start, et);
    pack_into_bits(ret, start, eta);
    pack_into_bits(ret, start, phi);
    pack_into_bits(ret, start, iso);
    pack_into_bits(ret, start, spare);
    return ret;
  }

  inline void initFromBits(const ap_uint<BITWIDTH> &src){
    unsigned int start = 0;
    unpack_from_bits(src, start, et);
    unpack_from_bits(src, start, eta);
    unpack_from_bits(src, start, phi);
    unpack_from_bits(src, start, iso);
    unpack_from_bits(src, start, spare);
  }

  inline static CaloCommon unpack(const ap_uint<BITWIDTH> &src){
    CaloCommon ret;
    ret.initFromBits(src);
    return ret;
  }

  inline static CaloCommon initFromPhysicalDoubles(const double et, const double eta, const double phi){
    CaloCommon ret;
    ret.clear();
    ret.et = et;
    ret.eta = round(eta / Scales::CALO_ETA_LSB);
    ret.phi = round(phi / Scales::CALO_PHI_LSB);
    return ret;
  }

  inline static CaloCommon initFromHWInt(const int et, const int eta, const int phi){
    CaloCommon ret;
    ret.clear();
    ret.et.V = et;
    ret.eta.V = eta;
    ret.phi.V = phi;
    return ret;
  }

}; // struct CaloCommon

typedef CaloCommon EGamma;
typedef CaloCommon Tau;

/* ---
 * Scalar Sums
 * --- */
struct ET{
  ap_ufixed<12,11> et;
  ap_ufixed<12,11> ettem;
  ap_uint<4> spare;
  ap_uint<4> minimum_bias_hf;

  static const int BITWIDTH = 32;

  inline void clear(){
    et = 0;
    ettem = 0;
    spare = 0;
    minimum_bias_hf = 0;
  }

  inline ap_uint<BITWIDTH> pack() const{
    ap_uint<BITWIDTH> ret;
    unsigned int start = 0;
    pack_into_bits(ret, start, et);
    pack_into_bits(ret, start, ettem);
    pack_into_bits(ret, start, spare);
    pack_into_bits(ret, start, minimum_bias_hf);
    return ret;
  }

  inline void initFromBits(const ap_uint<BITWIDTH> &src){
    unsigned int start = 0;
    unpack_from_bits(src, start, et);
    unpack_from_bits(src, start, ettem);
    unpack_from_bits(src, start, spare);
    unpack_from_bits(src, start, minimum_bias_hf);
  }

  inline static ET unpack(const ap_uint<BITWIDTH> &src){
    ET ret;
    ret.initFromBits(src);
    return ret;
  }

  inline static ET initFromPhysicalDoubles(const double et, const double ettem){
    ET ret;
    ret.clear();
    ret.et = et;
    ret.ettem = ettem;
    return ret;
  }

  inline static ET initFromHWInt(const int et, const int ettem){
    ET ret;
    ret.clear();
    ret.et.V = et;
    ret.ettem.V = ettem;
    return ret;
  }
}; // struct ET

struct HT{
  ap_ufixed<12,11> et;
  ap_uint<13> tower_count;
  ap_uint<3> spare;
  ap_uint<4> minimum_bias_hf;

  static const int BITWIDTH = 32;

  inline void clear(){
    et = 0;
    tower_count = 0;
    spare = 0;
    minimum_bias_hf = 0;
  }

  inline ap_uint<BITWIDTH> pack() const{
    ap_uint<BITWIDTH> ret;
    unsigned int start = 0;
    pack_into_bits(ret, start, et);
    pack_into_bits(ret, start, tower_count);
    pack_into_bits(ret, start, spare);
    pack_into_bits(ret, start, minimum_bias_hf);
    return ret;
  }

  inline void initFromBits(const ap_uint<BITWIDTH> &src){
    unsigned int start = 0;
    unpack_from_bits(src, start, et);
    unpack_from_bits(src, start, tower_count);
    unpack_from_bits(src, start, spare);
    unpack_from_bits(src, start, minimum_bias_hf);
  }

  inline static HT unpack(const ap_uint<BITWIDTH> &src){
    HT ret;
    ret.initFromBits(src);
    return ret;
  }

  inline static HT initFromPhysicalDoubles(const double et){
    HT ret;
    ret.clear();
    ret.et = et;
    return ret;
  }

  inline static HT initFromHWInt(const int et){
    HT ret;
    ret.clear();
    ret.et.V = et;
    return ret;
  }
}; // struct HT

/* ---
 * Vector Sums
 * --- */
struct VectorSumsCommon{
  ap_ufixed<12,11> et;
  ap_uint<8> phi;
  ap_uint<8> asy;
  ap_uint<4> other;

  static const int BITWIDTH = 32;

  inline void clear(){
    et = 0;
    phi = 0;
    asy = 0;
    other = 0;
  }

  inline ap_uint<BITWIDTH> pack() const{
    ap_uint<BITWIDTH> ret;
    unsigned int start = 0;
    pack_into_bits(ret, start, et);
    pack_into_bits(ret, start, phi);
    pack_into_bits(ret, start, asy);
    pack_into_bits(ret, start, other);
    return ret;
  }

  inline void initFromBits(const ap_uint<BITWIDTH> &src){
    unsigned int start = 0;
    unpack_from_bits(src, start, et);
    unpack_from_bits(src, start, phi);
    unpack_from_bits(src, start, asy);
    unpack_from_bits(src, start, other);
  }

  inline static VectorSumsCommon unpack(const ap_uint<BITWIDTH> &src){
    VectorSumsCommon ret;
    ret.initFromBits(src);
    return ret;
  }

  inline static VectorSumsCommon initFromPhysicalDoubles(const double et, const double phi){
    VectorSumsCommon ret;
    ret.clear();
    ret.et = et;
    ret.phi = round(phi / Scales::CALO_PHI_LSB);
    return ret;
  } 

  inline static VectorSumsCommon initFromHWInt(const int et, const int phi){
    VectorSumsCommon ret;
    ret.clear();
    ret.et.V = et;
    ret.phi.V = phi;
    return ret;
  }  
}; // struct VectorSumsCommon

typedef VectorSumsCommon ETMiss;
typedef VectorSumsCommon HTMiss;
typedef VectorSumsCommon ETHFMiss;
typedef VectorSumsCommon HTHFMiss;

static const int NMUONS = 8;
static const int NJETS = 12;
static const int NEGAMMAS = 12;
static const int NTAUS = 12;

/* ---
 * Definitions of common objects used for ML triggers
 * TODO: this is a first implementation, to be improved & expanded
 * TODO: these data types for px, py, pz are not optimized
 * --- */
typedef ap_fixed<18,13> unscaled_t;
struct PxPyPz{

  unscaled_t px;
  unscaled_t py;
  unscaled_t pz;


  static const int BITWIDTH = 36;

  inline void clear(){
    px = 0;
    py = 0;
    pz = 0;
  }

  inline ap_uint<BITWIDTH> pack() const{
    ap_uint<BITWIDTH> ret;
    unsigned int start = 0;
    pack_into_bits(ret, start, px);
    pack_into_bits(ret, start, py);
    pack_into_bits(ret, start, pz);
    return ret;
  }

  inline void initFromBits(const ap_uint<BITWIDTH> &src){
    unsigned int start = 0;
    unpack_from_bits(src, start, px);
    unpack_from_bits(src, start, py);
    unpack_from_bits(src, start, pz);
  }

  inline static PxPyPz unpack(const ap_uint<BITWIDTH> &src){
    PxPyPz ret;
    ret.initFromBits(src);
    return ret;
  }

  inline static PxPyPz initFromPhysicalDoubles(const double px, const double py, const double pz){
    PxPyPz ret;
    ret.clear();
    ret.px = px;
    ret.py = py;
    ret.pz = pz;
    return ret;
  }  
}; // struct PxPyPz

#endif