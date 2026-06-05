// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS

PairStyle(deepmd, PairDeepMD)

#else

#ifndef LMP_PAIR_NNP_H
#define LMP_PAIR_NNP_H

#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DeepPot.h"
#else
#include "deepmd/DeepPot.h"
#endif
namespace deepmd_compat = deepmd;
#else
#ifdef LMPPLUGIN
#include "deepmd.hpp"
#else
#include "deepmd/deepmd.hpp"
#endif
namespace deepmd_compat = deepmd::hpp;
#endif

#include <fstream>
#include <iostream>
#include <map>

#include "comm_brick.h"
#include "pair_base.h"
#define FLOAT_PREC double

namespace LAMMPS_NS {
class CommBrickDeepMD : public CommBrick {
  friend class PairDeepMD;
};
class PairDeepMD : public PairDeepBaseModel {
 public:
  PairDeepMD(class LAMMPS*);
  ~PairDeepMD() override;
  void settings(int, char**) override;
  void coeff(int, char**) override;
  void compute(int, int) override;
  void* extract(const char*, int&) override;
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;
  const std::vector<double>& current_fparam() const { return fparam; }
  double eval_energy_with_fparam(const std::vector<double>& fparam_override);
  double latest_dedn() const { return cached_dedn; }
  bool has_latest_dedn() const { return cached_dedn_valid; }

 protected:
  deepmd_compat::DeepPot deep_pot;
  deepmd_compat::DeepPotModelDevi deep_pot_model_devi;

 private:
  CommBrickDeepMD* commdata_;
  double cached_dedn;
  int cached_dedn_valid;
};

}  // namespace LAMMPS_NS

#endif
#endif
