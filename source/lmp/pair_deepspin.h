// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS

PairStyle(deepspin, PairDeepSpin)

#else

#ifndef LMP_PAIR_NNP_SPIN_H
#define LMP_PAIR_NNP_SPIN_H

#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DeepSpin.h"
#else
#include "deepmd/DeepSpin.h"
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
class CommBrickDeepSpin : public CommBrick {
  friend class PairDeepSpin;
};
class PairDeepSpin : public PairDeepBaseModel {
 public:
  PairDeepSpin(class LAMMPS *);
  ~PairDeepSpin() override;
  void settings(int, char **) override;
  void coeff(int, char **) override;
  void compute(int, int) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;

 protected:
  deepmd_compat::DeepSpin deep_spin;
  deepmd_compat::DeepSpinModelDevi deep_spin_model_devi;
  std::vector<std::vector<double> > all_force_mag;

 private:
  CommBrickDeepSpin *commdata_;
};

}  // namespace LAMMPS_NS

#endif
#endif
