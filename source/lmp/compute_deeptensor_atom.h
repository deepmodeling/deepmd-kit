// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(deeptensor/atom, ComputeDeeptensorAtom)
// clang-format on
#else

#ifndef LMP_COMPUTE_DEEPTENSOR_ATOM_H
#define LMP_COMPUTE_DEEPTENSOR_ATOM_H

#include "compute.h"
#include "pair_deepmd.h"
#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DeepTensor.h"
#else
#include "deepmd/DeepTensor.h"
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

namespace LAMMPS_NS {

class ComputeDeeptensorAtom : public Compute {
 public:
  ComputeDeeptensorAtom(class LAMMPS *, int, char **);
  ~ComputeDeeptensorAtom() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;
  void init_list(int, class NeighList *) override;
  double dist_unit_cvt_factor;

 private:
  int nmax;
  double **tensor;
  PairDeepMD dp;
  class NeighList *list;
  deepmd_compat::DeepTensor dt;
  std::vector<int> sel_types;
};

}  // namespace LAMMPS_NS

#endif
#endif
