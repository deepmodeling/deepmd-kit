// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef FIX_CLASS

FixStyle(dplr, FixDPLR)

#else

#ifndef LMP_FIX_DPLR_H
#define LMP_FIX_DPLR_H

#include <stdio.h>

#include <map>

#include "fix.h"
#include "pair_deepmd.h"
#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DataModifier.h"
#include "DeepTensor.h"
#else
#include "deepmd/DataModifier.h"
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

#define FLOAT_PREC double

namespace LAMMPS_NS {
class FixDPLR : public Fix {
 public:
  FixDPLR(class LAMMPS *, int, char **);
  ~FixDPLR() override;
  int setmask() override;
  void init() override;
  void setup(int) override;
  void setup_pre_force(int) override;
  void min_setup(int) override;
  void post_integrate() override;
  void pre_force(int) override;
  void post_force(int) override;
  void min_pre_exchange() override;
  void min_pre_force(int) override;
  void min_post_force(int) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double compute_scalar(void) override;
  double compute_vector(int) override;
  double ener_unit_cvt_factor, dist_unit_cvt_factor, force_unit_cvt_factor;

 private:
  PairDeepMD *pair_deepmd;
  deepmd_compat::DeepTensor dpt;
  deepmd_compat::DipoleChargeModifier dtm;
  std::string model;
  int ntypes;
  std::vector<int> sel_type;
  std::vector<int> dpl_type;
  std::vector<int> bond_type;
  std::map<int, int> type_asso;
  std::map<int, int> bk_type_asso;
  std::vector<FLOAT_PREC> dipole_recd;
  std::vector<double> dfcorr_buff;
  std::vector<double> efield;
  std::vector<double> efield_fsum, efield_fsum_all;
  int efield_force_flag;
  void get_valid_pairs(std::vector<std::pair<int, int> > &pairs);
  int varflag;
  char *xstr, *ystr, *zstr;
  int xvar, yvar, zvar, xstyle, ystyle, zstyle;
  double qe2f;
  void update_efield_variables();
  enum { NONE, CONSTANT, EQUAL };
  std::vector<int> type_idx_map;
};
}  // namespace LAMMPS_NS

#endif  // LMP_FIX_DPLR_H
#endif  // FIX_CLASS
