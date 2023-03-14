#ifdef FIX_CLASS

FixStyle(dplr, FixDPLR)

#else

#ifndef LMP_FIX_DPLR_H
#define LMP_FIX_DPLR_H

#include <stdio.h>

#include <map>

#include "fix.h"
#include "pair_deepmd.h"
#ifdef LMPPLUGIN
#include "DataModifier.h"
#include "DeepTensor.h"
#else
#include "deepmd/DataModifier.h"
#include "deepmd/DeepTensor.h"
#endif

#ifdef HIGH_PREC
#define FLOAT_PREC double
#else
#define FLOAT_PREC float
#endif

namespace LAMMPS_NS {
class FixDPLR : public Fix {
 public:
  FixDPLR(class LAMMPS *, int, char **);
  ~FixDPLR() override{};
  int setmask() override;
  void init() override;
  void setup(int) override;
  void post_integrate() override;
  void pre_force(int) override;
  void post_force(int) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  double compute_scalar(void) override;
  double compute_vector(int) override;

 private:
  PairDeepMD *pair_deepmd;
  deepmd::DeepTensor dpt;
  deepmd::DipoleChargeModifier dtm;
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
};
}  // namespace LAMMPS_NS

#endif  // LMP_FIX_DPLR_H
#endif  // FIX_CLASS
