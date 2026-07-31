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
  void init_style() override;
  double init_one(int, int) override;
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;
  double eval_energy_with_fparam(const std::vector<double>& fparam_override);
  bool compact_selection_enabled() const { return compact_selection_enabled_; }

 protected:
  deepmd_compat::DeepPot deep_pot;
  deepmd_compat::DeepPotModelDevi deep_pot_model_devi;
  // Assemble the send/recv swap metadata (a comm-only neighbor list; its
  // geometry fields are unused) for the device-resident message-passing path,
  // where ghost features are exchanged across ranks inside the forward pass.
  deepmd_compat::InputNlist make_comm_nlist();

 private:
  // Compact evaluation is implemented by assigning type -1 to atoms outside
  // the selected subsystem. Every supported DeepPot backend already compacts
  // such atoms, remaps its neighbor/communication data, and scatters outputs
  // back to the original atom order.
  bool compact_selection_enabled_;
  bool compact_include_molecule_;
  bool compact_center_group_dynamic_;
  int compact_center_group_bit_;
  double compact_environment_cutoff_;
  bigint compact_natoms_;
  std::string compact_center_group_id_;
  std::vector<tagint> compact_center_tags_;
  std::vector<unsigned char> compact_selected_;

  std::vector<tagint> allgather_unique_tagints(
      std::vector<tagint> local_values) const;
  void refresh_compact_center_tags();
  bool apply_compact_selection(std::vector<int>& model_types);
  void analyze_model_deviation(double& max,
                               double& min,
                               double& sum,
                               const std::vector<double>& deviation,
                               int nlocal) const;

  CommBrickDeepMD* commdata_;
};

}  // namespace LAMMPS_NS

#endif
#endif
