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

#include <array>
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
  int pack_reverse_comm(int, int, double*) override;
  void unpack_reverse_comm(int, int*, double*) override;
  double eval_energy_with_fparam(const std::vector<double>& fparam_override);

 protected:
  deepmd_compat::DeepPot deep_pot;
  deepmd_compat::DeepPotModelDevi deep_pot_model_devi;
  // Return whether the legacy ensemble was initialized, so settings only
  // forwards conditions to a live ensemble. Device styles may own models
  // separately while retaining deep_pot as their driving model.
  virtual bool initialize_models(const std::vector<std::string>& models);
  // Assemble the send/recv swap metadata (a comm-only neighbor list; its
  // geometry fields are unused) for the device-resident message-passing path,
  // where ghost features are exchanged across ranks inside the forward pass.
  deepmd_compat::InputNlist make_comm_nlist();
  bool model_deviation_step() const;
  // Report native-unit model outputs after the caller folds ghost forces.
  // This only reads all_force and never changes the MD driving forces.
  void write_model_deviation(
      const std::vector<std::vector<double> >& all_virial);
  // Final native-unit statistics, in file-column order. Optional atomic
  // deviations are in local atom order; this helper only converts and writes.
  void write_model_deviation_output(const std::array<double, 6>& deviation,
                                    const std::vector<double>& std_f);

 private:
  CommBrickDeepMD* commdata_;
};

}  // namespace LAMMPS_NS

#endif
#endif
