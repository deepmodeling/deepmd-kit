// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifdef PAIR_CLASS
// clang-format off
PairStyle(dpa4spin,PairDPA4Spin);
// clang-format on
#else

#ifndef LMP_PAIR_DPA4SPIN_H
#define LMP_PAIR_DPA4SPIN_H

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

#include <string>
#include <vector>

#include "comm_brick.h"
#include "pair.h"

namespace LAMMPS_NS {

// Opens the ghost-swap metadata that CommBrick keeps protected. The host
// neighbor-list interface of the model carries it so that a domain-decomposed
// run can describe its halo to the backend.
class CommBrickDPA4Spin : public CommBrick {
  friend class PairDPA4Spin;
};

// Host pair style for native-spin models, where the magnetic moment enters the
// descriptor as an equivariant input and there are no virtual atoms. Either
// graph lower the scheme defines is served; the artifact declares which one
// and the backend branches on it.
//
// Coordinates and moments are marshaled to the host neighbor-list interface of
// the model, which returns the per-atom force, magnetic force, energy and
// virial. Under domain decomposition the model reports extended (local plus
// ghost) force and magnetic force, both of which the spin atom style folds
// onto their owners through its reverse communication.
//
// The style evaluates exactly one artifact and passes no frame, atomic or
// charge/spin parameters, so a model that requires one must carry its default.
//
// The device-resident variant is ``dpa4spin/kk``; it needs the compact
// canonical artifact and evaluates it without the per-step host marshaling.
class PairDPA4Spin : public Pair {
 public:
  PairDPA4Spin(class LAMMPS*);
  ~PairDPA4Spin() override;

  // Load the artifact named by ``pair_style dpa4spin <model>``.
  void settings(int, char**) override;
  // Resolve the LAMMPS atom types onto the element list of the model.
  void coeff(int, char**) override;
  // Request the neighbor list and reject every setting the scheme cannot
  // serve. The Kokkos variant chains through this method and then adds the
  // requirements of device residency.
  void init_style() override;
  double init_one(int, int) override;
  void compute(int, int) override;
  void* extract(const char*, int&) override;

 protected:
  void allocate();
  // Emit the DeePMD-kit and LAMMPS module banner through the LAMMPS logger.
  void print_summary(const std::string& pre) const;
  // Rank within the shared-memory domain, which selects the accelerator the
  // model binds to.
  int get_node_rank() const;

  deepmd_compat::DeepSpin deep_spin;
  // Per-type-pair prefactor applied to every model output; ``fix adapt``
  // reaches it through extract().
  double** scale;
  // Interaction cutoff of the model, in LAMMPS distance units.
  double cutoff;
  // LAMMPS atom type (zero based) -> model element index, or -1 for a type the
  // model never sees.
  std::vector<int> type_idx_map;
  // LAMMPS unit system relative to the eV / Angstrom system of the model.
  double ener_unit_cvt_factor, dist_unit_cvt_factor, force_unit_cvt_factor;

 private:
  CommBrickDPA4Spin* commdata_;
};

}  // namespace LAMMPS_NS

#endif
#endif
