// SPDX-License-Identifier: LGPL-3.0-or-later
// C++ NeighborGraph inference for a NATIVE-SPIN DPA4 that is ALSO bridged
// with the analytical ZBL term (pt_expt .pt2, graph lower).
//
// The combination is not the conjunction of the two paths already covered by
// test_deepspin_dpa4_graph_ptexpt.cc (native spin) and
// test_deeppot_dpa4_zbl_ptexpt.cc (bridging): here ``spin`` must reach a
// COMPOSITION -- LinearEnergyModel over [learned DPA4,
// InnerPotentialAtomicModel] -- whose learned child consumes it and whose
// analytical child accepts and ignores it, and the archive must still declare
// ``is_spin`` so this DeepSpin path (not DeepPot) is the one that runs.
// Before this test the combination had NO coverage below the pt_expt Python
// layer, and its only export-seam test skips when CI=true.
//
// Single-rank only by construction: bridging enables the descriptor's Source
// Freeze Propagation Gate, whose per-node eta folds a node's full
// outgoing-edge set, and edges exist only for owned centres -- so
// source/tests/infer/gen_dpa4_spin_zbl.py asserts has_comm_artifact=false AND
// that no nested forward_lower_with_comm.pt2 exists.  Multi-rank is therefore
// out of scope here; under mpirun the C++ dispatch fails fast (asserted from
// the LAMMPS side, source/lmp/tests/test_lammps_dpa4_zbl_pt2.py).
//
// The references come from the Python DeepEval of the SAME archive
// (gen_dpa4_spin_zbl.py), which the generator in turn holds to its eager
// dpmodel, so a match here validates the whole C++ chain: metadata parse,
// graph + spin ingestion, and the compiled math of the composition.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

#include "DeepSpin.h"
// Defines BUILD_PT_EXPT_SPIN (via its __has_include probe for the inductor
// headers), which the SetUp guards below read.
#include "DeepSpinPTExpt.h"
// Defines BUILD_PT_EXPT, same probe.  Without these includes the guards see
// the macros undefined and every case GTEST_SKIPs with "PyTorch support is
// not enabled" -- silently, which is how the ZBL suite first ran as 8 skips
// that looked like passes.  The #errors below make that failure mode
// impossible to reintroduce: a missing include is now a build error, not a
// silent skip.
#include "DeepPotPTExpt.h"

#if defined(BUILD_PYTORCH) && !defined(BUILD_PT_EXPT)
#error "BUILD_PT_EXPT undefined: DeepPotPTExpt.h must be included."
#endif
#if defined(BUILD_PYTORCH) && !defined(BUILD_PT_EXPT_SPIN)
#error "BUILD_PT_EXPT_SPIN undefined: DeepSpinPTExpt.h must be included."
#endif

#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

namespace {
constexpr const char* kModelPath =
    "../../tests/infer/deeppot_dpa4_spin_zbl_graph.pt2";
constexpr const char* kRefPath =
    "../../tests/infer/deeppot_dpa4_spin_zbl_graph.expected";

// Magnitude-scaled bound, same reasoning as test_deeppot_dpa4_zbl_ptexpt.cc's
// ``zbl_tol``: the analytical ZBL term on the fixture's 0.9 A Ni-Ni pair
// makes the forces ~1.4e3 and the virial ~1.3e3 (deliberately, so the term
// cannot be mistaken for noise), and ONE fp32 ULP at that magnitude is
// 2^-13 = 1.22e-4.  The suite's flat 1e-4 float bound therefore asks for
// sub-ULP agreement, which no fp32 result can satisfy.  5e-7 relative is
// ~4 fp32 ULP; the 1e-4 floor keeps near-zero components (notably force_mag,
// whose largest entry is ~8e-3) at the suite's usual float bound.
//
// fp64 keeps a strict absolute 1e-10 -- the same bound the sibling native-spin
// graph suite uses, and still ~400x tighter than one fp64 ULP at this
// fixture's force magnitude (2.3e-13).
template <class VALUETYPE>
inline double zbl_spin_tol(double expected) {
  if (std::is_same<VALUETYPE, double>::value) {
    return 1e-10;
  }
  return 1e-4 + 5e-7 * fabs(expected);
}
}  // namespace

// ============================================================================
// PBC test fixture
// ============================================================================

template <class VALUETYPE>
class TestInferDeepSpinDpa4ZblPtExpt : public ::testing::Test {
 protected:
  // 6-atom system (3 Ni, spin-active; 3 O, non-magnetic) -- verbatim from
  // gen_dpa4_spin_zbl.py's _COORDS/_CELL/_SPINS/_ATYPES.  Atoms 0 and 1 sit
  // 0.9 A apart, inside bridging_r_outer, and are BOTH spin-carrying Ni, so
  // the analytical and spin channels act on the same atoms.
  std::vector<VALUETYPE> coord = {1.0, 1.0, 1.0, 1.9, 1.0, 1.0, 1.3, 1.8, 1.0,
                                  0.4, 1.2, 1.6, 3.6, 2.0, 1.3, 3.4, 0.7, 1.7};
  std::vector<VALUETYPE> spin = {0.11,  0.05,  -0.02, -0.07, 0.09,  0.03,
                                 0.02,  -0.06, 0.08,  0.01,  -0.01, 0.02,
                                 -0.02, 0.03,  -0.01, 0.015, 0.02,  -0.03};
  std::vector<int> atype = {0, 0, 0, 1, 1, 1};
  std::vector<VALUETYPE> box = {6., 0., 0., 0., 6., 0., 0., 0., 6.};

  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_fm;
  std::vector<VALUETYPE> expected_tot_v;
  std::vector<VALUETYPE> expected_atom_v;

  int natoms;
  double expected_tot_e;

  deepmd::DeepSpin dp;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT_SPIN
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    std::ifstream model_file(kModelPath);
    if (!model_file.good()) {
      GTEST_SKIP() << "Skip because " << kModelPath
                   << " was not generated (run "
                      "source/tests/infer/gen_dpa4_spin_zbl.py).";
    }
    dp.init(kModelPath);

    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("pbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("pbc", "expected_f");
    expected_fm = ref.get<VALUETYPE>("pbc", "expected_fm");
    expected_tot_v = ref.get<VALUETYPE>("pbc", "expected_tot_v");
    expected_atom_v = ref.get<VALUETYPE>("pbc", "expected_atom_v");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    EXPECT_EQ(9, expected_tot_v.size());
    EXPECT_EQ(natoms * 9, expected_atom_v.size());
    expected_tot_e = 0.;
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepSpinDpa4ZblPtExpt, ValueTypes);

TYPED_TEST(TestInferDeepSpinDpa4ZblPtExpt, test_get_use_spin) {
  deepmd::DeepSpin& dp = this->dp;
  std::vector<bool> use_spin = dp.get_use_spin();
  EXPECT_EQ(use_spin.size(), 2);
  EXPECT_TRUE(use_spin[0]);   // Ni carries a magnetic moment
  EXPECT_FALSE(use_spin[1]);  // O does not
}

TYPED_TEST(TestInferDeepSpinDpa4ZblPtExpt, type_map) {
  std::string type_map;
  this->dp.get_type_map(type_map);
  EXPECT_EQ(type_map, "Ni O");
}

// Standalone path (no InputNlist): exercises DeepSpinPTExpt::compute's
// buildGraphTensors branch on a linear composition.
TYPED_TEST(TestInferDeepSpinDpa4ZblPtExpt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);

  // Anti-vacuity, on the values this run actually produced (not only on the
  // reference): the close Ni-Ni pair must drive a large ZBL repulsion, and
  // the jittered learned half must drive a nonzero force_mag on the Ni atoms
  // while the model's own type gate keeps the O rows exactly zero.
  double fmax = 0.;
  for (int ii = 0; ii < natoms * 3; ++ii) {
    fmax = std::max(fmax, static_cast<double>(fabs(force[ii])));
  }
  EXPECT_GT(fmax, 1e-3) << "forces are trivially small; fixture is vacuous";
  double fm_spin_max = 0.;
  for (int ii = 0; ii < natoms; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      double v = fabs(force_mag[ii * 3 + dd]);
      if (atype[ii] == 0) {
        fm_spin_max = std::max(fm_spin_max, v);
      } else {
        EXPECT_EQ(force_mag[ii * 3 + dd], static_cast<VALUETYPE>(0))
            << "force_mag must be EXACTLY zero on non-spin (O) atom " << ii;
      }
    }
  }
  EXPECT_GT(fm_spin_max, 1e-6)
      << "force_mag is trivially small on the spin-active atoms";

  EXPECT_LT(fabs(ener - expected_tot_e),
            zbl_spin_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_spin_tol<VALUETYPE>(expected_f[ii]));
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]),
              zbl_spin_tol<VALUETYPE>(expected_fm[ii]));
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_tot_v[ii]));
  }
}

TYPED_TEST(TestInferDeepSpinDpa4ZblPtExpt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  std::vector<VALUETYPE>& expected_atom_v = this->expected_atom_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(atom_ener.size(), natoms);

  EXPECT_LT(fabs(ener - expected_tot_e),
            zbl_spin_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_spin_tol<VALUETYPE>(expected_f[ii]));
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]),
              zbl_spin_tol<VALUETYPE>(expected_fm[ii]));
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_tot_v[ii]));
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]),
              zbl_spin_tol<VALUETYPE>(expected_e[ii]));
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), natoms * 9);
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_atom_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_atom_v[ii]));
  }
}

// ============================================================================
// NoPBC test fixture
// ============================================================================

template <class VALUETYPE>
class TestInferDeepSpinDpa4ZblPtExptNopbc : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {1.0, 1.0, 1.0, 1.9, 1.0, 1.0, 1.3, 1.8, 1.0,
                                  0.4, 1.2, 1.6, 3.6, 2.0, 1.3, 3.4, 0.7, 1.7};
  std::vector<VALUETYPE> spin = {0.11,  0.05,  -0.02, -0.07, 0.09,  0.03,
                                 0.02,  -0.06, 0.08,  0.01,  -0.01, 0.02,
                                 -0.02, 0.03,  -0.01, 0.015, 0.02,  -0.03};
  std::vector<int> atype = {0, 0, 0, 1, 1, 1};
  std::vector<VALUETYPE> box = {};

  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_fm;
  std::vector<VALUETYPE> expected_tot_v;
  std::vector<VALUETYPE> expected_atom_v;

  int natoms;
  double expected_tot_e;

  deepmd::DeepSpin dp;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT_SPIN
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    std::ifstream model_file(kModelPath);
    if (!model_file.good()) {
      GTEST_SKIP() << "Skip because " << kModelPath
                   << " was not generated (run "
                      "source/tests/infer/gen_dpa4_spin_zbl.py).";
    }
    dp.init(kModelPath);

    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("nopbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("nopbc", "expected_f");
    expected_fm = ref.get<VALUETYPE>("nopbc", "expected_fm");
    expected_tot_v = ref.get<VALUETYPE>("nopbc", "expected_tot_v");
    expected_atom_v = ref.get<VALUETYPE>("nopbc", "expected_atom_v");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    EXPECT_EQ(9, expected_tot_v.size());
    EXPECT_EQ(natoms * 9, expected_atom_v.size());
    expected_tot_e = 0.;
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepSpinDpa4ZblPtExptNopbc, ValueTypes);

TYPED_TEST(TestInferDeepSpinDpa4ZblPtExptNopbc, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_LT(fabs(ener - expected_tot_e),
            zbl_spin_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_spin_tol<VALUETYPE>(expected_f[ii]));
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]),
              zbl_spin_tol<VALUETYPE>(expected_fm[ii]));
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_tot_v[ii]));
  }
}

TYPED_TEST(TestInferDeepSpinDpa4ZblPtExptNopbc, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  std::vector<VALUETYPE>& expected_atom_v = this->expected_atom_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box);

  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_LT(fabs(ener - expected_tot_e),
            zbl_spin_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_spin_tol<VALUETYPE>(expected_f[ii]));
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]),
              zbl_spin_tol<VALUETYPE>(expected_fm[ii]));
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]),
              zbl_spin_tol<VALUETYPE>(expected_e[ii]));
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_tot_v[ii]));
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), natoms * 9);
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_atom_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_atom_v[ii]));
  }
}

// LAMMPS path (explicit InputNlist, nghost=0): exercises DeepSpinPTExpt's
// graph branch under the LAMMPS-nlist overload.  Gas-phase system, so it is
// compared against the NoPBC reference.
TYPED_TEST(TestInferDeepSpinDpa4ZblPtExptNopbc, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;

  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, coord, spin, atype, box, 0, inlist,
             0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_LT(fabs(ener - expected_tot_e),
            zbl_spin_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_spin_tol<VALUETYPE>(expected_f[ii]));
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]),
              zbl_spin_tol<VALUETYPE>(expected_fm[ii]));
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_tot_v[ii]));
  }
}

TYPED_TEST(TestInferDeepSpinDpa4ZblPtExptNopbc, cpu_lmp_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  std::vector<VALUETYPE>& expected_atom_v = this->expected_atom_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  deepmd::DeepSpin& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial, atom_ener, atom_vir;

  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box, 0, inlist, 0);

  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_LT(fabs(ener - expected_tot_e),
            zbl_spin_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_spin_tol<VALUETYPE>(expected_f[ii]));
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]),
              zbl_spin_tol<VALUETYPE>(expected_fm[ii]));
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]),
              zbl_spin_tol<VALUETYPE>(expected_e[ii]));
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), natoms * 9);
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_atom_v[ii]),
              zbl_spin_tol<VALUETYPE>(expected_atom_v[ii]));
  }
}
