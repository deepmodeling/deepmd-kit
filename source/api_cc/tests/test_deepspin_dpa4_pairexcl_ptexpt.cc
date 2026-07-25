// SPDX-License-Identifier: LGPL-3.0-or-later
// C++ model-level pair-exclusion seam for the native-spin graph route
// (DeepSpinPTExpt).  Twin of test_deeppot_dpa1_pairexcl_ptexpt.cc, which
// covers the non-spin DeepPotPTExpt seam.
//
// Model-level exclusion is a BUILD-time transform owned by the neighbor-graph
// construction (decision #18/A4): the exported .pt2 lower consumes a
// pre-excluded ``edge_mask`` and never re-applies it, so every external feeder
// must fold it in.  ``DeepSpinPTExpt`` therefore parses ``pair_exclude_types``
// from metadata once in ``init`` and calls ``applyPairExclusion`` in BOTH
// graph branches (cached-nlist and standalone) -- these tests are what proves
// those calls are load-bearing.
//
// The fixture (source/tests/infer/gen_dpa4_spin.py) is deliberately
// anti-vacuous: ``deeppot_dpa4_spin_pairexcl.pt2`` carries
// ``pair_exclude_types=[[0, 1]]`` at the MODEL level with the descriptor's own
// ``exclude_types`` left EMPTY, and shares byte-identical weights with the
// no-exclusion baseline ``deeppot_dpa4_spin_graph.pt2``.  Nothing inside the
// compiled artifact reproduces the mask, so a dead C++ seam changes the
// numbers.  (The ``type="dpa4"`` model alias copies model-level pairs into
// ``descriptor.exclude_types``, which WOULD bake an equivalent mask into the
// artifact and hide exactly this bug -- hence the generic
// ``type="standard"``-style config in the generator.)
//
// Two assertions per ingestion branch:
//   1. C++ == the Python DeepEval reference for the SAME archive (1e-10),
//      i.e. the seam applies the exclusion the way Python does;
//   2. excluded != baseline, i.e. the exclusion is genuinely active.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "DeepSpin.h"
#include "DeepSpinPTExpt.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

// Spin models need relaxed epsilon (same bound as
// test_deepspin_dpa4_graph_ptexpt.cc).
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-10 : 1e-4)

namespace {
constexpr const char* kExclModel =
    "../../tests/infer/deeppot_dpa4_spin_pairexcl.pt2";
constexpr const char* kExclRef =
    "../../tests/infer/deeppot_dpa4_spin_pairexcl.expected";
constexpr const char* kBaseModel =
    "../../tests/infer/deeppot_dpa4_spin_graph.pt2";
}  // namespace

template <class VALUETYPE>
class TestInferDeepSpinDpa4PairExclPtExpt : public ::testing::Test {
 protected:
  // 6-atom system (3 Ni spin-active, 3 O non-magnetic) -- verbatim from
  // gen_dpa4_spin.py's _COORDS/_CELL/_SPINS/_ATYPES.  With
  // pair_exclude_types=[[0, 1]] every Ni-O edge is dropped, so only the
  // Ni-Ni and O-O interactions survive.
  std::vector<VALUETYPE> coord = {1.0, 1.0, 1.0, 3.2, 1.4, 1.1, 1.3, 1.8, 1.0,
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

  int natoms;
  double expected_tot_e;

  deepmd::DeepSpin dp_excl;
  deepmd::DeepSpin dp_base;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT_SPIN
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    std::ifstream excl_file(kExclModel);
    std::ifstream base_file(kBaseModel);
    if (!excl_file.good() || !base_file.good()) {
      GTEST_SKIP() << "Skip because the native-spin DPA4 fixtures were not "
                      "generated (run source/tests/infer/gen_dpa4_spin.py).";
    }
    dp_excl.init(kExclModel);
    dp_base.init(kBaseModel);

    deepmd_test::ExpectedRef ref;
    ref.load(kExclRef);
    expected_e = ref.get<VALUETYPE>("pbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("pbc", "expected_f");
    expected_fm = ref.get<VALUETYPE>("pbc", "expected_fm");
    expected_tot_v = ref.get<VALUETYPE>("pbc", "expected_tot_v");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 3, expected_fm.size());
    EXPECT_EQ(9, expected_tot_v.size());
    expected_tot_e = 0.;
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepSpinDpa4PairExclPtExpt, ValueTypes);

// Standalone branch: DeepSpinPTExpt::compute -> buildGraphTensors ->
// applyPairExclusion.
TYPED_TEST(TestInferDeepSpinDpa4PairExclPtExpt, cpu_build_nlist) {
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

  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  this->dp_excl.compute(ener, force, force_mag, virial, coord, spin, atype,
                        box);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON)
      << "model-level pair_exclude_types is dropped on the DeepSpinPTExpt "
         "standalone graph seam";
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

// Cached-nlist (LAMMPS) branch: DeepSpinPTExpt::compute_inner ->
// compactEdgeTensors -> applyPairExclusion.
TYPED_TEST(TestInferDeepSpinDpa4PairExclPtExpt, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_fm = this->expected_fm;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;

  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  // All-pairs nlist: the model's own rcut cut and the exclusion must both be
  // applied at the ingestion seam.
  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  this->dp_excl.compute(ener, force, force_mag, virial, coord, spin, atype, box,
                        0, inlist, 0);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON)
      << "model-level pair_exclude_types is dropped on the DeepSpinPTExpt "
         "cached-nlist graph seam";
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
}

// Anti-vacuity: the two archives share weights and differ ONLY by the
// model-level exclusion, so equal predictions would mean the exclusion never
// reached the graph build and both comparisons above would pass for the wrong
// reason.
TYPED_TEST(TestInferDeepSpinDpa4PairExclPtExpt, excluded_differs_from_baseline) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  int& natoms = this->natoms;

  double ener_excl, ener_base;
  std::vector<VALUETYPE> f_excl, fm_excl, v_excl;
  std::vector<VALUETYPE> f_base, fm_base, v_base;
  this->dp_excl.compute(ener_excl, f_excl, fm_excl, v_excl, coord, spin, atype,
                        box);
  this->dp_base.compute(ener_base, f_base, fm_base, v_base, coord, spin, atype,
                        box);

  EXPECT_GT(fabs(ener_excl - ener_base), 1e-6)
      << "excluding every Ni-O pair left the energy unchanged; the fixture or "
         "the exclusion seam is vacuous";
  double max_df = 0.;
  for (int ii = 0; ii < natoms * 3; ++ii) {
    max_df = std::max(max_df, static_cast<double>(fabs(f_excl[ii] - f_base[ii])));
  }
  EXPECT_GT(max_df, 1e-6)
      << "excluding every Ni-O pair left the forces unchanged";
}
