// SPDX-License-Identifier: LGPL-3.0-or-later
// C++ inference for a DPA4 model with analytical ZBL bridging (pt_expt
// .pt2, graph lower).
//
// ``bridging_method: ZBL`` builds a COMPOSITION -- LinearEnergyModel over
// [learned DPA4, InnerPotentialAtomicModel] -- so this exercises the graph
// lower of a linear composition, which no other C++ fixture covers.  Before
// this test ZBL bridging had NO C++ or LAMMPS coverage at all: its only
// end-to-end check drove the archive through the PYTHON DeepPot, which
// never reaches DeepPotPTExpt.
//
// Single-rank only by construction: bridging enables the descriptor's
// Source Freeze Propagation Gate, whose per-node eta folds a node's full
// outgoing-edge set, and edges exist only for owned centres -- so the
// generator asserts has_comm_artifact=false and multi-rank is out of scope
// here.
//
// The references come from the Python DeepEval of the SAME archive
// (source/tests/infer/gen_dpa4_zbl.py), so a 1e-10 match validates the
// whole C++ chain: metadata parse, graph ingestion, and the compiled math
// of the composition.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "DeepPot.h"
// Defines BUILD_PT_EXPT (via its __has_include probe for the inductor
// headers).  Without this include the guards below see it undefined and
// every case GTEST_SKIPs with "PyTorch support is not enabled" -- silently,
// which is how this suite first ran as 8 skips that looked like passes.
#include "DeepPotPTExpt.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

namespace {
constexpr const char* kModelPath =
    "../../tests/infer/deeppot_dpa4_zbl_graph.pt2";
constexpr const char* kRefPath =
    "../../tests/infer/deeppot_dpa4_zbl_graph.expected";

// Magnitude-scaled bound.  The analytical ZBL term on a 0.9 A pair makes
// this fixture's forces ~1.4e3 -- deliberately, so the term cannot be
// mistaken for noise -- and ONE fp32 ULP at that magnitude is 2^-13 =
// 1.22e-4.  The suite's flat 1e-4 float bound therefore asks for sub-ULP
// agreement, which no fp32 result can satisfy: the observed deltas were
// exact binary fractions (2^-13, 3*2^-14, 6*2^-14), i.e. 1-3 ULP of the
// representation, not numerical error.  fp64 keeps the strict absolute
// 1e-10 (one fp64 ULP here is 2.3e-13, so 1e-10 is still ~400x tighter
// than the representation).
template <class VALUETYPE>
inline double zbl_tol(double expected) {
  if (std::is_same<VALUETYPE, double>::value) {
    return 1e-10;
  }
  // 5e-7 relative is ~4 fp32 ULP; the 1e-4 floor keeps near-zero
  // components at the suite's usual float bound.
  return 1e-4 + 5e-7 * fabs(expected);
}
}  // namespace

template <class VALUETYPE>
class TestInferDeepPotDpa4ZblPtExpt : public ::testing::Test {
 protected:
  // Fixed 6-atom system -- verbatim from gen_dpa4_zbl.py's _COORDS/_ATYPES.
  // Atoms 0 and 1 sit 0.9 A apart, inside bridging_r_outer, so the
  // analytical ZBL term dominates their interaction.
  std::vector<VALUETYPE> coord = {1.0, 1.0, 1.0, 1.9, 1.0, 1.0, 1.3, 1.8, 1.0,
                                  0.4, 1.2, 1.6, 3.6, 2.0, 1.3, 3.4, 0.7, 1.7};
  std::vector<int> atype = {0, 0, 0, 1, 1, 1};
  std::vector<VALUETYPE> box = {6., 0., 0., 0., 6., 0., 0., 0., 6.};

  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_tot_v;
  std::vector<VALUETYPE> expected_e_nopbc;
  std::vector<VALUETYPE> expected_f_nopbc;

  int natoms;
  double expected_tot_e;
  double expected_tot_e_nopbc;

  deepmd::DeepPot dp;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    std::ifstream model_file(kModelPath);
    if (!model_file.good()) {
      GTEST_SKIP() << "Skip because " << kModelPath
                   << " was not generated (run "
                      "source/tests/infer/gen_dpa4_zbl.py).";
    }
    dp.init(kModelPath);

    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("pbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("pbc", "expected_f");
    expected_tot_v = ref.get<VALUETYPE>("pbc", "expected_tot_v");
    expected_e_nopbc = ref.get<VALUETYPE>("nopbc", "expected_e");
    expected_f_nopbc = ref.get<VALUETYPE>("nopbc", "expected_f");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(9, expected_tot_v.size());
    expected_tot_e = 0.;
    expected_tot_e_nopbc = 0.;
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
      expected_tot_e_nopbc += expected_e_nopbc[ii];
    }
  };

  void TearDown() override {};
};

TYPED_TEST_SUITE(TestInferDeepPotDpa4ZblPtExpt, ValueTypes);

TYPED_TEST(TestInferDeepPotDpa4ZblPtExpt, type_map) {
  std::string type_map;
  this->dp.get_type_map(type_map);
  EXPECT_EQ(type_map, "Ni O");
}

// Standalone path (no InputNlist): DeepPotPTExpt::compute -> the graph
// branch, on a linear composition.
TYPED_TEST(TestInferDeepPotDpa4ZblPtExpt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;

  double ener;
  std::vector<VALUETYPE> force, virial;
  this->dp.compute(ener, force, virial, coord, atype, box);

  EXPECT_EQ(force.size(), natoms * 3);
  // anti-vacuity: the close Ni-Ni pair must drive a large ZBL repulsion,
  // else the fixture would be degenerate and this comparison meaningless.
  double fmax = 0.;
  for (int ii = 0; ii < natoms * 3; ++ii) {
    fmax = std::max(fmax, static_cast<double>(fabs(force[ii])));
  }
  EXPECT_GT(fmax, 1e-3) << "forces are trivially small; fixture is vacuous";

  EXPECT_LT(fabs(ener - expected_tot_e), zbl_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_tol<VALUETYPE>(expected_f[ii]));
  }
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]),
              zbl_tol<VALUETYPE>(expected_tot_v[ii]));
  }
}

TYPED_TEST(TestInferDeepPotDpa4ZblPtExpt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;

  double ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  this->dp.compute(ener, force, virial, atom_ener, atom_vir, coord, atype, box);

  EXPECT_EQ(atom_ener.size(), natoms);
  EXPECT_LT(fabs(ener - expected_tot_e), zbl_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]),
              zbl_tol<VALUETYPE>(expected_e[ii]));
  }
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_tol<VALUETYPE>(expected_f[ii]));
  }
}

// LAMMPS path (explicit InputNlist, nghost=0): the gas-phase system, so it
// is compared against the NoPBC reference.
TYPED_TEST(TestInferDeepPotDpa4ZblPtExpt, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& expected_f = this->expected_f_nopbc;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e_nopbc;
  std::vector<VALUETYPE> box = {};

  double ener;
  std::vector<VALUETYPE> force, virial;
  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  this->dp.compute(ener, force, virial, coord, atype, box, 0, inlist, 0);

  EXPECT_LT(fabs(ener - expected_tot_e), zbl_tol<VALUETYPE>(expected_tot_e));
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]),
              zbl_tol<VALUETYPE>(expected_f[ii]));
  }
}
