// SPDX-License-Identifier: LGPL-3.0-or-later
// C++ graph-route regression for numb_aparam == 2 with DISTINCT per-component
// values (gen_dpa1.py section C).
//
// Two historical silent-corruption bugs on this path:
//  1. select_real_atoms_coord under-sized the selected aparam buffer (missing
//     the daparam factor): heap OOB write + a (1, nloc, 1)-shaped tensor whose
//     single column then BROADCAST over all daparam columns in the padding
//     copy_ -- finite but wrong energy/force.  A numb_aparam=1 fixture can
//     never expose this; the distinct column values here do.
//  2. The graph ABI carries aparam FLAT on the node axis (N, daparam);
//     extend_graph_aparam normalizes the rectangular runtime tensor and
//     validates the width instead of broadcasting.
//
// Reference values (deeppot_dpa1_graph_aparam.expected) come from an
// INDEPENDENT nlist (dense-quartet) evaluation of the same weights with the
// same aparam, so a match validates the graph aparam marshalling, not just
// self-consistency.  gen_dpa1.py asserts the fixture's aparam columns are
// non-degenerate (column swap shifts the energy), so broadcast corruption
// cannot pass vacuously.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

namespace {
constexpr const char* kGraphModel =
    "../../tests/infer/deeppot_dpa1_graph_aparam.pt2";
constexpr const char* kRefPath =
    "../../tests/infer/deeppot_dpa1_graph_aparam.expected";
}  // namespace

template <class VALUETYPE>
class TestInferDpa1GraphAparamPtExpt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  // numb_aparam == 2, DISTINCT per-atom AND per-component values; must match
  // ``aparam_vals`` in gen_dpa1.py section C (row-major per atom).
  std::vector<VALUETYPE> aparam = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                   0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  std::vector<VALUETYPE> fparam = {};
  // Per-atom reference (energy/force/virial) from the .expected sidecar.
  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_v;
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  static deepmd::DeepPot dp;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp.init(kGraphModel);
#endif
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("pbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("pbc", "expected_f");
    expected_v = ref.get<VALUETYPE>("pbc", "expected_v");

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, static_cast<int>(expected_f.size()));
    EXPECT_EQ(natoms * 9, static_cast<int>(expected_v.size()));
    expected_tot_e = 0.;
    expected_tot_v.assign(9, 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override {};

  static void TearDownTestSuite() { dp = deepmd::DeepPot(); }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDpa1GraphAparamPtExpt<VALUETYPE>::dp;

TYPED_TEST_SUITE(TestInferDpa1GraphAparamPtExpt, ValueTypes);

// Case 1: standalone graph branch (DeepPot builds its own neighbor list).
// Exercises the single-rank extend_graph_aparam normalization
// (nlocal == N: rectangular (1, natoms, 2) -> flat (N, 2)).
TYPED_TEST(TestInferDpa1GraphAparamPtExpt, cpu_build_nlist_aparam2) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box, fparam, aparam);

  EXPECT_EQ(force.size(), static_cast<size_t>(natoms * 3));
  EXPECT_EQ(virial.size(), 9u);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

// Case 2: the LAMMPS provided-nlist graph branch (compute_inner).  The
// single-rank folded graph has N == nloc, so aparam covers every node; the
// distinct column values guard the width handling on this route too.
TYPED_TEST(TestInferDpa1GraphAparamPtExpt, lammps_nlist_aparam2) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;

  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  // The graph branch folds ghost neighbours onto their local owners via the
  // LAMMPS atom-map; without it periodic (ghost) edges would be dropped.
  inlist.mapping = mapping.data();

  double ener;
  std::vector<VALUETYPE> force_, virial;
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}
