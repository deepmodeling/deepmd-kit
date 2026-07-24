// SPDX-License-Identifier: LGPL-3.0-or-later
// Test the C++ model-level pair-exclusion ingestion seam of the pt_expt
// backend (Task A3/A4).  Two DPA1(attn_layer=0) models with identical weights
// and model-level pair_exclude_types=[[0,1]] are exported, one through each C++
// ingestion route:
//   - deeppot_dpa1_pairexcl_graph.pt2 -> applyPairExclusion (graph route)
//   - deeppot_dpa1_pairexcl_nlist.pt2 -> applyPairExclusionNlist (dense route)
// A no-exclusion baseline (deeppot_dpa1_pairexcl_none.pt2, empty exclude table)
// exercises the identity/pre-change branch of both helpers.
//
// Exclusion is a BUILD-time transform (decision #18/A4): the exported .pt2
// lowers consume pre-excluded inputs (graph edge_mask / dense nlist) and never
// re-apply it, so the C++ helpers here are the SINGLE application site — these
// tests are what proves they are load-bearing.  The reference values
// (.expected sidecars) come from the Python DeepEval of the SAME .pt2, so a
// 1e-10 match validates the whole chain (pair_exclude_types metadata
// round-trip + init table build + build-time apply + compiled math).  A
// separate assertion (excluded energy != baseline energy) proves the exclusion
// is genuinely active and not silently dropped.
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "expected_ref.h"
#include "test_utils.h"

namespace {
constexpr const char* kGraphModel =
    "../../tests/infer/deeppot_dpa1_pairexcl_graph.pt2";
constexpr const char* kNlistModel =
    "../../tests/infer/deeppot_dpa1_pairexcl_nlist.pt2";
constexpr const char* kNoneModel =
    "../../tests/infer/deeppot_dpa1_pairexcl_none.pt2";
constexpr const char* kGraphRef =
    "../../tests/infer/deeppot_dpa1_pairexcl_graph.expected";
constexpr const char* kNlistRef =
    "../../tests/infer/deeppot_dpa1_pairexcl_nlist.expected";
}  // namespace

template <class VALUETYPE>
class TestInferDpa1PairExclPtExpt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};

  // Excluded models (one per ingestion route) + no-exclusion baseline.
  static deepmd::DeepPot dp_graph;
  static deepmd::DeepPot dp_nlist;
  static deepmd::DeepPot dp_none;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp_graph.init(kGraphModel);
    dp_nlist.init(kNlistModel);
    dp_none.init(kNoneModel);
#endif
  }

  static void TearDownTestSuite() {
    dp_graph = deepmd::DeepPot();
    dp_nlist = deepmd::DeepPot();
    dp_none = deepmd::DeepPot();
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
  }

  // Load per-atom reference from a .expected sidecar and reduce to totals.
  void load_ref(const char* path,
                double& tot_e,
                std::vector<VALUETYPE>& per_f,
                std::vector<VALUETYPE>& tot_v,
                int& natoms) {
    deepmd_test::ExpectedRef ref;
    ref.load(path);
    const auto per_e = ref.get<VALUETYPE>("pbc", "expected_e");
    per_f = ref.get<VALUETYPE>("pbc", "expected_f");
    const auto per_v = ref.get<VALUETYPE>("pbc", "expected_v");
    natoms = per_e.size();
    tot_e = 0.;
    for (int ii = 0; ii < natoms; ++ii) {
      tot_e += per_e[ii];
    }
    tot_v.assign(9, 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        tot_v[dd] += per_v[ii * 9 + dd];
      }
    }
  }

  // Run one model through the standalone build-nlist path and check it against
  // its Python DeepEval reference at EPSILON.
  void check_against_ref(deepmd::DeepPot& dp, const char* ref_path) {
    double tot_e;
    std::vector<VALUETYPE> per_f, tot_v;
    int natoms;
    load_ref(ref_path, tot_e, per_f, tot_v, natoms);

    double ener;
    std::vector<VALUETYPE> force, virial;
    dp.compute(ener, force, virial, coord, atype, box);

    EXPECT_EQ(force.size(), static_cast<size_t>(natoms * 3));
    EXPECT_EQ(virial.size(), 9u);
    EXPECT_LT(fabs(ener - tot_e), EPSILON);
    for (int ii = 0; ii < natoms * 3; ++ii) {
      EXPECT_LT(fabs(force[ii] - per_f[ii]), EPSILON);
    }
    for (int ii = 0; ii < 9; ++ii) {
      EXPECT_LT(fabs(virial[ii] - tot_v[ii]), EPSILON);
    }
  }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDpa1PairExclPtExpt<VALUETYPE>::dp_graph;
template <class VALUETYPE>
deepmd::DeepPot TestInferDpa1PairExclPtExpt<VALUETYPE>::dp_nlist;
template <class VALUETYPE>
deepmd::DeepPot TestInferDpa1PairExclPtExpt<VALUETYPE>::dp_none;

TYPED_TEST_SUITE(TestInferDpa1PairExclPtExpt, ValueTypes);

// Graph route: applyPairExclusion at the ingestion seam + compiled exclusion.
TYPED_TEST(TestInferDpa1PairExclPtExpt, graph_route_matches_python_ref) {
  this->check_against_ref(this->dp_graph, kGraphRef);
}

// Dense route: applyPairExclusionNlist at the ingestion seam + compiled
// exclusion.
TYPED_TEST(TestInferDpa1PairExclPtExpt, nlist_route_matches_python_ref) {
  this->check_against_ref(this->dp_nlist, kNlistRef);
}

// The two ingestion routes carry the SAME weights and exclusion, so at
// non-binding sel they must agree bit-for-bit (fp64 ~1e-10).
TYPED_TEST(TestInferDpa1PairExclPtExpt, graph_equals_nlist_route) {
  using VALUETYPE = TypeParam;
  double e_g, e_n;
  std::vector<VALUETYPE> f_g, v_g, f_n, v_n;
  this->dp_graph.compute(e_g, f_g, v_g, this->coord, this->atype, this->box);
  this->dp_nlist.compute(e_n, f_n, v_n, this->coord, this->atype, this->box);
  EXPECT_LT(fabs(e_g - e_n), EPSILON);
  ASSERT_EQ(f_g.size(), f_n.size());
  for (size_t ii = 0; ii < f_g.size(); ++ii) {
    EXPECT_LT(fabs(f_g[ii] - f_n[ii]), EPSILON);
  }
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(v_g[ii] - v_n[ii]), EPSILON);
  }
}

// The no-exclusion baseline exercises the EMPTY-table (identity) branch of the
// C++ helpers; it must run cleanly and produce an energy that DIFFERS from the
// excluded models (proving pair_exclude_types is genuinely active, not
// dropped).
TYPED_TEST(TestInferDpa1PairExclPtExpt, exclusion_is_active_vs_baseline) {
  using VALUETYPE = TypeParam;
  double e_none, e_g, e_n;
  std::vector<VALUETYPE> f, v;
  this->dp_none.compute(e_none, f, v, this->coord, this->atype, this->box);
  this->dp_graph.compute(e_g, f, v, this->coord, this->atype, this->box);
  this->dp_nlist.compute(e_n, f, v, this->coord, this->atype, this->box);

  EXPECT_TRUE(std::isfinite(e_none));
  // Excluding all O-H pairs changes the energy well above the fp64 tolerance.
  EXPECT_GT(fabs(e_g - e_none), 1e-6);
  EXPECT_GT(fabs(e_n - e_none), 1e-6);
}
