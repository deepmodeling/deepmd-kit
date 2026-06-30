// SPDX-License-Identifier: LGPL-3.0-or-later
// Test C++ inference for the NeighborGraph (graph-schema) .pt2 path of the
// pt_expt backend.  The graph model is a dpa1(attn_layer=0) descriptor exported
// with lower_kind="graph" (gen_dpa1.py section B); this is the FIRST runtime
// exercise of the C++ graph ingestion added in PR-B Phase B2
// (lower_input_is_graph_ / run_model_graph / buildGraphTensors / the
// compute_inner graph branch).
//
// Reference values (deeppot_dpa1_graph.expected) come from an INDEPENDENT
// nlist (dense-quartet) evaluation of the same weights, so a match validates
// the graph AOTI ABI/geometry, not just self-consistency.  A second, persisted
// nlist .pt2 of the same weights (deeppot_dpa1_graph_nlist_ref.pt2) is loaded
// alongside the graph model so arbitrary system sizes (dynamic edge axis) can
// be cross-checked graph≈dense live without baking more reference blocks.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

namespace {
constexpr const char* kGraphModel = "../../tests/infer/deeppot_dpa1_graph.pt2";
constexpr const char* kNlistRefModel =
    "../../tests/infer/deeppot_dpa1_graph_nlist_ref.pt2";
constexpr const char* kRefPath =
    "../../tests/infer/deeppot_dpa1_graph.expected";
}  // namespace

template <class VALUETYPE>
class TestInferDpa1GraphPtExpt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  // Per-atom reference (energy/force/virial) loaded from the .expected sidecar.
  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_v;
  int natoms;
  double expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  // Graph-schema model under test.
  static deepmd::DeepPot dp;
  // Independent nlist (dense) model with identical weights — used as a live
  // graph≈dense oracle for arbitrary system sizes.
  static deepmd::DeepPot dp_ref;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp.init(kGraphModel);
    dp_ref.init(kNlistRefModel);
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

  static void TearDownTestSuite() {
    dp = deepmd::DeepPot();
    dp_ref = deepmd::DeepPot();
  }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDpa1GraphPtExpt<VALUETYPE>::dp;
template <class VALUETYPE>
deepmd::DeepPot TestInferDpa1GraphPtExpt<VALUETYPE>::dp_ref;

TYPED_TEST_SUITE(TestInferDpa1GraphPtExpt, ValueTypes);

// Case 1: DeepPot builds its own neighbor list and runs the standalone graph
// branch (lower_input_is_graph_, build_nlist -> buildGraphTensors).  Validates
// the graph AOTI ABI/geometry against the independent nlist reference.
TYPED_TEST(TestInferDpa1GraphPtExpt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  double ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box);

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

// Case 2: a SECOND, larger system (12 atoms, different edge count) through the
// SAME loaded graph model — proves the dynamic edge axis works in C++.  The
// graph result is cross-checked against the dense nlist .pt2 (same weights);
// at non-binding sel they must agree bit-for-bit (fp64 ~1e-10).
TYPED_TEST(TestInferDpa1GraphPtExpt, cpu_build_nlist_sys2_dynamic_edges) {
  using VALUETYPE = TypeParam;
  deepmd::DeepPot& dp = this->dp;
  deepmd::DeepPot& dp_ref = this->dp_ref;

  // 12 atoms: original 6 stacked with a +13 z-shifted copy, box doubled in z.
  // Same local density as the 6-atom fixture, so per-atom neighbor counts stay
  // far below sel=30 and graph(carry-all) == dense(sel-truncated).
  std::vector<VALUETYPE> coord2 = {
      12.83, 2.56, 2.18,  12.09, 2.87, 2.74,  00.25, 3.32, 1.68,
      3.36,  3.00, 1.81,  3.51,  2.51, 2.60,  4.27,  3.22, 1.56,
      12.83, 2.56, 15.18, 12.09, 2.87, 15.74, 00.25, 3.32, 14.68,
      3.36,  3.00, 14.81, 3.51,  2.51, 15.60, 4.27,  3.22, 14.56};
  std::vector<int> atype2 = {0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box2 = {13., 0., 0., 0., 13., 0., 0., 0., 26.};
  int natoms2 = atype2.size();

  double ener_g, ener_r;
  std::vector<VALUETYPE> force_g, virial_g, force_r, virial_r;
  dp.compute(ener_g, force_g, virial_g, coord2, atype2, box2);
  dp_ref.compute(ener_r, force_r, virial_r, coord2, atype2, box2);

  EXPECT_EQ(force_g.size(), static_cast<size_t>(natoms2 * 3));
  EXPECT_EQ(virial_g.size(), 9u);

  EXPECT_LT(fabs(ener_g - ener_r), EPSILON);
  for (int ii = 0; ii < natoms2 * 3; ++ii) {
    EXPECT_LT(fabs(force_g[ii] - force_r[ii]), EPSILON);
  }
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial_g[ii] - virial_r[ii]), EPSILON);
  }
}

// Case 3 (CRITICAL): exercise the LAMMPS compute_inner graph branch with an
// explicit InputNlist and the `ago` cache.  Calling compute twice WITHOUT
// rebuilding the nlist — first ago=0 (rebuild), then ago=1 (reuse) — must give
// identical results.  This is the only case that hits compute_inner + the
// member-cached mapping_ vector; the build-nlist cases above never touch it.
// Regression guard for the OOB-on-ago>0 bug fixed by caching mapping_ as a
// member (commit 7c70db47b).
TYPED_TEST(TestInferDpa1GraphPtExpt, lammps_nlist_ago) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
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

  // ago=0: rebuild the cached nlist/mapping, then run the graph branch.
  double ener;
  std::vector<VALUETYPE> force_, virial;
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 0);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

  EXPECT_EQ(force.size(), static_cast<size_t>(natoms * 3));
  EXPECT_EQ(virial.size(), 9u);
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }

  // ago=1: reuse the cached nlist/mapping (NO rebuild).  Must match again.
  // This is the path that previously read the local mapping vector OOB.
  ener = 0.;
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box, nall - nloc,
             inlist, 1);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3);

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

// Case 5: exercise the DeepPot::compute ATOMIC overload on the graph .pt2.
// This is the first test to reach the ``if (atomic)`` branch inside
// remap_graph_outputs_to_dense_keys (the atom_energy/atom_virial remapping).
// The per-atom reference values are already loaded from
// deeppot_dpa1_graph.expected into this->expected_e and this->expected_v by
// SetUp().
TYPED_TEST(TestInferDpa1GraphPtExpt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  double& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;

  double ener;
  std::vector<VALUETYPE> force, virial, atom_energy, atom_virial;
  // Standalone atomic overload: DeepPot builds its own nlist (graph branch),
  // then returns per-atom energy + atom-virial alongside total
  // energy/force/virial.
  dp.compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box);

  EXPECT_EQ(force.size(), static_cast<size_t>(natoms * 3));
  EXPECT_EQ(virial.size(), 9u);
  EXPECT_EQ(atom_energy.size(), static_cast<size_t>(natoms));
  EXPECT_EQ(atom_virial.size(), static_cast<size_t>(natoms * 9));

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_energy[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_virial[ii] - expected_v[ii]), EPSILON);
  }
}

// Case 4: a tiny system with no in-cutoff neighbors — only the two masked
// dummy edges survive (nedge_min=2 guard / SIGFPE-edge family).  The graph
// must run cleanly, produce finite, interaction-free output (zero force/virial)
// and agree with the dense reference.
TYPED_TEST(TestInferDpa1GraphPtExpt, cpu_build_nlist_tiny_no_edges) {
  using VALUETYPE = TypeParam;
  deepmd::DeepPot& dp = this->dp;
  deepmd::DeepPot& dp_ref = this->dp_ref;

  // Two atoms ~33 apart in a 40-box: no neighbor within rcut=6 and no periodic
  // image either, so the graph sees zero real edges (only the 2 dummy edges).
  std::vector<VALUETYPE> coord_t = {1.0, 1.0, 1.0, 20.0, 20.0, 20.0};
  std::vector<int> atype_t = {0, 1};
  std::vector<VALUETYPE> box_t = {40., 0., 0., 0., 40., 0., 0., 0., 40.};
  int natoms_t = atype_t.size();

  double ener_g, ener_r;
  std::vector<VALUETYPE> force_g, virial_g, force_r, virial_r;
  ASSERT_NO_THROW(
      dp.compute(ener_g, force_g, virial_g, coord_t, atype_t, box_t));
  dp_ref.compute(ener_r, force_r, virial_r, coord_t, atype_t, box_t);

  EXPECT_EQ(force_g.size(), static_cast<size_t>(natoms_t * 3));
  EXPECT_EQ(virial_g.size(), 9u);

  EXPECT_TRUE(std::isfinite(ener_g));
  // No interactions: force and virial must vanish.
  for (int ii = 0; ii < natoms_t * 3; ++ii) {
    EXPECT_TRUE(std::isfinite(force_g[ii]));
    EXPECT_LT(fabs(force_g[ii]), EPSILON);
  }
  for (int ii = 0; ii < 9; ++ii) {
    EXPECT_TRUE(std::isfinite(virial_g[ii]));
    EXPECT_LT(fabs(virial_g[ii]), EPSILON);
  }
  // graph == dense for the isolated-atom limit.
  EXPECT_LT(fabs(ener_g - ener_r), EPSILON);
  for (int ii = 0; ii < natoms_t * 3; ++ii) {
    EXPECT_LT(fabs(force_g[ii] - force_r[ii]), EPSILON);
  }
}
