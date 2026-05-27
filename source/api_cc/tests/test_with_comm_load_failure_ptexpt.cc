// SPDX-License-Identifier: LGPL-3.0-or-later
// Tests for the dispatch-site fail-fast guard when the with-comm AOTI
// artifact failed to load at init time. The fixtures are produced by
// source/tests/infer/gen_corrupt_with_comm.py: copies of the valid
// multi-rank .pt2 archives whose nested
// ``model/extra/forward_lower_with_comm.pt2`` entry has been replaced
// with garbage bytes. The outer metadata still claims
// ``has_comm_artifact: true`` so the loader exercises the catch path.
//
// Expectations:
//   * init() succeeds (the loader logs and falls back instead of aborting).
//   * Single-rank dispatch (nswap == 0) keeps working through the regular
//     forward_lower artifact.
//   * Multi-rank dispatch (nswap > 0) throws a deepmd::deepmd_exception
//     instead of silently dropping the MPI ghost-embedding exchange.
#include <gtest/gtest.h>

#include <fstream>
#include <vector>

#include "DeepPot.h"
// Include the PT_Expt headers so BUILD_PT_EXPT / BUILD_PT_EXPT_SPIN are
// visible to the GTEST_SKIP guard below.
#include "DeepPotPTExpt.h"
#include "DeepSpin.h"
#include "DeepSpinPTExpt.h"
#include "common.h"
#include "neighbor_list.h"
#include "test_utils.h"

namespace {
constexpr const char* kPotCorrupt =
    "../../tests/infer/deeppot_dpa3_mpi_corrupt_with_comm.pt2";
constexpr const char* kSpinCorrupt =
    "../../tests/infer/deeppot_dpa3_spin_mpi_corrupt_with_comm.pt2";

bool file_exists(const char* path) {
  std::ifstream f(path);
  return f.good();
}
}  // namespace

// ============================================================================
// DeepPot (non-spin) — corrupted with-comm artifact
// ============================================================================

class TestDeepPotPTExptWithCommLoadFailure : public ::testing::Test {
 protected:
  // Coordinates / atype / box copied from gen_dpa3.py so the regular
  // forward_lower artifact has well-formed inputs to evaluate.
  std::vector<double> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                               00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                               3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<double> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};

  deepmd::DeepPot dp;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch / pt_expt support is not enabled.";
#endif
    if (!file_exists(kPotCorrupt)) {
      GTEST_SKIP() << "Skipping: " << kPotCorrupt
                   << " not found. Run source/tests/infer/"
                      "gen_corrupt_with_comm.py first.";
    }
    // Init must succeed: the with-comm loader fails internally and the
    // catch block keeps the regular single-rank artifact usable.
    ASSERT_NO_THROW(dp.init(kPotCorrupt));
  }
};

TEST_F(TestDeepPotPTExptWithCommLoadFailure, single_rank_compute_succeeds) {
  // nswap == 0 (default InputNlist) routes through the regular
  // forward_lower artifact; the broken with-comm artifact is not
  // consulted, so compute must succeed.
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<double> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<double>(nlist_data, coord_cpy, atype_cpy, mapping, coord, atype,
                       box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);
  inlist.mapping = mapping.data();
  ASSERT_EQ(inlist.nswap, 0);  // pre-condition: single-rank dispatch

  double ener;
  std::vector<double> force_, virial;
  EXPECT_NO_THROW(dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box,
                             nall - nloc, inlist, 0));
  EXPECT_EQ(force_.size(), nall * 3);
  EXPECT_EQ(virial.size(), 9);
}

TEST_F(TestDeepPotPTExptWithCommLoadFailure, multi_rank_compute_throws) {
  // nswap > 0 forces the dispatch site to ``run_model_with_comm``; the
  // load-failure guard added by PR #5430 must throw rather than silently
  // falling back to the single-rank path. The send/recv arrays remain
  // null — the guard fires before any of them are dereferenced.
  float rc = dp.cutoff();
  int nloc = coord.size() / 3;
  std::vector<double> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<double>(nlist_data, coord_cpy, atype_cpy, mapping, coord, atype,
                       box, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);
  inlist.mapping = mapping.data();
  inlist.nswap = 1;  // simulate multi-rank without populating send/recv

  double ener;
  std::vector<double> force_, virial;
  EXPECT_THROW(dp.compute(ener, force_, virial, coord_cpy, atype_cpy, box,
                          nall - nloc, inlist, 0),
               deepmd::deepmd_exception);
}

// ============================================================================
// DeepSpin — corrupted with-comm artifact
// ============================================================================

class TestDeepSpinPTExptWithCommLoadFailure : public ::testing::Test {
 protected:
  std::vector<double> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                               00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                               3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  // Match deeppot_dpa3_spin_mpi.pt2 spin layout (type 0 has spin, types
  // 1+ do not) — spin vector packed alongside coord.
  std::vector<double> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                              0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<double> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};

  deepmd::DeepSpin dp;

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT_SPIN
    GTEST_SKIP() << "Skip because PyTorch / pt_expt spin support is not "
                    "enabled.";
#endif
    if (!file_exists(kSpinCorrupt)) {
      GTEST_SKIP() << "Skipping: " << kSpinCorrupt
                   << " not found. Run source/tests/infer/"
                      "gen_corrupt_with_comm.py first.";
    }
    ASSERT_NO_THROW(dp.init(kSpinCorrupt));
  }
};

TEST_F(TestDeepSpinPTExptWithCommLoadFailure, single_rank_compute_succeeds) {
  // NoPBC + hardcoded all-pairs nlist mirrors the
  // ``cpu_lmp_nlist`` pattern in test_deeppot_dpa_ptexpt_spin.cc:
  // nloc == natoms == nall, no ghost atoms.
  const int natoms = static_cast<int>(atype.size());
  std::vector<double> empty_box;
  std::vector<std::vector<int>> nlist_data = {{1, 2, 3, 4, 5}, {0, 2, 3, 4, 5},
                                              {0, 1, 3, 4, 5}, {0, 1, 2, 4, 5},
                                              {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);
  ASSERT_EQ(inlist.nswap, 0);

  double ener;
  std::vector<double> force_, force_mag, virial;
  EXPECT_NO_THROW(dp.compute(ener, force_, force_mag, virial, coord, spin,
                             atype, empty_box, 0, inlist, 0));
}

TEST_F(TestDeepSpinPTExptWithCommLoadFailure, multi_rank_compute_throws) {
  const int natoms = static_cast<int>(atype.size());
  std::vector<double> empty_box;
  std::vector<std::vector<int>> nlist_data = {{1, 2, 3, 4, 5}, {0, 2, 3, 4, 5},
                                              {0, 1, 3, 4, 5}, {0, 1, 2, 4, 5},
                                              {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);
  inlist.nswap = 1;  // simulate multi-rank without populating send/recv

  double ener;
  std::vector<double> force_, force_mag, virial;
  EXPECT_THROW(dp.compute(ener, force_, force_mag, virial, coord, spin, atype,
                          empty_box, 0, inlist, 0),
               deepmd::deepmd_exception);
}
