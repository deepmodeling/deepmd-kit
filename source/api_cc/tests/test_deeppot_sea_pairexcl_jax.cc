// SPDX-License-Identifier: LGPL-3.0-or-later
// End-to-end test of the C++ DeepPotJAX model-level pair-exclusion ingestion
// seam on the LAMMPS InputNlist path.
//
// DeepPotJAX consumes BOTH the jax2tf ``.savedmodel`` and the tf2
// ``.savedmodeltf`` SavedModel flavours through the TensorFlow C API.
// Model-level pair_exclude_types is a BUILD-time transform (decision #18/A4):
// the exported
// ``call_lower_*`` consumes a pre-excluded nlist and never re-applies it, so
// the C++ ingestion seam is the SINGLE application site on this path. It reads
// the exported ``get_pair_exclude_types`` getter at init and folds exclusion
// into the LAMMPS nlist before ``call_lower_*`` runs.
//
// ``deeppot_sea_pairexcl.{savedmodel,savedmodeltf}`` carry the SAME weights as
// the no-exclusion baseline ``deeppot_sea.{savedmodel,savedmodeltf}`` (derived
// by injecting only ``pair_exclude_types`` — see convert-models.sh), so the
// ONLY difference is the exclusion. Two assertions make this load-bearing:
//   1. excluded energy differs from the baseline (exclusion is genuinely active
//      on the InputNlist path, not silently dropped);
//   2. the InputNlist (lower) path agrees with the coord-only (upper) path,
//   which
//      builds and pre-excludes its own nlist -- i.e. the seam excludes the SAME
//      pairs the upper path does.
// Without the seam, (1) collapses to equality and (2) diverges.
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "DeepPot.h"
#include "deeppot_universal_test_common.h"
#include "neighbor_list.h"

namespace {
using deepmd_test::universal::Backend;
using deepmd_test::universal::backend_enabled;
using deepmd_test::universal::path_exists;

constexpr const char* kJaxBase = "../../tests/infer/deeppot_sea.savedmodel";
constexpr const char* kJaxExcl =
    "../../tests/infer/deeppot_sea_pairexcl.savedmodel";
constexpr const char* kTf2Base = "../../tests/infer/deeppot_sea.savedmodeltf";
constexpr const char* kTf2Excl =
    "../../tests/infer/deeppot_sea_pairexcl.savedmodeltf";

// Full (all-pairs) neighbour list: every atom neighbours every other atom.
std::vector<std::vector<int>> make_full_nlist(const int natoms) {
  std::vector<std::vector<int>> nlist_data(natoms);
  for (int ii = 0; ii < natoms; ++ii) {
    for (int jj = 0; jj < natoms; ++jj) {
      if (ii != jj) {
        nlist_data[ii].push_back(jj);
      }
    }
  }
  return nlist_data;
}

// Evaluate the total energy on the NoPBC LAMMPS InputNlist path (the DeepPotJAX
// ``call_lower_*`` route that folds in model-level exclusion).
double eval_input_nlist(const char* model,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype) {
  deepmd::DeepPot dp;
  dp.init(model);
  const int natoms = static_cast<int>(atype.size());
  auto nlist_data = make_full_nlist(natoms);
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, ilist.data(), numneigh.data(),
                            firstneigh.data());
  deepmd::convert_nlist(inlist, nlist_data);
  double energy = 0.0;
  std::vector<double> force, virial;
  const std::vector<double> box;  // empty => NoPBC
  dp.compute(energy, force, virial, coord, atype, box, 0, inlist, 0);
  return energy;
}

class TestDeepPotSeaPairExclJax : public ::testing::Test {
 protected:
  // Two compact O-H clusters; every atom is well within the model cutoff, so
  // there are O-H pairs to exclude and the exclusion visibly changes the
  // energy.
  std::vector<double> coord = {
      0.00,  0.00, 0.00,  // O
      0.90,  0.00, 0.00,  // H
      -0.30, 0.90, 0.00,  // H
      2.50,  0.00, 0.00,  // O
      3.40,  0.00, 0.00,  // H
      2.20,  0.90, 0.00,  // H
  };
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};

  void SetUp() override {
    if (!backend_enabled(Backend::JAX)) {
      GTEST_SKIP() << "JAX backend support is not enabled.";
    }
    if (!path_exists(kJaxExcl) || !path_exists(kJaxBase)) {
      GTEST_SKIP() << "jax SavedModel pairexcl fixtures are not available.";
    }
  }
};

// jax2tf .savedmodel: exclusion is applied on the InputNlist (lower) path.
TEST_F(TestDeepPotSeaPairExclJax, jax_savedmodel_inputnlist_excludes) {
  const double e_base = eval_input_nlist(kJaxBase, coord, atype);
  const double e_excl = eval_input_nlist(kJaxExcl, coord, atype);
  EXPECT_TRUE(std::isfinite(e_base));
  EXPECT_TRUE(std::isfinite(e_excl));
  EXPECT_GT(std::fabs(e_excl - e_base), 1e-6)
      << "model-level pair_exclude_types is silently dropped on the DeepPotJAX "
         "InputNlist path (excluded energy == baseline)";
}

// The InputNlist (lower) route must exclude the SAME pairs as the coord-only
// (upper) route, which builds and pre-excludes its own nlist.
TEST_F(TestDeepPotSeaPairExclJax, jax_inputnlist_matches_coord_only) {
  const double e_nlist = eval_input_nlist(kJaxExcl, coord, atype);
  deepmd::DeepPot dp;
  dp.init(kJaxExcl);
  double e_coord = 0.0;
  std::vector<double> force, virial;
  const std::vector<double> box;  // empty => NoPBC, model builds its own nlist
  dp.compute(e_coord, force, virial, coord, atype, box);
  EXPECT_LT(std::fabs(e_nlist - e_coord), 1e-6)
      << "InputNlist-path exclusion disagrees with the coord-only path";
}

// tf2 .savedmodeltf goes through the SAME C++ DeepPotJAX loader/seam.
TEST_F(TestDeepPotSeaPairExclJax, tf2_savedmodeltf_inputnlist_excludes) {
  if (!path_exists(kTf2Excl) || !path_exists(kTf2Base)) {
    GTEST_SKIP() << "tf2 SavedModel pairexcl fixtures are not available.";
  }
  const double e_base = eval_input_nlist(kTf2Base, coord, atype);
  const double e_excl = eval_input_nlist(kTf2Excl, coord, atype);
  EXPECT_TRUE(std::isfinite(e_base));
  EXPECT_TRUE(std::isfinite(e_excl));
  EXPECT_GT(std::fabs(e_excl - e_base), 1e-6)
      << "model-level pair_exclude_types is silently dropped on the DeepPotJAX "
         "tf2 SavedModel InputNlist path";
}

}  // namespace
