// SPDX-License-Identifier: LGPL-3.0-or-later
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "DeepSpin.h"
#include "expected_ref.h"
#include "neighbor_list.h"
#include "test_utils.h"

// 1e-10 cannot pass; unclear bug or not
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-6 : 1e-1)

namespace {
constexpr const char* kRefPath = "../../tests/infer/deeppot_dpa_spin.expected";
constexpr const char* kModelPath = "../../tests/infer/deeppot_dpa_spin.pth";
}  // namespace

template <class VALUETYPE>
class TestInferDeepSpinDpaPt : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                                 0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};

  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_fm;
  std::vector<VALUETYPE> expected_tot_v;
  std::vector<VALUETYPE> expected_atom_v;

  int natoms;
  double expected_tot_e;

  deepmd::DeepSpin dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("pbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("pbc", "expected_f");
    expected_fm = ref.get<VALUETYPE>("pbc", "expected_fm");
    expected_tot_v = ref.get<VALUETYPE>("pbc", "expected_tot_v");
    expected_atom_v = ref.get<VALUETYPE>("pbc", "expected_atom_v");

    dp.init(kModelPath);

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

TYPED_TEST_SUITE(TestInferDeepSpinDpaPt, ValueTypes);

TYPED_TEST(TestInferDeepSpinDpaPt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
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
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpinDpaPt, cpu_build_nlist_atomic) {
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

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), natoms * 9);
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_atom_v[ii]), EPSILON);
  }
}

template <class VALUETYPE>
class TestInferDeepSpinDpaPtNopbc : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74,
                                  00.25, 3.32, 1.68, 3.36,  3.00, 1.81,
                                  3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<VALUETYPE> spin = {0.13, 0.02, 0.03, 0., 0., 0., 0., 0., 0.,
                                 0.14, 0.10, 0.12, 0., 0., 0., 0., 0., 0.};
  std::vector<int> atype = {0, 1, 1, 0, 1, 1};
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
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_e = ref.get<VALUETYPE>("nopbc", "expected_e");
    expected_f = ref.get<VALUETYPE>("nopbc", "expected_f");
    expected_fm = ref.get<VALUETYPE>("nopbc", "expected_fm");
    expected_tot_v = ref.get<VALUETYPE>("nopbc", "expected_tot_v");
    expected_atom_v = ref.get<VALUETYPE>("nopbc", "expected_atom_v");

    dp.init(kModelPath);

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

TYPED_TEST_SUITE(TestInferDeepSpinDpaPtNopbc, ValueTypes);

TYPED_TEST(TestInferDeepSpinDpaPtNopbc, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
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
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpinDpaPtNopbc, cpu_build_nlist_atomic) {
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

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), natoms * 9);
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_atom_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpinDpaPtNopbc, cpu_lmp_nlist) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
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
  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepSpinDpaPtNopbc, cpu_lmp_nlist_null_atom_in_middle) {
  using VALUETYPE = TypeParam;
  const std::vector<VALUETYPE>& coord = this->coord;
  const std::vector<VALUETYPE>& spin = this->spin;
  const std::vector<int>& atype = this->atype;
  const std::vector<VALUETYPE>& box = this->box;
  deepmd::DeepSpin& dp = this->dp;

  // This fixture deliberately uses the TorchScript ``.pth`` artifact, which
  // dispatches through DeepSpinPT rather than the separate PTExpt ``.pt2``
  // implementation.
  ASSERT_EQ(deepmd::get_backend(kModelPath), deepmd::DPBackend::PyTorch);

  auto compute_with_nlist = [&dp, &box](
                                double& energy, std::vector<VALUETYPE>& force,
                                std::vector<VALUETYPE>& force_mag,
                                std::vector<VALUETYPE>& virial,
                                const std::vector<VALUETYPE>& input_coord,
                                const std::vector<VALUETYPE>& input_spin,
                                const std::vector<int>& input_atype,
                                std::vector<std::vector<int> >& nlist_data) {
    const int natoms = static_cast<int>(input_atype.size());
    std::vector<int> ilist(natoms), numneigh(natoms);
    std::vector<int*> firstneigh(natoms);
    deepmd::InputNlist inlist(natoms, ilist.data(), numneigh.data(),
                              firstneigh.data());
    convert_nlist(inlist, nlist_data);
    dp.compute(energy, force, force_mag, virial, input_coord, input_spin,
               input_atype, box, 0, inlist, 0);
  };

  // First evaluate the compact six-atom representation.  The padded system
  // below describes the same physical atoms and neighbor graph, so all real
  // atom predictions must remain unchanged after the NULL atom is filtered.
  std::vector<std::vector<int> > compact_nlist = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  double compact_energy;
  std::vector<VALUETYPE> compact_force, compact_force_mag, compact_virial;
  compute_with_nlist(compact_energy, compact_force, compact_force_mag,
                     compact_virial, coord, spin, atype, compact_nlist);

  constexpr int null_index = 3;
  std::vector<VALUETYPE> padded_coord = coord;
  padded_coord.insert(padded_coord.begin() + null_index * 3,
                      {static_cast<VALUETYPE>(8.1), static_cast<VALUETYPE>(7.2),
                       static_cast<VALUETYPE>(6.3)});
  std::vector<VALUETYPE> padded_spin = spin;
  padded_spin.insert(
      padded_spin.begin() + null_index * 3,
      {static_cast<VALUETYPE>(0.91), static_cast<VALUETYPE>(-0.73),
       static_cast<VALUETYPE>(0.57)});
  std::vector<int> padded_atype = atype;
  padded_atype.insert(padded_atype.begin() + null_index, -1);

  // The NULL row is empty and no real row references it.  After fwd_map
  // compaction this becomes exactly ``compact_nlist``; the distinct NULL spin
  // makes an incorrectly prefix-truncated spin tensor observably different.
  std::vector<std::vector<int> > padded_nlist = {
      {1, 2, 4, 5, 6}, {0, 2, 4, 5, 6}, {0, 1, 4, 5, 6}, {},
      {0, 1, 2, 5, 6}, {0, 1, 2, 4, 6}, {0, 1, 2, 4, 5}};
  double padded_energy;
  std::vector<VALUETYPE> padded_force, padded_force_mag, padded_virial;
  compute_with_nlist(padded_energy, padded_force, padded_force_mag,
                     padded_virial, padded_coord, padded_spin, padded_atype,
                     padded_nlist);

  // These calls describe identical real atoms and neighbor graphs, so use a
  // strict equivalence tolerance instead of this file's loose float reference
  // tolerance.  The old prefix-based spin mapping differs by less than 0.1 and
  // would otherwise let the float regression pass despite the wrong inputs.
  constexpr double equivalence_tolerance =
      std::is_same<VALUETYPE, double>::value ? 1e-10 : 1e-5;
  EXPECT_NEAR(padded_energy, compact_energy, equivalence_tolerance);
  ASSERT_EQ(padded_virial.size(), compact_virial.size());
  for (size_t ii = 0; ii < compact_virial.size(); ++ii) {
    EXPECT_NEAR(padded_virial[ii], compact_virial[ii], equivalence_tolerance);
  }

  ASSERT_EQ(padded_force.size(), padded_atype.size() * 3);
  ASSERT_EQ(padded_force_mag.size(), padded_atype.size() * 3);
  const std::vector<int> compact_to_padded = {0, 1, 2, 4, 5, 6};
  for (size_t ii = 0; ii < compact_to_padded.size(); ++ii) {
    const int padded_index = compact_to_padded[ii];
    for (int dd = 0; dd < 3; ++dd) {
      EXPECT_NEAR(padded_force[padded_index * 3 + dd],
                  compact_force[ii * 3 + dd], equivalence_tolerance);
      EXPECT_NEAR(padded_force_mag[padded_index * 3 + dd],
                  compact_force_mag[ii * 3 + dd], equivalence_tolerance);
    }
  }
  for (int dd = 0; dd < 3; ++dd) {
    EXPECT_EQ(padded_force[null_index * 3 + dd], static_cast<VALUETYPE>(0));
    EXPECT_EQ(padded_force_mag[null_index * 3 + dd], static_cast<VALUETYPE>(0));
  }
}

TYPED_TEST(TestInferDeepSpinDpaPtNopbc, cpu_lmp_nlist_atomic) {
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

  std::vector<std::vector<int> > nlist_data = {
      {1, 2, 3, 4, 5}, {0, 2, 3, 4, 5}, {0, 1, 3, 4, 5},
      {0, 1, 2, 4, 5}, {0, 1, 2, 3, 5}, {0, 1, 2, 3, 4}};
  std::vector<int> ilist(natoms), numneigh(natoms);
  std::vector<int*> firstneigh(natoms);
  deepmd::InputNlist inlist(natoms, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  dp.compute(ener, force, force_mag, virial, atom_ener, atom_vir, coord, spin,
             atype, box, 0, inlist, 0);

  EXPECT_EQ(force.size(), natoms * 3);
  EXPECT_EQ(force_mag.size(), natoms * 3);
  EXPECT_EQ(atom_ener.size(), natoms);

  EXPECT_LT(fabs(ener - expected_tot_e), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - expected_fm[ii]), EPSILON);
  }
  EXPECT_FALSE(virial.empty()) << "Virial should not be empty";
  EXPECT_EQ(virial.size(), 9);
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  EXPECT_FALSE(atom_vir.empty()) << "Atomic virial should not be empty";
  EXPECT_EQ(atom_vir.size(), natoms * 9);
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_atom_v[ii]), EPSILON);
  }
}
