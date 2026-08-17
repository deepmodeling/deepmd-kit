// SPDX-License-Identifier: LGPL-3.0-or-later
// Test C++ inference for the native-spin DPA4C archive, the family the
// backend dispatch routes to NativeSpinPTExpt: it declares spin_scheme
// "native" and, unlike DPA4's graph lower, ships no with-comm artifact (see
// source/tests/infer/gen_dpa4c_spin.py and DeepPotPTExptPlugin.cc).
//
// The cases here pin how the standalone entry point divides its inputs among
// frames, which is the whole of what distinguishes a multi-frame call from a
// sequence of single-frame ones.
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "DeepSpin.h"
// Defines BUILD_PT_EXPT_NATIVE_SPIN, which says whether the class under test
// was compiled at all, and names the class these cases reach.
#include "NativeSpinPTExpt.h"
#include "expected_ref.h"
#include "test_utils.h"

// Spin models need a relaxed epsilon, as in the sibling DPA4 suites.
#undef EPSILON
#define EPSILON (std::is_same<VALUETYPE, double>::value ? 1e-10 : 1e-4)

namespace {
constexpr const char* kRefPath =
    "../../tests/infer/deeppot_dpa4c_spin_graph.expected";
constexpr const char* kModelPath =
    "../../tests/infer/deeppot_dpa4c_spin_graph.pt2";
}  // namespace

template <class VALUETYPE>
class TestInferNativeSpinDpa4cPtExpt : public ::testing::Test {
 protected:
  // 6-atom system (3 Ni, magnetic; 3 O, not), verbatim from the gen script.
  std::vector<VALUETYPE> coord = {1.0, 1.0, 1.0, 3.2, 1.4, 1.1, 1.3, 1.8, 1.0,
                                  0.4, 1.2, 1.6, 3.6, 2.0, 1.3, 3.4, 0.7, 1.7};
  std::vector<VALUETYPE> coord_alt = {1.2, 0.9, 1.3, 3.0, 1.6, 0.9,
                                      1.5, 2.0, 1.2, 0.6, 1.0, 1.4,
                                      3.4, 2.2, 1.5, 3.6, 0.5, 1.9};
  std::vector<VALUETYPE> spin = {0.11,  0.05,  -0.02, -0.07, 0.09,  0.03,
                                 0.02,  -0.06, 0.08,  0.01,  -0.01, 0.02,
                                 -0.02, 0.03,  -0.01, 0.015, 0.02,  -0.03};
  std::vector<int> atype = {0, 0, 0, 1, 1, 1};
  std::vector<VALUETYPE> box = {6., 0., 0., 0., 6., 0., 0., 0., 6.};
  std::vector<VALUETYPE> fparam = {0.3, -0.2};
  std::vector<VALUETYPE> aparam = {0.1, -0.4, 0.7, -0.3, 0.5, 0.2};
  std::vector<double> charge_spin_default = {0.0, 1.0};
  std::vector<double> charge_spin_other = {1.0, 2.0};

  int natoms;
  double expected_e_frame0, expected_e_frame1, expected_e_other_state;
  std::vector<VALUETYPE> expected_f_frame0, expected_f_frame1;
  std::vector<VALUETYPE> expected_fm_frame0, expected_fm_frame1;

  deepmd::DeepSpin dp;

  double total_energy(const deepmd_test::ExpectedRef& ref,
                      const char* section) {
    const std::vector<VALUETYPE> atom_energy =
        ref.template get<VALUETYPE>(section, "expected_e");
    double total = 0.;
    for (std::size_t ii = 0; ii < atom_energy.size(); ++ii) {
      total += atom_energy[ii];
    }
    return total;
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT_NATIVE_SPIN
    GTEST_SKIP() << "Skip because native-spin PyTorch support is not enabled.";
#endif
    std::ifstream model_file(kModelPath);
    if (!model_file.good()) {
      GTEST_SKIP() << "Skip because " << kModelPath
                   << " was not generated (run "
                      "source/tests/infer/gen_dpa4c_spin.py).";
    }
    dp.init(kModelPath);

    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    expected_f_frame0 = ref.template get<VALUETYPE>("frame0", "expected_f");
    expected_f_frame1 = ref.template get<VALUETYPE>("frame1", "expected_f");
    expected_fm_frame0 = ref.template get<VALUETYPE>("frame0", "expected_fm");
    expected_fm_frame1 = ref.template get<VALUETYPE>("frame1", "expected_fm");
    expected_e_frame0 = total_energy(ref, "frame0");
    expected_e_frame1 = total_energy(ref, "frame1");
    expected_e_other_state = total_energy(ref, "frame0_other_state");
    natoms = static_cast<int>(atype.size());
  };

  void TearDown() override {};

  /** @brief Lay two frames' worth of a per-frame input end to end. */
  template <typename T>
  static std::vector<T> two(const std::vector<T>& first,
                            const std::vector<T>& second) {
    std::vector<T> both(first);
    both.insert(both.end(), second.begin(), second.end());
    return both;
  }
};

TYPED_TEST_SUITE(TestInferNativeSpinDpa4cPtExpt, ValueTypes);

TYPED_TEST(TestInferNativeSpinDpa4cPtExpt, type_map) {
  std::string type_map;
  this->dp.get_type_map(type_map);
  EXPECT_EQ(type_map, "Ni O");
}

TYPED_TEST(TestInferNativeSpinDpa4cPtExpt, one_frame) {
  using VALUETYPE = TypeParam;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  this->dp.compute(ener, force, force_mag, virial, this->coord, this->spin,
                   this->atype, this->box, this->fparam, this->aparam,
                   this->charge_spin_default);

  EXPECT_EQ(force.size(), static_cast<std::size_t>(this->natoms * 3));
  EXPECT_LT(fabs(ener - this->expected_e_frame0), EPSILON);
  for (int ii = 0; ii < this->natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - this->expected_f_frame0[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - this->expected_fm_frame0[ii]), EPSILON);
  }
}

// Each frame carries its own cell and therefore its own ghost set, so a
// multi-frame call has to answer for every frame rather than repeat the
// first, which is what it did while the entry point read one frame.
TYPED_TEST(TestInferNativeSpinDpa4cPtExpt, frames_are_answered_independently) {
  using VALUETYPE = TypeParam;
  const int natoms = this->natoms;
  std::vector<double> ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  this->dp.compute(
      ener, force, force_mag, virial, this->two(this->coord, this->coord_alt),
      this->two(this->spin, this->spin), this->atype,
      this->two(this->box, this->box), this->two(this->fparam, this->fparam),
      this->two(this->aparam, this->aparam),
      this->two(this->charge_spin_default, this->charge_spin_default));

  ASSERT_EQ(ener.size(), 2u);
  EXPECT_EQ(force.size(), static_cast<std::size_t>(2 * natoms * 3));
  EXPECT_EQ(force_mag.size(), static_cast<std::size_t>(2 * natoms * 3));
  EXPECT_EQ(virial.size(), 18u);
  EXPECT_LT(fabs(ener[0] - this->expected_e_frame0), EPSILON);
  EXPECT_LT(fabs(ener[1] - this->expected_e_frame1), EPSILON);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - this->expected_f_frame0[ii]), EPSILON);
    EXPECT_LT(fabs(force[natoms * 3 + ii] - this->expected_f_frame1[ii]),
              EPSILON);
    EXPECT_LT(fabs(force_mag[ii] - this->expected_fm_frame0[ii]), EPSILON);
    EXPECT_LT(fabs(force_mag[natoms * 3 + ii] - this->expected_fm_frame1[ii]),
              EPSILON);
  }
}

// computew documents that a caller may pass one block of parameters for
// every frame to reuse, alongside the per-frame form.
TYPED_TEST(TestInferNativeSpinDpa4cPtExpt, parameters_may_be_shared_by_frames) {
  using VALUETYPE = TypeParam;
  std::vector<double> shared, per_frame;
  std::vector<VALUETYPE> force, force_mag, virial;

  this->dp.compute(shared, force, force_mag, virial,
                   this->two(this->coord, this->coord_alt),
                   this->two(this->spin, this->spin), this->atype,
                   this->two(this->box, this->box), this->fparam, this->aparam,
                   this->charge_spin_default);
  this->dp.compute(
      per_frame, force, force_mag, virial,
      this->two(this->coord, this->coord_alt),
      this->two(this->spin, this->spin), this->atype,
      this->two(this->box, this->box), this->two(this->fparam, this->fparam),
      this->two(this->aparam, this->aparam),
      this->two(this->charge_spin_default, this->charge_spin_default));

  ASSERT_EQ(shared.size(), 2u);
  ASSERT_EQ(per_frame.size(), 2u);
  EXPECT_DOUBLE_EQ(shared[0], per_frame[0]);
  EXPECT_DOUBLE_EQ(shared[1], per_frame[1]);
}

// This backend serves one state at a time, installed ahead of inference, and
// a call may name that state as well. The condition is then checked against
// the frames the call carries, so naming it once per frame must be accepted.
TYPED_TEST(TestInferNativeSpinDpa4cPtExpt,
           an_installed_state_reaches_every_frame) {
  using VALUETYPE = TypeParam;
  this->dp.set_charge_spin(this->charge_spin_other);

  std::vector<double> ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  this->dp.compute(
      ener, force, force_mag, virial, this->two(this->coord, this->coord),
      this->two(this->spin, this->spin), this->atype,
      this->two(this->box, this->box), this->two(this->fparam, this->fparam),
      this->two(this->aparam, this->aparam),
      this->two(this->charge_spin_other, this->charge_spin_other));

  ASSERT_EQ(ener.size(), 2u);
  EXPECT_LT(fabs(ener[0] - this->expected_e_other_state), EPSILON);
  EXPECT_LT(fabs(ener[1] - this->expected_e_other_state), EPSILON);
  // Anti-vacuity: the installed state has to be what moved the energy.
  EXPECT_GT(fabs(this->expected_e_other_state - this->expected_e_frame0), 1e-6);
}

// The model keeps serving the state it was frozen with until a caller
// installs another, so naming a different one per call cannot be honoured.
TYPED_TEST(TestInferNativeSpinDpa4cPtExpt,
           a_state_the_model_does_not_serve_is_refused) {
  using VALUETYPE = TypeParam;
  double ener;
  std::vector<VALUETYPE> force, force_mag, virial;
  EXPECT_THROW(
      this->dp.compute(ener, force, force_mag, virial, this->coord, this->spin,
                       this->atype, this->box, this->fparam, this->aparam,
                       this->charge_spin_other),
      deepmd::deepmd_exception);
}
