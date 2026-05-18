// SPDX-License-Identifier: LGPL-3.0-or-later
// Test multi-frame batch inference for pt_expt (.pt2) backend with
// fparam/aparam.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "expected_ref.h"
#include "test_utils.h"

namespace {
constexpr const char* kRefPath = "../../tests/infer/fparam_aparam.expected";
constexpr const char* kModelPath = "../../tests/infer/fparam_aparam.pt2";
}  // namespace

template <class VALUETYPE>
class TestInferDeepPotAFparamAparamNFramesPtExpt : public ::testing::Test {
 protected:
  // nframes=2: duplicate single-frame data
  std::vector<VALUETYPE> coord = {
      12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56,
      12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 0, 0, 0, 0, 0};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.,
                                13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> fparam = {0.25852028, 0.25852028};
  std::vector<VALUETYPE> aparam = {
      0.25852028, 0.25852028, 0.25852028, 0.25852028, 0.25852028, 0.25852028,
      0.25852028, 0.25852028, 0.25852028, 0.25852028, 0.25852028, 0.25852028};
  std::vector<VALUETYPE> expected_e;
  std::vector<VALUETYPE> expected_f;
  std::vector<VALUETYPE> expected_v;
  int natoms;
  int nframes = 2;
  std::vector<double> expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  static deepmd::DeepPot dp;

  static void SetUpTestSuite() {
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
    dp.init(kModelPath);
#endif
  }

  void SetUp() override {
#if !defined(BUILD_PYTORCH) || !BUILD_PT_EXPT
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    deepmd_test::ExpectedRef ref;
    ref.load(kRefPath);
    auto e_single = ref.get<VALUETYPE>("default", "expected_e");
    auto f_single = ref.get<VALUETYPE>("default", "expected_f");
    auto v_single = ref.get<VALUETYPE>("default", "expected_v");
    // Replicate single-frame reference for nframes batched inference.
    expected_e.reserve(nframes * e_single.size());
    expected_f.reserve(nframes * f_single.size());
    expected_v.reserve(nframes * v_single.size());
    for (int kk = 0; kk < nframes; ++kk) {
      expected_e.insert(expected_e.end(), e_single.begin(), e_single.end());
      expected_f.insert(expected_f.end(), f_single.begin(), f_single.end());
      expected_v.insert(expected_v.end(), v_single.begin(), v_single.end());
    }

    natoms = expected_e.size() / nframes;
    EXPECT_EQ(nframes * natoms * 3, expected_f.size());
    EXPECT_EQ(nframes * natoms * 9, expected_v.size());
    expected_tot_e.assign(nframes, 0.);
    expected_tot_v.assign(static_cast<size_t>(nframes) * 9, 0.);
    for (int kk = 0; kk < nframes; ++kk) {
      for (int ii = 0; ii < natoms; ++ii) {
        expected_tot_e[kk] += expected_e[kk * natoms + ii];
      }
      for (int ii = 0; ii < natoms; ++ii) {
        for (int dd = 0; dd < 9; ++dd) {
          expected_tot_v[kk * 9 + dd] +=
              expected_v[kk * natoms * 9 + ii * 9 + dd];
        }
      }
    }
  };

  void TearDown() override {};

  static void TearDownTestSuite() { dp = deepmd::DeepPot(); }
};

template <class VALUETYPE>
deepmd::DeepPot TestInferDeepPotAFparamAparamNFramesPtExpt<VALUETYPE>::dp;

TYPED_TEST_SUITE(TestInferDeepPotAFparamAparamNFramesPtExpt, ValueTypes);

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesPtExpt, cpu_build_nlist) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  int& natoms = this->natoms;
  int& nframes = this->nframes;
  std::vector<double>& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  std::vector<double> ener;
  std::vector<VALUETYPE> force, virial;
  dp.compute(ener, force, virial, coord, atype, box, fparam, aparam);

  EXPECT_EQ(ener.size(), nframes);
  EXPECT_EQ(force.size(), nframes * natoms * 3);
  EXPECT_EQ(virial.size(), nframes * 9);

  for (int ii = 0; ii < nframes; ++ii) {
    EXPECT_LT(fabs(ener[ii] - expected_tot_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < nframes * natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < nframes * 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
}

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesPtExpt, cpu_build_nlist_atomic) {
  using VALUETYPE = TypeParam;
  std::vector<VALUETYPE>& coord = this->coord;
  std::vector<int>& atype = this->atype;
  std::vector<VALUETYPE>& box = this->box;
  std::vector<VALUETYPE>& fparam = this->fparam;
  std::vector<VALUETYPE>& aparam = this->aparam;
  std::vector<VALUETYPE>& expected_e = this->expected_e;
  std::vector<VALUETYPE>& expected_f = this->expected_f;
  std::vector<VALUETYPE>& expected_v = this->expected_v;
  int& natoms = this->natoms;
  int& nframes = this->nframes;
  std::vector<double>& expected_tot_e = this->expected_tot_e;
  std::vector<VALUETYPE>& expected_tot_v = this->expected_tot_v;
  deepmd::DeepPot& dp = this->dp;
  std::vector<double> ener;
  std::vector<VALUETYPE> force, virial, atom_ener, atom_vir;
  dp.compute(ener, force, virial, atom_ener, atom_vir, coord, atype, box,
             fparam, aparam);

  EXPECT_EQ(ener.size(), nframes);
  EXPECT_EQ(force.size(), nframes * natoms * 3);
  EXPECT_EQ(virial.size(), nframes * 9);
  EXPECT_EQ(atom_ener.size(), nframes * natoms);
  EXPECT_EQ(atom_vir.size(), nframes * natoms * 9);

  for (int ii = 0; ii < nframes; ++ii) {
    EXPECT_LT(fabs(ener[ii] - expected_tot_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < nframes * natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), EPSILON);
  }
  for (int ii = 0; ii < nframes * 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), EPSILON);
  }
  for (int ii = 0; ii < nframes * natoms; ++ii) {
    EXPECT_LT(fabs(atom_ener[ii] - expected_e[ii]), EPSILON);
  }
  for (int ii = 0; ii < nframes * natoms * 9; ++ii) {
    EXPECT_LT(fabs(atom_vir[ii] - expected_v[ii]), EPSILON);
  }
}
