// SPDX-License-Identifier: LGPL-3.0-or-later
// Test multi-frame batch inference for pt_expt (.pt2) backend with
// fparam/aparam.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "DeepPot.h"
#include "test_utils.h"

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
  // Same reference values as single-frame, duplicated for 2 frames
  std::vector<VALUETYPE> expected_e = {
      1.596836265688982293e-01, 1.596933624175455035e-01,
      1.596859462832928844e-01, 1.596779837732069107e-01,
      1.596776702807257142e-01, 1.596869048501883825e-01,
      1.596836265688982293e-01, 1.596933624175455035e-01,
      1.596859462832928844e-01, 1.596779837732069107e-01,
      1.596776702807257142e-01, 1.596869048501883825e-01};
  std::vector<VALUETYPE> expected_f = {
      1.112134318320094352e-05,  1.085230789880272100e-04,
      9.298442641358874074e-07,  1.491517597320257530e-04,
      -1.250419527572717225e-05, -9.768265174690741028e-05,
      -5.021052645725073397e-05, -9.741678916887848341e-05,
      9.375317764392495572e-05,  8.664103999852425394e-05,
      -4.538513400016661465e-05, 8.561605116672722879e-05,
      -2.454811055475981441e-05, 1.079491454988312104e-04,
      -1.656974003674590440e-04, -1.721555059017403750e-04,
      -6.116610604208616705e-05, 8.308097903957835525e-05,
      1.112134318320094352e-05,  1.085230789880272100e-04,
      9.298442641358874074e-07,  1.491517597320257530e-04,
      -1.250419527572717225e-05, -9.768265174690741028e-05,
      -5.021052645725073397e-05, -9.741678916887848341e-05,
      9.375317764392495572e-05,  8.664103999852425394e-05,
      -4.538513400016661465e-05, 8.561605116672722879e-05,
      -2.454811055475981441e-05, 1.079491454988312104e-04,
      -1.656974003674590440e-04, -1.721555059017403750e-04,
      -6.116610604208616705e-05, 8.308097903957835525e-05};
  std::vector<VALUETYPE> expected_v = {
      -1.264062189119726106e-04, -1.636544077308309524e-05,
      4.453224130911556590e-05,  -7.947403699519458518e-06,
      -4.603504987332719071e-05, 9.491045850088816000e-06,
      4.131028921467394082e-05,  9.691472468201876704e-06,
      -3.323572704427471520e-05, -1.024556912293147473e-04,
      5.530809120954559223e-06,  5.211030391191971272e-05,
      -3.851138686809524851e-06, 2.101414374153427902e-07,
      3.247573516972862344e-06,  4.561253716254950400e-05,
      -3.865680092083280590e-06, -3.262252150841829144e-05,
      -1.166788692566841262e-04, -1.814499890570940753e-05,
      2.155064011880968559e-05,  -1.629918981392229854e-05,
      -3.245631268444005592e-05, 2.968538417601228937e-05,
      2.463149007223104176e-05,  3.660689861518736949e-05,
      -3.586518711234946879e-05, -1.424206401855391917e-04,
      -1.017840928263488617e-05, 1.421307534994552908e-05,
      -8.618294024757048886e-06, -2.192409332705383732e-05,
      3.461715847634955364e-05,  1.277625693457703254e-05,
      3.486479415793123331e-05,  -5.604161168847292539e-05,
      -8.612844407008925294e-05, 2.508361660152536905e-06,
      -1.633895954532816838e-07, 1.903591783622687189e-06,
      -3.028341203071209831e-05, 4.685511271783763848e-05,
      2.876824509984768127e-06,  4.576515617130315025e-05,
      -7.108738780331672730e-05, -1.062354815105974551e-04,
      -2.954644717832223544e-05, 4.075640001084848793e-05,
      -3.138369091725678469e-05, -1.316088004849699161e-05,
      1.786389692843508954e-05,  4.579187321116929540e-05,
      1.869753034515593833e-05,  -2.550749273395911822e-05,
      -1.264062189119726106e-04, -1.636544077308309524e-05,
      4.453224130911556590e-05,  -7.947403699519458518e-06,
      -4.603504987332719071e-05, 9.491045850088816000e-06,
      4.131028921467394082e-05,  9.691472468201876704e-06,
      -3.323572704427471520e-05, -1.024556912293147473e-04,
      5.530809120954559223e-06,  5.211030391191971272e-05,
      -3.851138686809524851e-06, 2.101414374153427902e-07,
      3.247573516972862344e-06,  4.561253716254950400e-05,
      -3.865680092083280590e-06, -3.262252150841829144e-05,
      -1.166788692566841262e-04, -1.814499890570940753e-05,
      2.155064011880968559e-05,  -1.629918981392229854e-05,
      -3.245631268444005592e-05, 2.968538417601228937e-05,
      2.463149007223104176e-05,  3.660689861518736949e-05,
      -3.586518711234946879e-05, -1.424206401855391917e-04,
      -1.017840928263488617e-05, 1.421307534994552908e-05,
      -8.618294024757048886e-06, -2.192409332705383732e-05,
      3.461715847634955364e-05,  1.277625693457703254e-05,
      3.486479415793123331e-05,  -5.604161168847292539e-05,
      -8.612844407008925294e-05, 2.508361660152536905e-06,
      -1.633895954532816838e-07, 1.903591783622687189e-06,
      -3.028341203071209831e-05, 4.685511271783763848e-05,
      2.876824509984768127e-06,  4.576515617130315025e-05,
      -7.108738780331672730e-05, -1.062354815105974551e-04,
      -2.954644717832223544e-05, 4.075640001084848793e-05,
      -3.138369091725678469e-05, -1.316088004849699161e-05,
      1.786389692843508954e-05,  4.579187321116929540e-05,
      1.869753034515593833e-05,  -2.550749273395911822e-05};
  int natoms;
  int nframes = 2;
  std::vector<double> expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::DeepPot dp;

  void SetUp() override {
#ifndef BUILD_PYTORCH
    GTEST_SKIP() << "Skip because PyTorch support is not enabled.";
#endif
    dp.init("../../tests/infer/fparam_aparam.pt2");

    natoms = expected_e.size() / nframes;
    EXPECT_EQ(nframes * natoms * 3, expected_f.size());
    EXPECT_EQ(nframes * natoms * 9, expected_v.size());
    expected_tot_e.resize(nframes);
    expected_tot_v.resize(static_cast<size_t>(nframes) * 9);
    std::fill(expected_tot_e.begin(), expected_tot_e.end(), 0.);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
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
};

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
