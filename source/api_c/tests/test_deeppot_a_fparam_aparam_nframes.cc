#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <vector>

#include "deepmd.hpp"
#include "test_utils.h"

template <class VALUETYPE>
class TestInferDeepPotAFparamAparamNFrames : public ::testing::Test {
 protected:
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
  std::vector<VALUETYPE> expected_e = {
      -1.038271183039953804e-01, -7.285433575272914908e-02,
      -9.467600174099155552e-02, -1.467050086239614082e-01,
      -7.660561620618722145e-02, -7.277295998502930630e-02,
      -1.038271183039953804e-01, -7.285433575272914908e-02,
      -9.467600174099155552e-02, -1.467050086239614082e-01,
      -7.660561620618722145e-02, -7.277295998502930630e-02};
  std::vector<VALUETYPE> expected_f = {
      6.622266817497907132e-02,  5.278739055693523058e-02,
      2.265727495541422845e-02,  -2.606047850915838363e-02,
      -4.538811686410718776e-02, 1.058247569147072187e-02,
      1.679392490937766935e-01,  -2.257828022687320690e-03,
      -4.490145670355452645e-02, -1.148364103573685929e-01,
      -1.169790466695089237e-02, 6.140402504113953025e-02,
      -8.078778132132799494e-02, -5.838878056243369807e-02,
      6.773639989682191109e-02,  -1.247724708090079161e-02,
      6.494523955924384750e-02,  -1.174787188812918687e-01,
      6.622266817497907132e-02,  5.278739055693523058e-02,
      2.265727495541422845e-02,  -2.606047850915838363e-02,
      -4.538811686410718776e-02, 1.058247569147072187e-02,
      1.679392490937766935e-01,  -2.257828022687320690e-03,
      -4.490145670355452645e-02, -1.148364103573685929e-01,
      -1.169790466695089237e-02, 6.140402504113953025e-02,
      -8.078778132132799494e-02, -5.838878056243369807e-02,
      6.773639989682191109e-02,  -1.247724708090079161e-02,
      6.494523955924384750e-02,  -1.174787188812918687e-01};
  std::vector<VALUETYPE> expected_v = {
      -1.589185553287162656e-01, 2.586163333170100279e-03,
      -1.575127933809472624e-04, -1.855360380105876630e-02,
      1.949822090859933826e-02,  -1.006552056166355388e-02,
      3.177029853276916449e-02,  1.714349636720383010e-03,
      -1.290389175187874483e-03, -8.553510339477603253e-02,
      -5.654637257232508415e-03, -1.286954833787038420e-02,
      2.464156457499515687e-02,  -2.398202886026797043e-02,
      -1.957110465239037672e-02, 2.233492928605742764e-02,
      6.107843207824020099e-03,  1.707078295947736047e-03,
      -1.653994088976195043e-01, 3.894358678172111371e-02,
      -2.169595969759342477e-02, 6.819704294738503786e-03,
      -5.018242039618424008e-03, 2.640664428663210429e-03,
      -1.985298275686078057e-03, -3.638421609610945767e-02,
      2.342932331075030239e-02,  -8.501331914753691710e-02,
      -2.181253413538992297e-03, 4.311300069651782287e-03,
      -1.910329328333908129e-03, -1.808810159508548836e-03,
      -1.540075281450827612e-03, -1.173703213175551763e-02,
      -2.596306629910121507e-03, 6.705025662372287101e-03,
      -9.038455005073858795e-02, 3.011717773578577451e-02,
      -5.083054073419784880e-02, -2.951210292616929069e-03,
      2.342445652898489383e-02,  -4.091207474993674431e-02,
      -1.648470649301832236e-02, -2.872261885460645689e-02,
      4.763924972552112391e-02,  -8.300036532764677732e-02,
      1.020429228955421243e-03,  -1.026734151199098881e-03,
      5.678534096113684732e-02,  1.273635718045938205e-02,
      -1.530143225195957322e-02, -1.061671865629566225e-01,
      -2.486859433265622629e-02, 2.875323131744185121e-02,
      -1.589185553287162656e-01, 2.586163333170100279e-03,
      -1.575127933809472624e-04, -1.855360380105876630e-02,
      1.949822090859933826e-02,  -1.006552056166355388e-02,
      3.177029853276916449e-02,  1.714349636720383010e-03,
      -1.290389175187874483e-03, -8.553510339477603253e-02,
      -5.654637257232508415e-03, -1.286954833787038420e-02,
      2.464156457499515687e-02,  -2.398202886026797043e-02,
      -1.957110465239037672e-02, 2.233492928605742764e-02,
      6.107843207824020099e-03,  1.707078295947736047e-03,
      -1.653994088976195043e-01, 3.894358678172111371e-02,
      -2.169595969759342477e-02, 6.819704294738503786e-03,
      -5.018242039618424008e-03, 2.640664428663210429e-03,
      -1.985298275686078057e-03, -3.638421609610945767e-02,
      2.342932331075030239e-02,  -8.501331914753691710e-02,
      -2.181253413538992297e-03, 4.311300069651782287e-03,
      -1.910329328333908129e-03, -1.808810159508548836e-03,
      -1.540075281450827612e-03, -1.173703213175551763e-02,
      -2.596306629910121507e-03, 6.705025662372287101e-03,
      -9.038455005073858795e-02, 3.011717773578577451e-02,
      -5.083054073419784880e-02, -2.951210292616929069e-03,
      2.342445652898489383e-02,  -4.091207474993674431e-02,
      -1.648470649301832236e-02, -2.872261885460645689e-02,
      4.763924972552112391e-02,  -8.300036532764677732e-02,
      1.020429228955421243e-03,  -1.026734151199098881e-03,
      5.678534096113684732e-02,  1.273635718045938205e-02,
      -1.530143225195957322e-02, -1.061671865629566225e-01,
      -2.486859433265622629e-02, 2.875323131744185121e-02};
  int natoms;
  int nframes = 2;
  std::vector<double> expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::hpp::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/fparam_aparam.pbtxt";
    deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/fparam_aparam.pbtxt",
                                     "fparam_aparam.pb");

    dp.init("fparam_aparam.pb");

    natoms = expected_e.size() / nframes;
    EXPECT_EQ(nframes * natoms * 3, expected_f.size());
    EXPECT_EQ(nframes * natoms * 9, expected_v.size());
    expected_tot_e.resize(nframes);
    expected_tot_v.resize(nframes * 9);
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

  void TearDown() override { remove("fparam_aparam.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepPotAFparamAparamNFrames, ValueTypes);

TYPED_TEST(TestInferDeepPotAFparamAparamNFrames, cpu_build_nlist) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
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

TYPED_TEST(TestInferDeepPotAFparamAparamNFrames, cpu_build_nlist_atomic) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
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

TYPED_TEST(TestInferDeepPotAFparamAparamNFrames, cpu_lmp_nlist) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  std::vector<double> ener;
  std::vector<VALUETYPE> force_, virial;
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

  std::fill(ener.begin(), ener.end(), 0.0);
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFrames, cpu_lmp_nlist_atomic) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  std::vector<double> ener;
  std::vector<VALUETYPE> force_, atom_ener_, atom_vir_, virial;
  std::vector<VALUETYPE> force, atom_ener, atom_vir;
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy2, atype_cpy,
             box, nall - nloc, inlist, 0, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1, nframes);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9, nframes);

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

  std::fill(ener.begin(), ener.end(), 0.0);
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  std::fill(atom_ener_.begin(), atom_ener_.end(), 0.0);
  std::fill(atom_vir_.begin(), atom_vir_.end(), 0.0);
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy2, atype_cpy,
             box, nall - nloc, inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1, nframes);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFrames, cpu_lmp_nlist_2rc) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc * 2);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  std::vector<double> ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(nframes * 9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

  std::fill(ener.begin(), ener.end(), 0.0);
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFrames, cpu_lmp_nlist_type_sel) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();

  // add vir atoms
  int nvir = 2;
  std::vector<VALUETYPE> coord_vir(nvir * 3);
  std::vector<int> atype_vir(nvir, 2);
  std::vector<VALUETYPE> aparam_vir(nvir * 1);
  for (int ii = 0; ii < nvir; ++ii) {
    coord_vir[ii] = coord[ii];
  }
  for (int ii = 0; ii < nvir; ++ii) {
    aparam_vir[ii] = aparam[ii];
  }
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), atype_vir.begin(), atype_vir.end());
  natoms += nvir;
  std::vector<VALUETYPE> expected_f_vir(nvir * 3, 0.0);
  // two frames
  expected_f.insert(expected_f.begin(), expected_f_vir.begin(),
                    expected_f_vir.end());
  expected_f.insert(expected_f.begin() + natoms * 3, expected_f_vir.begin(),
                    expected_f_vir.end());
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);

  // build nlist
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }
  aparam.insert(aparam.begin(), aparam_vir.begin(), aparam_vir.end());
  aparam.insert(aparam.begin() + nloc, aparam_vir.begin(), aparam_vir.end());

  // dp compute
  std::vector<double> ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(nframes * 9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  // fold back
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFrames,
           cpu_lmp_nlist_type_sel_atomic) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();

  // add vir atoms
  int nvir = 2;
  std::vector<VALUETYPE> coord_vir(nvir * 3);
  std::vector<int> atype_vir(nvir, 2);
  std::vector<VALUETYPE> aparam_vir(nvir * 2);
  for (int ii = 0; ii < nvir; ++ii) {
    coord_vir[ii] = coord[ii];
  }
  for (int ii = 0; ii < nvir * 2; ++ii) {
    aparam_vir[ii] = aparam[ii];
  }
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), atype_vir.begin(), atype_vir.end());
  aparam.insert(aparam.begin(), aparam_vir.begin(), aparam_vir.end());
  natoms += nvir;
  std::vector<VALUETYPE> expected_f_vir(nvir * 3, 0.0);
  // two frames
  expected_f.insert(expected_f.begin(), expected_f_vir.begin(),
                    expected_f_vir.end());
  expected_f.insert(expected_f.begin() + natoms * 3, expected_f_vir.begin(),
                    expected_f_vir.end());
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);

  // build nlist
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  // dp compute
  std::vector<double> ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(nframes * 9, 0.0),
      atomic_energy, atomic_virial;
  dp.compute(ener, force_, virial, atomic_energy, atomic_virial, coord_cpy2,
             atype_cpy, box, nall - nloc, inlist, 0, fparam, aparam);
  // fold back
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

template <class VALUETYPE>
class TestInferDeepPotAFparamAparamNFramesSingleParam : public ::testing::Test {
 protected:
  std::vector<VALUETYPE> coord = {
      12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56,
      12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  std::vector<int> atype = {0, 0, 0, 0, 0, 0};
  std::vector<VALUETYPE> box = {13., 0., 0., 0., 13., 0., 0., 0., 13.,
                                13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<VALUETYPE> fparam = {0.25852028};
  std::vector<VALUETYPE> aparam = {0.25852028, 0.25852028, 0.25852028,
                                   0.25852028, 0.25852028, 0.25852028};
  std::vector<VALUETYPE> expected_e = {
      -1.038271183039953804e-01, -7.285433575272914908e-02,
      -9.467600174099155552e-02, -1.467050086239614082e-01,
      -7.660561620618722145e-02, -7.277295998502930630e-02,
      -1.038271183039953804e-01, -7.285433575272914908e-02,
      -9.467600174099155552e-02, -1.467050086239614082e-01,
      -7.660561620618722145e-02, -7.277295998502930630e-02};
  std::vector<VALUETYPE> expected_f = {
      6.622266817497907132e-02,  5.278739055693523058e-02,
      2.265727495541422845e-02,  -2.606047850915838363e-02,
      -4.538811686410718776e-02, 1.058247569147072187e-02,
      1.679392490937766935e-01,  -2.257828022687320690e-03,
      -4.490145670355452645e-02, -1.148364103573685929e-01,
      -1.169790466695089237e-02, 6.140402504113953025e-02,
      -8.078778132132799494e-02, -5.838878056243369807e-02,
      6.773639989682191109e-02,  -1.247724708090079161e-02,
      6.494523955924384750e-02,  -1.174787188812918687e-01,
      6.622266817497907132e-02,  5.278739055693523058e-02,
      2.265727495541422845e-02,  -2.606047850915838363e-02,
      -4.538811686410718776e-02, 1.058247569147072187e-02,
      1.679392490937766935e-01,  -2.257828022687320690e-03,
      -4.490145670355452645e-02, -1.148364103573685929e-01,
      -1.169790466695089237e-02, 6.140402504113953025e-02,
      -8.078778132132799494e-02, -5.838878056243369807e-02,
      6.773639989682191109e-02,  -1.247724708090079161e-02,
      6.494523955924384750e-02,  -1.174787188812918687e-01};
  std::vector<VALUETYPE> expected_v = {
      -1.589185553287162656e-01, 2.586163333170100279e-03,
      -1.575127933809472624e-04, -1.855360380105876630e-02,
      1.949822090859933826e-02,  -1.006552056166355388e-02,
      3.177029853276916449e-02,  1.714349636720383010e-03,
      -1.290389175187874483e-03, -8.553510339477603253e-02,
      -5.654637257232508415e-03, -1.286954833787038420e-02,
      2.464156457499515687e-02,  -2.398202886026797043e-02,
      -1.957110465239037672e-02, 2.233492928605742764e-02,
      6.107843207824020099e-03,  1.707078295947736047e-03,
      -1.653994088976195043e-01, 3.894358678172111371e-02,
      -2.169595969759342477e-02, 6.819704294738503786e-03,
      -5.018242039618424008e-03, 2.640664428663210429e-03,
      -1.985298275686078057e-03, -3.638421609610945767e-02,
      2.342932331075030239e-02,  -8.501331914753691710e-02,
      -2.181253413538992297e-03, 4.311300069651782287e-03,
      -1.910329328333908129e-03, -1.808810159508548836e-03,
      -1.540075281450827612e-03, -1.173703213175551763e-02,
      -2.596306629910121507e-03, 6.705025662372287101e-03,
      -9.038455005073858795e-02, 3.011717773578577451e-02,
      -5.083054073419784880e-02, -2.951210292616929069e-03,
      2.342445652898489383e-02,  -4.091207474993674431e-02,
      -1.648470649301832236e-02, -2.872261885460645689e-02,
      4.763924972552112391e-02,  -8.300036532764677732e-02,
      1.020429228955421243e-03,  -1.026734151199098881e-03,
      5.678534096113684732e-02,  1.273635718045938205e-02,
      -1.530143225195957322e-02, -1.061671865629566225e-01,
      -2.486859433265622629e-02, 2.875323131744185121e-02,
      -1.589185553287162656e-01, 2.586163333170100279e-03,
      -1.575127933809472624e-04, -1.855360380105876630e-02,
      1.949822090859933826e-02,  -1.006552056166355388e-02,
      3.177029853276916449e-02,  1.714349636720383010e-03,
      -1.290389175187874483e-03, -8.553510339477603253e-02,
      -5.654637257232508415e-03, -1.286954833787038420e-02,
      2.464156457499515687e-02,  -2.398202886026797043e-02,
      -1.957110465239037672e-02, 2.233492928605742764e-02,
      6.107843207824020099e-03,  1.707078295947736047e-03,
      -1.653994088976195043e-01, 3.894358678172111371e-02,
      -2.169595969759342477e-02, 6.819704294738503786e-03,
      -5.018242039618424008e-03, 2.640664428663210429e-03,
      -1.985298275686078057e-03, -3.638421609610945767e-02,
      2.342932331075030239e-02,  -8.501331914753691710e-02,
      -2.181253413538992297e-03, 4.311300069651782287e-03,
      -1.910329328333908129e-03, -1.808810159508548836e-03,
      -1.540075281450827612e-03, -1.173703213175551763e-02,
      -2.596306629910121507e-03, 6.705025662372287101e-03,
      -9.038455005073858795e-02, 3.011717773578577451e-02,
      -5.083054073419784880e-02, -2.951210292616929069e-03,
      2.342445652898489383e-02,  -4.091207474993674431e-02,
      -1.648470649301832236e-02, -2.872261885460645689e-02,
      4.763924972552112391e-02,  -8.300036532764677732e-02,
      1.020429228955421243e-03,  -1.026734151199098881e-03,
      5.678534096113684732e-02,  1.273635718045938205e-02,
      -1.530143225195957322e-02, -1.061671865629566225e-01,
      -2.486859433265622629e-02, 2.875323131744185121e-02};
  int natoms;
  int nframes = 2;
  std::vector<double> expected_tot_e;
  std::vector<VALUETYPE> expected_tot_v;

  deepmd::hpp::DeepPot dp;

  void SetUp() override {
    std::string file_name = "../../tests/infer/fparam_aparam.pbtxt";
    deepmd::hpp::convert_pbtxt_to_pb("../../tests/infer/fparam_aparam.pbtxt",
                                     "fparam_aparam.pb");

    dp.init("fparam_aparam.pb");

    natoms = expected_e.size() / nframes;
    EXPECT_EQ(nframes * natoms * 3, expected_f.size());
    EXPECT_EQ(nframes * natoms * 9, expected_v.size());
    expected_tot_e.resize(nframes);
    expected_tot_v.resize(nframes * 9);
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

  void TearDown() override { remove("fparam_aparam.pb"); };
};

TYPED_TEST_SUITE(TestInferDeepPotAFparamAparamNFramesSingleParam, ValueTypes);

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesSingleParam, cpu_build_nlist) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
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

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesSingleParam,
           cpu_build_nlist_atomic) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
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

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesSingleParam, cpu_lmp_nlist) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  std::vector<double> ener;
  std::vector<VALUETYPE> force_, virial;
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

  std::fill(ener.begin(), ener.end(), 0.0);
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesSingleParam,
           cpu_lmp_nlist_atomic) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  std::vector<double> ener;
  std::vector<VALUETYPE> force_, atom_ener_, atom_vir_, virial;
  std::vector<VALUETYPE> force, atom_ener, atom_vir;
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy2, atype_cpy,
             box, nall - nloc, inlist, 0, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1, nframes);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9, nframes);

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

  std::fill(ener.begin(), ener.end(), 0.0);
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  std::fill(atom_ener_.begin(), atom_ener_.end(), 0.0);
  std::fill(atom_vir_.begin(), atom_vir_.end(), 0.0);
  dp.compute(ener, force_, virial, atom_ener_, atom_vir_, coord_cpy2, atype_cpy,
             box, nall - nloc, inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);
  _fold_back<VALUETYPE>(atom_ener, atom_ener_, mapping, nloc, nall, 1, nframes);
  _fold_back<VALUETYPE>(atom_vir, atom_vir_, mapping, nloc, nall, 9, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesSingleParam, cpu_lmp_nlist_2rc) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc * 2);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  std::vector<double> ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(nframes * 9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

  std::fill(ener.begin(), ener.end(), 0.0);
  std::fill(force_.begin(), force_.end(), 0.0);
  std::fill(virial.begin(), virial.end(), 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 1, fparam, aparam);
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesSingleParam,
           cpu_lmp_nlist_type_sel) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();

  // add vir atoms
  int nvir = 2;
  std::vector<VALUETYPE> coord_vir(nvir * 3);
  std::vector<int> atype_vir(nvir, 2);
  std::vector<VALUETYPE> aparam_vir(nvir * 1);
  for (int ii = 0; ii < nvir; ++ii) {
    coord_vir[ii] = coord[ii];
  }
  for (int ii = 0; ii < nvir; ++ii) {
    aparam_vir[ii] = aparam[ii];
  }
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), atype_vir.begin(), atype_vir.end());
  natoms += nvir;
  std::vector<VALUETYPE> expected_f_vir(nvir * 3, 0.0);
  // two frames
  expected_f.insert(expected_f.begin(), expected_f_vir.begin(),
                    expected_f_vir.end());
  expected_f.insert(expected_f.begin() + natoms * 3, expected_f_vir.begin(),
                    expected_f_vir.end());
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);

  // build nlist
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }
  aparam.insert(aparam.begin(), aparam_vir.begin(), aparam_vir.end());

  // dp compute
  std::vector<double> ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(nframes * 9, 0.0);
  dp.compute(ener, force_, virial, coord_cpy2, atype_cpy, box, nall - nloc,
             inlist, 0, fparam, aparam);
  // fold back
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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

TYPED_TEST(TestInferDeepPotAFparamAparamNFramesSingleParam,
           cpu_lmp_nlist_type_sel_atomic) {
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
  deepmd::hpp::DeepPot& dp = this->dp;
  float rc = dp.cutoff();

  // add vir atoms
  int nvir = 2;
  std::vector<VALUETYPE> coord_vir(nvir * 3);
  std::vector<int> atype_vir(nvir, 2);
  std::vector<VALUETYPE> aparam_vir(nvir);
  for (int ii = 0; ii < nvir; ++ii) {
    coord_vir[ii] = coord[ii];
  }
  for (int ii = 0; ii < nvir; ++ii) {
    aparam_vir[ii] = aparam[ii];
  }
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), atype_vir.begin(), atype_vir.end());
  aparam.insert(aparam.begin(), aparam_vir.begin(), aparam_vir.end());
  natoms += nvir;
  std::vector<VALUETYPE> expected_f_vir(nvir * 3, 0.0);
  // two frames
  expected_f.insert(expected_f.begin(), expected_f_vir.begin(),
                    expected_f_vir.end());
  expected_f.insert(expected_f.begin() + natoms * 3, expected_f_vir.begin(),
                    expected_f_vir.end());
  std::vector<VALUETYPE> coord_first(coord.begin(), coord.begin() + 3 * natoms);
  std::vector<VALUETYPE> box_first(box.begin(), box.begin() + 9);

  // build nlist
  int nloc = coord_first.size() / 3;
  std::vector<VALUETYPE> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int> > nlist_data;
  _build_nlist<VALUETYPE>(nlist_data, coord_cpy, atype_cpy, mapping,
                          coord_first, atype, box_first, rc);
  int nall = coord_cpy.size() / 3;
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::hpp::InputNlist inlist(nloc, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(inlist, nlist_data);
  std::vector<VALUETYPE> coord_cpy2(nframes * nall * 3);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord_cpy2[ii * nall * 3 + jj] = coord_cpy[jj];
    }
  }

  // dp compute
  std::vector<double> ener;
  std::vector<VALUETYPE> force_(nall * 3, 0.0), virial(nframes * 9, 0.0),
      atomic_energy, atomic_virial;
  dp.compute(ener, force_, virial, atomic_energy, atomic_virial, coord_cpy2,
             atype_cpy, box, nall - nloc, inlist, 0, fparam, aparam);
  // fold back
  std::vector<VALUETYPE> force;
  _fold_back<VALUETYPE>(force, force_, mapping, nloc, nall, 3, nframes);

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
