// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "c_api.h"

class TestInferDeepPotA : public ::testing::Test {
 protected:
  double coord[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  int atype[6] = {0, 1, 1, 0, 1, 1};
  double box[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  float boxf[9] = {13., 0., 0., 0., 13., 0., 0., 0., 13.};
  std::vector<double> expected_e = {
      -9.275780747115504710e+01, -1.863501786584258468e+02,
      -1.863392472863538103e+02, -9.279281325486221021e+01,
      -1.863671545232153903e+02, -1.863619822847602165e+02};
  std::vector<double> expected_f = {
      -3.034045420701179663e-01, 8.405844663871177014e-01,
      7.696947487118485642e-02,  7.662001266663505117e-01,
      -1.880601391333554251e-01, -6.183333871091722944e-01,
      -5.036172391059643427e-01, -6.529525836149027151e-01,
      5.432962643022043459e-01,  6.382357912332115024e-01,
      -1.748518296794561167e-01, 3.457363524891907125e-01,
      1.286482986991941552e-03,  3.757251165286925043e-01,
      -5.972588700887541124e-01, -5.987006197104716154e-01,
      -2.004450304880958100e-01, 2.495901655353461868e-01};
  std::vector<double> expected_v = {
      -2.912234126853306959e-01, -3.800610846612756388e-02,
      2.776624987489437202e-01,  -5.053761003913598976e-02,
      -3.152373041953385746e-01, 1.060894290092162379e-01,
      2.826389131596073745e-01,  1.039129970665329250e-01,
      -2.584378792325942586e-01, -3.121722367954994914e-01,
      8.483275876786681990e-02,  2.524662342344257682e-01,
      4.142176771106586414e-02,  -3.820285230785245428e-02,
      -2.727311173065460545e-02, 2.668859789777112135e-01,
      -6.448243569420382404e-02, -2.121731470426218846e-01,
      -8.624335220278558922e-02, -1.809695356746038597e-01,
      1.529875294531883312e-01,  -1.283658185172031341e-01,
      -1.992682279795223999e-01, 1.409924999632362341e-01,
      1.398322735274434292e-01,  1.804318474574856390e-01,
      -1.470309318999652726e-01, -2.593983661598450730e-01,
      -4.236536279233147489e-02, 3.386387920184946720e-02,
      -4.174017537818433543e-02, -1.003500282164128260e-01,
      1.525690815194478966e-01,  3.398976109910181037e-02,
      1.522253908435125536e-01,  -2.349125581341701963e-01,
      9.515545977581392825e-04,  -1.643218849228543846e-02,
      1.993234765412972564e-02,  6.027265332209678569e-04,
      -9.563256398907417355e-02, 1.510815124001868293e-01,
      -7.738094816888557714e-03, 1.502832772532304295e-01,
      -2.380965783745832010e-01, -2.309456719810296654e-01,
      -6.666961081213038098e-02, 7.955566551234216632e-02,
      -8.099093777937517447e-02, -3.386641099800401927e-02,
      4.447884755740908608e-02,  1.008593228579038742e-01,
      4.556718179228393811e-02,  -6.078081273849572641e-02};
  int natoms;
  double expected_tot_e;
  std::vector<double> expected_tot_v;

  DP_DeepPot* dp;

  void SetUp() override {
    const char* file_name = "../../tests/infer/deeppot.pbtxt";
    const char* model_file = "deeppot.pb";
    DP_ConvertPbtxtToPb(file_name, model_file);

    dp = DP_NewDeepPot(model_file);

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.resize(9);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override {
    remove("deeppot.pb");
    DP_DeleteDeepPot(dp);
  };
};

TEST_F(TestInferDeepPotA, double_infer) {
  double* ener_ = new double;
  double* force_ = new double[natoms * 3];
  double* virial_ = new double[9];
  double* atomic_ener_ = new double[natoms];
  double* atomic_virial_ = new double[natoms * 9];

  DP_DeepPotCompute(dp, natoms, coord, atype, box, ener_, force_, virial_,
                    atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<double> force(force_, force_ + natoms * 3);
  std::vector<double> virial(virial_, virial_ + 9);
  std::vector<double> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<double> atomic_virial(atomic_virial_,
                                    atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-10);
  }

  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepPotA, float_infer) {
  double* ener_ = new double;
  float* force_ = new float[natoms * 3];
  float* virial_ = new float[9];
  float* atomic_ener_ = new float[natoms];
  float* atomic_virial_ = new float[natoms * 9];

  DP_DeepPotComputef(dp, natoms, coordf, atype, boxf, ener_, force_, virial_,
                     atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<float> force(force_, force_ + natoms * 3);
  std::vector<float> virial(virial_, virial_ + 9);
  std::vector<float> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<float> atomic_virial(atomic_virial_, atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-6);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-6);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-6);
  }

  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-5);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-6);
  }
  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepPotA, cutoff) {
  double cutoff = DP_DeepPotGetCutoff(dp);
  EXPECT_EQ(cutoff, 6.0);
}

TEST_F(TestInferDeepPotA, numb_types) {
  int numb_types = DP_DeepPotGetNumbTypes(dp);
  EXPECT_EQ(numb_types, 2);
}

TEST_F(TestInferDeepPotA, numb_types_spin) {
  int numb_types_spin = DP_DeepPotGetNumbTypesSpin(dp);
  EXPECT_EQ(numb_types_spin, 0);
}

TEST_F(TestInferDeepPotA, type_map) {
  const char* type_map = DP_DeepPotGetTypeMap(dp);
  char expected_type_map[] = "O H";
  EXPECT_EQ(strcmp(type_map, expected_type_map), 0);
  DP_DeleteChar(type_map);
}

class TestInferDeepPotANoPBC : public ::testing::Test {
 protected:
  double coord[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  float coordf[18] = {12.83, 2.56, 2.18, 12.09, 2.87, 2.74, 00.25, 3.32, 1.68,
                      3.36,  3.00, 1.81, 3.51,  2.51, 2.60, 4.27,  3.22, 1.56};
  int atype[6] = {0, 1, 1, 0, 1, 1};
  std::vector<double> expected_e = {
      -9.255934839310273787e+01, -1.863253376736990106e+02,
      -1.857237299341402945e+02, -9.279308539717486326e+01,
      -1.863708105823244239e+02, -1.863635196514972563e+02};
  std::vector<double> expected_f = {
      -2.161037360255332107e+00, 9.052994347015581589e-01,
      1.635379623977007979e+00,  2.161037360255332107e+00,
      -9.052994347015581589e-01, -1.635379623977007979e+00,
      -1.167128117249453811e-02, 1.371975700096064992e-03,
      -1.575265180249604477e-03, 6.226508593971802341e-01,
      -1.816734122009256991e-01, 3.561766019664774907e-01,
      -1.406075393906316626e-02, 3.789140061530929526e-01,
      -6.018777878642909140e-01, -5.969188242856223736e-01,
      -1.986125696522633155e-01, 2.472764510780630642e-01};
  std::vector<double> expected_v = {
      -7.042445481792056761e-01, 2.950213647777754078e-01,
      5.329418202437231633e-01,  2.950213647777752968e-01,
      -1.235900311906896754e-01, -2.232594111831812944e-01,
      5.329418202437232743e-01,  -2.232594111831813499e-01,
      -4.033073234276823849e-01, -8.949230984097404917e-01,
      3.749002169013777030e-01,  6.772391014992630298e-01,
      3.749002169013777586e-01,  -1.570527935667933583e-01,
      -2.837082722496912512e-01, 6.772391014992631408e-01,
      -2.837082722496912512e-01, -5.125052659994422388e-01,
      4.858210330291591605e-02,  -6.902596153269104431e-03,
      6.682612642430500391e-03,  -5.612247004554610057e-03,
      9.767795567660207592e-04,  -9.773758942738038254e-04,
      5.638322117219018645e-03,  -9.483806049779926932e-04,
      8.493873281881353637e-04,  -2.941738570564985666e-01,
      -4.482529909499673171e-02, 4.091569840186781021e-02,
      -4.509020615859140463e-02, -1.013919988807244071e-01,
      1.551440772665269030e-01,  4.181857726606644232e-02,
      1.547200233064863484e-01,  -2.398213304685777592e-01,
      -3.218625798524068354e-02, -1.012438450438508421e-02,
      1.271639330380921855e-02,  3.072814938490859779e-03,
      -9.556241797915024372e-02, 1.512251983492413077e-01,
      -8.277872384009607454e-03, 1.505412040827929787e-01,
      -2.386150620881526407e-01, -2.312295470054945568e-01,
      -6.631490213524345034e-02, 7.932427266386249398e-02,
      -8.053754366323923053e-02, -3.294595881137418747e-02,
      4.342495071150231922e-02,  1.004599500126941436e-01,
      4.450400364869536163e-02,  -5.951077548033092968e-02};
  int natoms;
  double expected_tot_e;
  std::vector<double> expected_tot_v;

  DP_DeepPot* dp;

  void SetUp() override {
    const char* file_name = "../../tests/infer/deeppot.pbtxt";
    const char* model_file = "deeppot.pb";
    DP_ConvertPbtxtToPb(file_name, model_file);

    dp = DP_NewDeepPot(model_file);

    natoms = expected_e.size();
    EXPECT_EQ(natoms * 3, expected_f.size());
    EXPECT_EQ(natoms * 9, expected_v.size());
    expected_tot_e = 0.;
    expected_tot_v.resize(9);
    std::fill(expected_tot_v.begin(), expected_tot_v.end(), 0.);
    for (int ii = 0; ii < natoms; ++ii) {
      expected_tot_e += expected_e[ii];
    }
    for (int ii = 0; ii < natoms; ++ii) {
      for (int dd = 0; dd < 9; ++dd) {
        expected_tot_v[dd] += expected_v[ii * 9 + dd];
      }
    }
  };

  void TearDown() override {
    remove("deeppot.pb");

    DP_DeleteDeepPot(dp);
  };
};

TEST_F(TestInferDeepPotANoPBC, double_infer) {
  double* ener_ = new double;
  double* force_ = new double[natoms * 3];
  double* virial_ = new double[9];
  double* atomic_ener_ = new double[natoms];
  double* atomic_virial_ = new double[natoms * 9];

  DP_DeepPotCompute(dp, natoms, coord, atype, nullptr, ener_, force_, virial_,
                    atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<double> force(force_, force_ + natoms * 3);
  std::vector<double> virial(virial_, virial_ + 9);
  std::vector<double> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<double> atomic_virial(atomic_virial_,
                                    atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-10);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-10);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-10);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-10);
  }
  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}

TEST_F(TestInferDeepPotANoPBC, float_infer) {
  double* ener_ = new double;
  float* force_ = new float[natoms * 3];
  float* virial_ = new float[9];
  float* atomic_ener_ = new float[natoms];
  float* atomic_virial_ = new float[natoms * 9];

  DP_DeepPotComputef(dp, natoms, coordf, atype, nullptr, ener_, force_, virial_,
                     atomic_ener_, atomic_virial_);

  double ener = *ener_;
  std::vector<float> force(force_, force_ + natoms * 3);
  std::vector<float> virial(virial_, virial_ + 9);
  std::vector<float> atomic_ener(atomic_ener_, atomic_ener_ + natoms);
  std::vector<float> atomic_virial(atomic_virial_, atomic_virial_ + natoms * 9);

  EXPECT_LT(fabs(ener - expected_tot_e), 1e-6);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_LT(fabs(force[ii] - expected_f[ii]), 1e-6);
  }
  for (int ii = 0; ii < 3 * 3; ++ii) {
    EXPECT_LT(fabs(virial[ii] - expected_tot_v[ii]), 1e-6);
  }

  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_LT(fabs(atomic_ener[ii] - expected_e[ii]), 1e-5);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_LT(fabs(atomic_virial[ii] - expected_v[ii]), 1e-6);
  }
  delete ener_;
  delete[] force_;
  delete[] virial_;
  delete[] atomic_ener_;
  delete[] atomic_virial_;
}
