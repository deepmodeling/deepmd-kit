// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "../../api_cc/tests/expected_ref.h"
#include "../../tests/infer/deeppot_universal_data.h"
#include "DeepPotPTExpt.h"
#include "c_api.h"

namespace {

enum class Backend { TensorFlow, PyTorch, PTExpt, JAX, Paddle };

struct ModelCase {
  std::string name;
  Backend backend;
  std::string model_path;
  bool convert_pbtxt;
  const deepmd_test::DeepPotRef* ref;
  const deepmd_test::DeepPotRef* no_pbc_ref;
  double double_tol;
  double float_tol;
  bool supports_float;
};

struct FParamAParamCase {
  std::string name;
  Backend backend;
  std::string model_path;
  bool convert_pbtxt;
  const deepmd_test::DeepPotRef* builtin_ref;
  std::string ref_path;
  double double_tol;
  double float_tol;
  bool supports_float;
};

bool path_exists(const std::string& path) {
  struct stat statbuf;
  return stat(path.c_str(), &statbuf) == 0;
}

bool backend_enabled(Backend backend) {
  switch (backend) {
    case Backend::TensorFlow:
#ifdef BUILD_TENSORFLOW
      return true;
#else
      return false;
#endif
    case Backend::PyTorch:
#ifdef BUILD_PYTORCH
      return true;
#else
      return false;
#endif
    case Backend::PTExpt:
#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT
      return true;
#else
      return false;
#endif
    case Backend::JAX:
#ifdef BUILD_JAX
      return true;
#else
      return false;
#endif
    case Backend::Paddle:
#ifdef BUILD_PADDLE
      return true;
#else
      return false;
#endif
  }
  return false;
}

std::string backend_name(Backend backend) {
  switch (backend) {
    case Backend::TensorFlow:
      return "TensorFlow";
    case Backend::PyTorch:
      return "PyTorch";
    case Backend::PTExpt:
      return "PTExpt";
    case Backend::JAX:
      return "JAX";
    case Backend::Paddle:
      return "Paddle";
  }
  return "Unknown";
}

std::vector<ModelCase> model_cases() {
  return {
      {"tensorflow_pb", Backend::TensorFlow, "../../tests/infer/deeppot.pbtxt",
       true, &deepmd_test::tf_deeppot_ref(),
       &deepmd_test::tf_deeppot_no_pbc_ref(), 1e-10, 1e-4, true},
      {"pytorch_pth", Backend::PyTorch, "../../tests/infer/deeppot_sea.pth",
       false, &deepmd_test::sea_deeppot_ref(),
       &deepmd_test::sea_deeppot_no_pbc_ref(), 1e-10, 1e-4, true},
      {"pytorch_pt2", Backend::PTExpt, "../../tests/infer/deeppot_sea.pt2",
       false, &deepmd_test::sea_deeppot_ref(),
       &deepmd_test::sea_deeppot_no_pbc_ref(), 1e-10, 1e-4, true},
      {"jax_savedmodel", Backend::JAX,
       "../../tests/infer/deeppot_sea.savedmodel", false,
       &deepmd_test::sea_deeppot_ref(), nullptr, 1e-10, 1e-4, true},
      {"paddle_json", Backend::Paddle, "../../tests/infer/deeppot_sea.json",
       false, &deepmd_test::sea_deeppot_ref(), nullptr, 1e-7, 1e-4, false}};
}

std::vector<FParamAParamCase> fparam_aparam_cases() {
  return {{"tensorflow_pb", Backend::TensorFlow,
           "../../tests/infer/fparam_aparam.pbtxt", true,
           &deepmd_test::tf_fparam_aparam_ref(), "", 1e-10, 1e-4, true},
          {"pytorch_pth", Backend::PyTorch,
           "../../tests/infer/fparam_aparam.pth", false, nullptr,
           "../../tests/infer/fparam_aparam.expected", 1e-7, 1e-4, true},
          {"pytorch_pt2", Backend::PTExpt,
           "../../tests/infer/fparam_aparam.pt2", false, nullptr,
           "../../tests/infer/fparam_aparam.expected", 1e-7, 1e-4, true}};
}

deepmd_test::DeepPotRef load_fparam_ref(const std::string& ref_path) {
  deepmd_test::ExpectedRef ref_file;
  ref_file.load(ref_path);
  deepmd_test::DeepPotRef ref;
  ref.atomic_energy = ref_file.get<double>("default", "expected_e");
  ref.force = ref_file.get<double>("default", "expected_f");
  ref.atomic_virial = ref_file.get<double>("default", "expected_v");
  ref.numb_types = 1;
  ref.dim_fparam = 1;
  ref.dim_aparam = 1;
  ref.type_map = "O";
  return ref;
}

class UniversalDeepPotCTest : public ::testing::TestWithParam<ModelCase> {
 protected:
  DP_DeepPot* dp = nullptr;
  std::string converted_model;

  void SetUp() override {
    const auto& param = GetParam();
    if (!backend_enabled(param.backend)) {
      GTEST_SKIP() << backend_name(param.backend) << " support is not enabled.";
    }
    ASSERT_TRUE(path_exists(param.model_path))
        << "Model artifact is not available: " << param.model_path;

    std::string model_path = param.model_path;
    if (param.convert_pbtxt) {
      converted_model = "deeppot_c_universal_" + param.name + ".pb";
      DP_ConvertPbtxtToPb(param.model_path.c_str(), converted_model.c_str());
      model_path = converted_model;
    }
    dp = DP_NewDeepPot(model_path.c_str());
    const char* error = DP_DeepPotCheckOK(dp);
    const std::string error_message(error);
    DP_DeleteChar(error);
    ASSERT_TRUE(error_message.empty())
        << "Model artifact cannot be loaded by this backend: " << error_message;
  }

  void TearDown() override {
    DP_DeleteDeepPot(dp);
    if (!converted_model.empty()) {
      remove(converted_model.c_str());
    }
  }
};

class FParamAParamDeepPotCTest
    : public ::testing::TestWithParam<FParamAParamCase> {
 protected:
  DP_DeepPot* dp = nullptr;
  std::string converted_model;
  deepmd_test::DeepPotRef loaded_ref;
  const deepmd_test::DeepPotRef* ref = nullptr;

  void SetUp() override {
    const auto& param = GetParam();
    if (!backend_enabled(param.backend)) {
      GTEST_SKIP() << backend_name(param.backend) << " support is not enabled.";
    }
    ASSERT_TRUE(path_exists(param.model_path))
        << "Model artifact is not available: " << param.model_path;
    if (param.builtin_ref != nullptr) {
      ref = param.builtin_ref;
    } else {
      ASSERT_TRUE(path_exists(param.ref_path))
          << "Reference artifact is not available: " << param.ref_path;
      loaded_ref = load_fparam_ref(param.ref_path);
      ref = &loaded_ref;
    }

    std::string model_path = param.model_path;
    if (param.convert_pbtxt) {
      converted_model = "deeppot_c_fparam_aparam_" + param.name + ".pb";
      DP_ConvertPbtxtToPb(param.model_path.c_str(), converted_model.c_str());
      model_path = converted_model;
    }
    dp = DP_NewDeepPot(model_path.c_str());
    const char* error = DP_DeepPotCheckOK(dp);
    const std::string error_message(error);
    DP_DeleteChar(error);
    ASSERT_TRUE(error_message.empty())
        << "Model artifact cannot be loaded by this backend: " << error_message;
  }

  void TearDown() override {
    DP_DeleteDeepPot(dp);
    if (!converted_model.empty()) {
      remove(converted_model.c_str());
    }
  }
};

TEST_P(UniversalDeepPotCTest, Metadata) {
  const auto& ref = *GetParam().ref;

  EXPECT_DOUBLE_EQ(DP_DeepPotGetCutoff(dp), ref.cutoff);
  EXPECT_EQ(DP_DeepPotGetNumbTypes(dp), ref.numb_types);
  EXPECT_EQ(DP_DeepPotGetNumbTypesSpin(dp), ref.numb_types_spin);
  EXPECT_EQ(DP_DeepPotGetDimFParam(dp), ref.dim_fparam);
  EXPECT_EQ(DP_DeepPotGetDimAParam(dp), ref.dim_aparam);
  EXPECT_EQ(DP_DeepPotIsAParamNAll(dp), ref.aparam_nall);
  EXPECT_EQ(DP_DeepPotHasDefaultFParam(dp), ref.has_default_fparam);

  const char* type_map = DP_DeepPotGetTypeMap(dp);
  EXPECT_STREQ(type_map, ref.type_map.c_str());
  DP_DeleteChar(type_map);
}

void check_compute_double(DP_DeepPot* dp,
                          const deepmd_test::DeepPotRef& ref,
                          const double tol) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<double>& coord = deepmd_test::deeppot_coord();
  const std::vector<double>& box = deepmd_test::deeppot_box();
  const std::vector<int>& atype = deepmd_test::deeppot_atype();
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  double energy = 0.0;
  std::vector<double> force(natoms * 3);
  std::vector<double> virial(9);
  std::vector<double> atomic_energy(natoms);
  std::vector<double> atomic_virial(natoms * 9);
  DP_DeepPotCompute2(dp, 1, natoms, coord.data(), atype.data(), box.data(),
                     nullptr, nullptr, &energy, force.data(), virial.data(),
                     atomic_energy.data(), atomic_virial.data());

  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_NEAR(force[ii], ref.force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_NEAR(atomic_energy[ii], ref.atomic_energy[ii], tol);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_NEAR(atomic_virial[ii], ref.atomic_virial[ii], tol);
  }
}

void check_compute_float(DP_DeepPot* dp,
                         const deepmd_test::DeepPotRef& ref,
                         const double tol) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<float> coord(deepmd_test::deeppot_coord().begin(),
                                 deepmd_test::deeppot_coord().end());
  const std::vector<float> box(deepmd_test::deeppot_box().begin(),
                               deepmd_test::deeppot_box().end());
  const std::vector<int>& atype = deepmd_test::deeppot_atype();
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  double energy = 0.0;
  std::vector<float> force(natoms * 3);
  std::vector<float> virial(9);
  std::vector<float> atomic_energy(natoms);
  std::vector<float> atomic_virial(natoms * 9);
  DP_DeepPotComputef2(dp, 1, natoms, coord.data(), atype.data(), box.data(),
                      nullptr, nullptr, &energy, force.data(), virial.data(),
                      atomic_energy.data(), atomic_virial.data());

  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_NEAR(force[ii], ref.force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_NEAR(atomic_energy[ii], ref.atomic_energy[ii], tol);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_NEAR(atomic_virial[ii], ref.atomic_virial[ii], tol);
  }
}

void check_compute_legacy_double(DP_DeepPot* dp,
                                 const deepmd_test::DeepPotRef& ref,
                                 const double* box,
                                 const double tol) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<double>& coord = deepmd_test::deeppot_coord();
  const std::vector<int>& atype = deepmd_test::deeppot_atype();
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  double energy = 0.0;
  std::vector<double> force(natoms * 3);
  std::vector<double> virial(9);
  std::vector<double> atomic_energy(natoms);
  std::vector<double> atomic_virial(natoms * 9);
  DP_DeepPotCompute(dp, natoms, coord.data(), atype.data(), box, &energy,
                    force.data(), virial.data(), atomic_energy.data(),
                    atomic_virial.data());

  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_NEAR(force[ii], ref.force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_NEAR(atomic_energy[ii], ref.atomic_energy[ii], tol);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_NEAR(atomic_virial[ii], ref.atomic_virial[ii], tol);
  }
}

void check_compute_legacy_float(DP_DeepPot* dp,
                                const deepmd_test::DeepPotRef& ref,
                                const float* box,
                                const double tol) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<float> coord(deepmd_test::deeppot_coord().begin(),
                                 deepmd_test::deeppot_coord().end());
  const std::vector<int>& atype = deepmd_test::deeppot_atype();
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  double energy = 0.0;
  std::vector<float> force(natoms * 3);
  std::vector<float> virial(9);
  std::vector<float> atomic_energy(natoms);
  std::vector<float> atomic_virial(natoms * 9);
  DP_DeepPotComputef(dp, natoms, coord.data(), atype.data(), box, &energy,
                     force.data(), virial.data(), atomic_energy.data(),
                     atomic_virial.data());

  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_NEAR(force[ii], ref.force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_NEAR(atomic_energy[ii], ref.atomic_energy[ii], tol);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_NEAR(atomic_virial[ii], ref.atomic_virial[ii], tol);
  }
}

void check_fparam_compute_double(DP_DeepPot* dp,
                                 const deepmd_test::DeepPotRef& ref,
                                 const double tol) {
  const int natoms =
      static_cast<int>(deepmd_test::fparam_aparam_atype().size());
  const std::vector<double>& coord = deepmd_test::deeppot_coord();
  const std::vector<double>& box = deepmd_test::deeppot_box();
  const std::vector<int>& atype = deepmd_test::fparam_aparam_atype();
  const std::vector<double>& fparam = deepmd_test::fparam_value();
  const std::vector<double>& aparam = deepmd_test::aparam_value();
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  double energy = 0.0;
  std::vector<double> force(natoms * 3);
  std::vector<double> virial(9);
  std::vector<double> atomic_energy(natoms);
  std::vector<double> atomic_virial(natoms * 9);
  DP_DeepPotCompute2(dp, 1, natoms, coord.data(), atype.data(), box.data(),
                     fparam.data(), aparam.data(), &energy, force.data(),
                     virial.data(), atomic_energy.data(), atomic_virial.data());

  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_NEAR(force[ii], ref.force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_NEAR(atomic_energy[ii], ref.atomic_energy[ii], tol);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_NEAR(atomic_virial[ii], ref.atomic_virial[ii], tol);
  }
}

void check_fparam_compute_float(DP_DeepPot* dp,
                                const deepmd_test::DeepPotRef& ref,
                                const double tol) {
  const int natoms =
      static_cast<int>(deepmd_test::fparam_aparam_atype().size());
  const std::vector<float> coord(deepmd_test::deeppot_coord().begin(),
                                 deepmd_test::deeppot_coord().end());
  const std::vector<float> box(deepmd_test::deeppot_box().begin(),
                               deepmd_test::deeppot_box().end());
  const std::vector<int>& atype = deepmd_test::fparam_aparam_atype();
  const std::vector<float> fparam(deepmd_test::fparam_value().begin(),
                                  deepmd_test::fparam_value().end());
  const std::vector<float> aparam(deepmd_test::aparam_value().begin(),
                                  deepmd_test::aparam_value().end());
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  double energy = 0.0;
  std::vector<float> force(natoms * 3);
  std::vector<float> virial(9);
  std::vector<float> atomic_energy(natoms);
  std::vector<float> atomic_virial(natoms * 9);
  DP_DeepPotComputef2(dp, 1, natoms, coord.data(), atype.data(), box.data(),
                      fparam.data(), aparam.data(), &energy, force.data(),
                      virial.data(), atomic_energy.data(),
                      atomic_virial.data());

  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_NEAR(force[ii], ref.force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_NEAR(atomic_energy[ii], ref.atomic_energy[ii], tol);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_NEAR(atomic_virial[ii], ref.atomic_virial[ii], tol);
  }
}

TEST_P(UniversalDeepPotCTest, ComputeDouble) {
  check_compute_double(dp, *GetParam().ref, GetParam().double_tol);
}

TEST_P(UniversalDeepPotCTest, ComputeFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_compute_float(dp, *GetParam().ref, GetParam().float_tol);
}

TEST_P(UniversalDeepPotCTest, ComputeLegacyDouble) {
  const std::vector<double>& box = deepmd_test::deeppot_box();
  check_compute_legacy_double(dp, *GetParam().ref, box.data(),
                              GetParam().double_tol);
}

TEST_P(UniversalDeepPotCTest, ComputeLegacyFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  const std::vector<double>& box_double = deepmd_test::deeppot_box();
  const std::vector<float> box(box_double.begin(), box_double.end());
  check_compute_legacy_float(dp, *GetParam().ref, box.data(),
                             GetParam().float_tol);
}

TEST_P(UniversalDeepPotCTest, ComputeLegacyNoPbcDouble) {
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute_legacy_double(dp, *GetParam().no_pbc_ref, nullptr,
                              GetParam().double_tol);
}

TEST_P(UniversalDeepPotCTest, ComputeLegacyNoPbcFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute_legacy_float(dp, *GetParam().no_pbc_ref, nullptr,
                             GetParam().float_tol);
}

TEST_P(FParamAParamDeepPotCTest, Metadata) {
  EXPECT_EQ(DP_DeepPotGetDimFParam(dp), 1);
  EXPECT_EQ(DP_DeepPotGetDimAParam(dp), 1);
  EXPECT_FALSE(DP_DeepPotHasDefaultFParam(dp));
}

TEST_P(FParamAParamDeepPotCTest, ComputeDouble) {
  check_fparam_compute_double(dp, *ref, GetParam().double_tol);
}

TEST_P(FParamAParamDeepPotCTest, ComputeFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_compute_float(dp, *ref, GetParam().float_tol);
}

INSTANTIATE_TEST_SUITE_P(
    AvailableBackends,
    UniversalDeepPotCTest,
    ::testing::ValuesIn(model_cases()),
    [](const ::testing::TestParamInfo<UniversalDeepPotCTest::ParamType>& info) {
      return info.param.name;
    });

INSTANTIATE_TEST_SUITE_P(
    FParamAParamBackends,
    FParamAParamDeepPotCTest,
    ::testing::ValuesIn(fparam_aparam_cases()),
    [](const ::testing::TestParamInfo<FParamAParamDeepPotCTest::ParamType>&
           info) { return info.param.name; });

}  // namespace
