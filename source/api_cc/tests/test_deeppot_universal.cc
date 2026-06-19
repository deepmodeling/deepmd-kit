// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>
#include <sys/stat.h>

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "../../tests/infer/deeppot_universal_data.h"
#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "test_utils.h"

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

class UniversalDeepPotTest : public ::testing::TestWithParam<ModelCase> {
 protected:
  deepmd::DeepPot dp;
  std::string converted_model;

  void SetUp() override {
    const auto& param = GetParam();
    if (!backend_enabled(param.backend)) {
      GTEST_SKIP() << backend_name(param.backend) << " support is not enabled.";
    }
    if (!path_exists(param.model_path)) {
      GTEST_SKIP() << "Model artifact is not available: " << param.model_path;
    }

    std::string model_path = param.model_path;
    if (param.convert_pbtxt) {
      converted_model = "deeppot_universal_" + param.name + ".pb";
      deepmd::convert_pbtxt_to_pb(param.model_path, converted_model);
      model_path = converted_model;
    }
    dp.init(model_path);
  }

  void TearDown() override {
    if (!converted_model.empty()) {
      remove(converted_model.c_str());
    }
  }
};

TEST_P(UniversalDeepPotTest, Metadata) {
  const auto& ref = *GetParam().ref;
  std::string type_map;

  EXPECT_DOUBLE_EQ(dp.cutoff(), ref.cutoff);
  EXPECT_EQ(dp.numb_types(), ref.numb_types);
  EXPECT_EQ(dp.numb_types_spin(), ref.numb_types_spin);
  EXPECT_EQ(dp.dim_fparam(), ref.dim_fparam);
  EXPECT_EQ(dp.dim_aparam(), ref.dim_aparam);
  EXPECT_EQ(dp.is_aparam_nall(), ref.aparam_nall);
  EXPECT_EQ(dp.has_default_fparam(), ref.has_default_fparam);
  dp.get_type_map(type_map);
  EXPECT_EQ(type_map, ref.type_map);
}

template <typename ValueType>
void check_compute(deepmd::DeepPot& dp,
                   const deepmd_test::DeepPotRef& ref,
                   const double tol,
                   const bool use_box = true) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<ValueType> coord(deepmd_test::deeppot_coord().begin(),
                                     deepmd_test::deeppot_coord().end());
  std::vector<ValueType> box;
  if (use_box) {
    box.assign(deepmd_test::deeppot_box().begin(),
               deepmd_test::deeppot_box().end());
  }
  const std::vector<int> atype = deepmd_test::deeppot_atype();
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  double energy = 0.0;
  std::vector<ValueType> force;
  std::vector<ValueType> virial;
  std::vector<ValueType> atomic_energy;
  std::vector<ValueType> atomic_virial;
  dp.compute(energy, force, virial, atomic_energy, atomic_virial, coord, atype,
             box);

  ASSERT_EQ(force.size(), static_cast<size_t>(natoms * 3));
  ASSERT_EQ(virial.size(), 9U);
  ASSERT_EQ(atomic_energy.size(), static_cast<size_t>(natoms));
  ASSERT_EQ(atomic_virial.size(), static_cast<size_t>(natoms * 9));

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

TEST_P(UniversalDeepPotTest, ComputeDouble) {
  check_compute<double>(dp, *GetParam().ref, GetParam().double_tol);
}

TEST_P(UniversalDeepPotTest, ComputeFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_compute<float>(dp, *GetParam().ref, GetParam().float_tol);
}

TEST_P(UniversalDeepPotTest, ComputeNoPbcDouble) {
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute<double>(dp, *GetParam().no_pbc_ref, GetParam().double_tol,
                        false);
}

TEST_P(UniversalDeepPotTest, ComputeNoPbcFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute<float>(dp, *GetParam().no_pbc_ref, GetParam().float_tol, false);
}

INSTANTIATE_TEST_SUITE_P(
    AvailableBackends,
    UniversalDeepPotTest,
    ::testing::ValuesIn(model_cases()),
    [](const ::testing::TestParamInfo<UniversalDeepPotTest::ParamType>& info) {
      return info.param.name;
    });

}  // namespace
