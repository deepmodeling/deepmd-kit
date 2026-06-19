// SPDX-License-Identifier: LGPL-3.0-or-later
#include <gtest/gtest.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include "../../tests/infer/deeppot_universal_data.h"
#include "DeepPot.h"
#include "DeepPotPTExpt.h"
#include "expected_ref.h"
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
  bool supports_nframes;
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

struct DefaultFParamCase {
  std::string name;
  Backend backend;
  std::string model_path;
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
       &deepmd_test::tf_deeppot_no_pbc_ref(), 1e-10, 1e-4, true, true},
      {"pytorch_pth", Backend::PyTorch, "../../tests/infer/deeppot_sea.pth",
       false, &deepmd_test::sea_deeppot_ref(),
       &deepmd_test::sea_deeppot_no_pbc_ref(), 1e-10, 1e-4, true, false},
      {"pytorch_pt2", Backend::PTExpt, "../../tests/infer/deeppot_sea.pt2",
       false, &deepmd_test::sea_deeppot_ref(),
       &deepmd_test::sea_deeppot_no_pbc_ref(), 1e-10, 1e-4, true, false},
      {"jax_savedmodel", Backend::JAX,
       "../../tests/infer/deeppot_sea.savedmodel", false,
       &deepmd_test::sea_deeppot_ref(), nullptr, 1e-10, 1e-4, true, false},
      {"paddle_json", Backend::Paddle, "../../tests/infer/deeppot_sea.json",
       false, &deepmd_test::sea_deeppot_ref(), nullptr, 1e-7, 1e-4, false,
       false}};
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

std::vector<DefaultFParamCase> default_fparam_cases() {
  return {
      {"pytorch_pth", Backend::PyTorch,
       "../../tests/infer/fparam_aparam_default.pth",
       "../../tests/infer/fparam_aparam_default.expected", 1e-7, 1e-4, true},
      {"pytorch_pt2", Backend::PTExpt,
       "../../tests/infer/fparam_aparam_default.pt2",
       "../../tests/infer/fparam_aparam_default.expected", 1e-7, 1e-4, true}};
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

class UniversalDeepPotTest : public ::testing::TestWithParam<ModelCase> {
 protected:
  deepmd::DeepPot dp;
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

class FParamAParamDeepPotTest
    : public ::testing::TestWithParam<FParamAParamCase> {
 protected:
  deepmd::DeepPot dp;
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
      converted_model = "deeppot_fparam_aparam_" + param.name + ".pb";
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

class DefaultFParamDeepPotTest
    : public ::testing::TestWithParam<DefaultFParamCase> {
 protected:
  deepmd::DeepPot dp;
  deepmd_test::DeepPotRef ref;

  void SetUp() override {
    const auto& param = GetParam();
    if (!backend_enabled(param.backend)) {
      GTEST_SKIP() << backend_name(param.backend) << " support is not enabled.";
    }
    ASSERT_TRUE(path_exists(param.model_path))
        << "Model artifact is not available: " << param.model_path;
    ASSERT_TRUE(path_exists(param.ref_path))
        << "Reference artifact is not available: " << param.ref_path;
    ref = load_fparam_ref(param.ref_path);
    ref.has_default_fparam = true;
    dp.init(param.model_path);
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
void expect_reference(const double energy,
                      const std::vector<ValueType>& force,
                      const std::vector<ValueType>& virial,
                      const deepmd_test::DeepPotRef& ref,
                      const double tol) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  ASSERT_EQ(force.size(), static_cast<size_t>(natoms * 3));
  ASSERT_EQ(virial.size(), 9U);

  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (int ii = 0; ii < natoms * 3; ++ii) {
    EXPECT_NEAR(force[ii], ref.force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
}

template <typename ValueType>
void expect_atomic_reference(const double energy,
                             const std::vector<ValueType>& force,
                             const std::vector<ValueType>& virial,
                             const std::vector<ValueType>& atomic_energy,
                             const std::vector<ValueType>& atomic_virial,
                             const deepmd_test::DeepPotRef& ref,
                             const double tol) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());

  expect_reference(energy, force, virial, ref, tol);
  ASSERT_EQ(atomic_energy.size(), static_cast<size_t>(natoms));
  ASSERT_EQ(atomic_virial.size(), static_cast<size_t>(natoms * 9));

  for (int ii = 0; ii < natoms; ++ii) {
    EXPECT_NEAR(atomic_energy[ii], ref.atomic_energy[ii], tol);
  }
  for (int ii = 0; ii < natoms * 9; ++ii) {
    EXPECT_NEAR(atomic_virial[ii], ref.atomic_virial[ii], tol);
  }
}

template <typename ValueType>
void check_compute_simple(deepmd::DeepPot& dp,
                          const deepmd_test::DeepPotRef& ref,
                          const double tol,
                          const bool use_box = true) {
  const std::vector<ValueType> coord(deepmd_test::deeppot_coord().begin(),
                                     deepmd_test::deeppot_coord().end());
  std::vector<ValueType> box;
  if (use_box) {
    box.assign(deepmd_test::deeppot_box().begin(),
               deepmd_test::deeppot_box().end());
  }
  const std::vector<int> atype = deepmd_test::deeppot_atype();

  double energy = 0.0;
  std::vector<ValueType> force;
  std::vector<ValueType> virial;
  dp.compute(energy, force, virial, coord, atype, box);

  expect_reference(energy, force, virial, ref, tol);
}

template <typename ValueType>
void check_compute_atomic(deepmd::DeepPot& dp,
                          const deepmd_test::DeepPotRef& ref,
                          const double tol,
                          const bool use_box = true) {
  const std::vector<ValueType> coord(deepmd_test::deeppot_coord().begin(),
                                     deepmd_test::deeppot_coord().end());
  std::vector<ValueType> box;
  if (use_box) {
    box.assign(deepmd_test::deeppot_box().begin(),
               deepmd_test::deeppot_box().end());
  }
  const std::vector<int> atype = deepmd_test::deeppot_atype();

  double energy = 0.0;
  std::vector<ValueType> force;
  std::vector<ValueType> virial;
  std::vector<ValueType> atomic_energy;
  std::vector<ValueType> atomic_virial;
  dp.compute(energy, force, virial, atomic_energy, atomic_virial, coord, atype,
             box);

  expect_atomic_reference(energy, force, virial, atomic_energy, atomic_virial,
                          ref, tol);
}

template <typename ValueType>
void check_finite_difference(deepmd::DeepPot& dp) {
  class Model : public EnergyModelTest<ValueType> {
    deepmd::DeepPot& dp_;
    const std::vector<int> atype_;

   public:
    Model(deepmd::DeepPot& dp, const std::vector<int>& atype)
        : dp_(dp), atype_(atype) {}

    void compute(double& energy,
                 std::vector<ValueType>& force,
                 std::vector<ValueType>& virial,
                 const std::vector<ValueType>& coord,
                 const std::vector<ValueType>& box) override {
      dp_.compute(energy, force, virial, coord, atype_, box);
    }
  };

  const std::vector<ValueType> coord(deepmd_test::deeppot_coord().begin(),
                                     deepmd_test::deeppot_coord().end());
  const std::vector<int> atype = deepmd_test::deeppot_atype();
  std::vector<ValueType> box(deepmd_test::deeppot_box().begin(),
                             deepmd_test::deeppot_box().end());
  Model model(dp, atype);
  model.test_f(coord, box);
  model.test_v(coord, box);
  box[1] -= 0.4;
  model.test_f(coord, box);
  model.test_v(coord, box);
  box[2] += 0.5;
  model.test_f(coord, box);
  model.test_v(coord, box);
  box[4] += 0.2;
  model.test_f(coord, box);
  model.test_v(coord, box);
  box[3] -= 0.3;
  model.test_f(coord, box);
  model.test_v(coord, box);
  box[6] -= 0.7;
  model.test_f(coord, box);
  model.test_v(coord, box);
  box[7] += 0.6;
  model.test_f(coord, box);
  model.test_v(coord, box);
}

template <typename ValueType>
void check_lmp_nlist(deepmd::DeepPot& dp,
                     const deepmd_test::DeepPotRef& ref,
                     const double tol,
                     const double cutoff_scale = 1.0) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<ValueType> coord(deepmd_test::deeppot_coord().begin(),
                                     deepmd_test::deeppot_coord().end());
  const std::vector<int> atype = deepmd_test::deeppot_atype();
  const std::vector<ValueType> box(deepmd_test::deeppot_box().begin(),
                                   deepmd_test::deeppot_box().end());
  const float rc = static_cast<float>(dp.cutoff() * cutoff_scale);
  const int nloc = natoms;
  std::vector<ValueType> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<ValueType>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  const int nall = static_cast<int>(coord_cpy.size() / 3);
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);

  for (int ago = 0; ago < 2; ++ago) {
    double energy = 0.0;
    std::vector<ValueType> force_all(nall * 3, 0.0);
    std::vector<ValueType> virial(9, 0.0);
    dp.compute(energy, force_all, virial, coord_cpy, atype_cpy, box,
               nall - nloc, inlist, ago);

    std::vector<ValueType> force;
    _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3);
    expect_reference(energy, force, virial, ref, tol);
  }
}

template <typename ValueType>
void check_lmp_nlist_atomic(deepmd::DeepPot& dp,
                            const deepmd_test::DeepPotRef& ref,
                            const double tol) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<ValueType> coord(deepmd_test::deeppot_coord().begin(),
                                     deepmd_test::deeppot_coord().end());
  const std::vector<int> atype = deepmd_test::deeppot_atype();
  const std::vector<ValueType> box(deepmd_test::deeppot_box().begin(),
                                   deepmd_test::deeppot_box().end());
  const float rc = static_cast<float>(dp.cutoff());
  const int nloc = natoms;
  std::vector<ValueType> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<ValueType>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  const int nall = static_cast<int>(coord_cpy.size() / 3);
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);

  for (int ago = 0; ago < 2; ++ago) {
    double energy = 0.0;
    std::vector<ValueType> force_all(nall * 3, 0.0), virial(9, 0.0);
    std::vector<ValueType> atomic_energy_all(nall, 0.0);
    std::vector<ValueType> atomic_virial_all(nall * 9, 0.0);
    dp.compute(energy, force_all, virial, atomic_energy_all, atomic_virial_all,
               coord_cpy, atype_cpy, box, nall - nloc, inlist, ago);

    std::vector<ValueType> force, atomic_energy, atomic_virial;
    _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3);
    _fold_back<ValueType>(atomic_energy, atomic_energy_all, mapping, nloc, nall,
                          1);
    _fold_back<ValueType>(atomic_virial, atomic_virial_all, mapping, nloc, nall,
                          9);
    expect_atomic_reference(energy, force, virial, atomic_energy, atomic_virial,
                            ref, tol);
  }
}

template <typename ValueType>
void check_lmp_nlist_type_sel(deepmd::DeepPot& dp,
                              const deepmd_test::DeepPotRef& ref,
                              const double tol,
                              const bool atomic) {
  constexpr int nvir = 2;
  std::vector<ValueType> coord(deepmd_test::deeppot_coord().begin(),
                               deepmd_test::deeppot_coord().end());
  std::vector<int> atype = deepmd_test::deeppot_atype();
  const std::vector<ValueType> box(deepmd_test::deeppot_box().begin(),
                                   deepmd_test::deeppot_box().end());
  std::vector<ValueType> expected_force(ref.force.begin(), ref.force.end());
  std::vector<ValueType> coord_vir(nvir * 3);
  std::copy(coord.begin(), coord.begin() + nvir * 3, coord_vir.begin());
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  atype.insert(atype.begin(), nvir, 2);
  expected_force.insert(expected_force.begin(), nvir * 3, 0.0);

  const int natoms = static_cast<int>(atype.size());
  const int nloc = natoms;
  const float rc = static_cast<float>(dp.cutoff());
  std::vector<ValueType> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<ValueType>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  const int nall = static_cast<int>(coord_cpy.size() / 3);
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);

  double energy = 0.0;
  std::vector<ValueType> force_all(nall * 3, 0.0), virial(9, 0.0);
  if (atomic) {
    std::vector<ValueType> atomic_energy, atomic_virial;
    dp.compute(energy, force_all, virial, atomic_energy, atomic_virial,
               coord_cpy, atype_cpy, box, nall - nloc, inlist, 0);
  } else {
    dp.compute(energy, force_all, virial, coord_cpy, atype_cpy, box,
               nall - nloc, inlist, 0);
  }

  std::vector<ValueType> force;
  _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3);
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);
  ASSERT_EQ(force.size(), expected_force.size());
  EXPECT_NEAR(energy, deepmd_test::total_energy(ref), tol);
  for (size_t ii = 0; ii < expected_force.size(); ++ii) {
    EXPECT_NEAR(force[ii], expected_force[ii], tol);
  }
  for (size_t ii = 0; ii < 9; ++ii) {
    EXPECT_NEAR(virial[ii], expected_virial[ii], tol);
  }
}

template <typename ValueType>
std::vector<ValueType> cast_values(const std::vector<double>& values) {
  return std::vector<ValueType>(values.begin(), values.end());
}

template <typename ValueType>
std::vector<ValueType> repeat_values(const std::vector<ValueType>& values,
                                     const int nframes) {
  std::vector<ValueType> repeated;
  repeated.reserve(static_cast<size_t>(nframes) * values.size());
  for (int kk = 0; kk < nframes; ++kk) {
    repeated.insert(repeated.end(), values.begin(), values.end());
  }
  return repeated;
}

template <typename ValueType>
void expect_reference_frames(const std::vector<double>& energy,
                             const std::vector<ValueType>& force,
                             const std::vector<ValueType>& virial,
                             const deepmd_test::DeepPotRef& ref,
                             const double tol,
                             const int nframes) {
  const int natoms = static_cast<int>(ref.atomic_energy.size());
  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);

  ASSERT_EQ(energy.size(), static_cast<size_t>(nframes));
  ASSERT_EQ(force.size(), static_cast<size_t>(nframes * natoms * 3));
  ASSERT_EQ(virial.size(), static_cast<size_t>(nframes * 9));

  for (int kk = 0; kk < nframes; ++kk) {
    EXPECT_NEAR(energy[kk], deepmd_test::total_energy(ref), tol);
    for (int ii = 0; ii < natoms * 3; ++ii) {
      EXPECT_NEAR(force[kk * natoms * 3 + ii], ref.force[ii], tol);
    }
    for (int ii = 0; ii < 9; ++ii) {
      EXPECT_NEAR(virial[kk * 9 + ii], expected_virial[ii], tol);
    }
  }
}

template <typename ValueType>
void expect_atomic_reference_frames(const std::vector<double>& energy,
                                    const std::vector<ValueType>& force,
                                    const std::vector<ValueType>& virial,
                                    const std::vector<ValueType>& atomic_energy,
                                    const std::vector<ValueType>& atomic_virial,
                                    const deepmd_test::DeepPotRef& ref,
                                    const double tol,
                                    const int nframes) {
  const int natoms = static_cast<int>(ref.atomic_energy.size());

  expect_reference_frames(energy, force, virial, ref, tol, nframes);
  ASSERT_EQ(atomic_energy.size(), static_cast<size_t>(nframes * natoms));
  ASSERT_EQ(atomic_virial.size(), static_cast<size_t>(nframes * natoms * 9));

  for (int kk = 0; kk < nframes; ++kk) {
    for (int ii = 0; ii < natoms; ++ii) {
      EXPECT_NEAR(atomic_energy[kk * natoms + ii], ref.atomic_energy[ii], tol);
    }
    for (int ii = 0; ii < natoms * 9; ++ii) {
      EXPECT_NEAR(atomic_virial[kk * natoms * 9 + ii], ref.atomic_virial[ii],
                  tol);
    }
  }
}

template <typename ValueType>
void check_compute_frames_simple(deepmd::DeepPot& dp,
                                 const deepmd_test::DeepPotRef& ref,
                                 const double tol,
                                 const int nframes,
                                 const bool use_box = true) {
  const std::vector<ValueType> coord = repeat_values(
      cast_values<ValueType>(deepmd_test::deeppot_coord()), nframes);
  std::vector<ValueType> box;
  if (use_box) {
    box = repeat_values(cast_values<ValueType>(deepmd_test::deeppot_box()),
                        nframes);
  }
  const std::vector<int> atype = deepmd_test::deeppot_atype();

  std::vector<double> energy;
  std::vector<ValueType> force;
  std::vector<ValueType> virial;
  dp.compute(energy, force, virial, coord, atype, box);

  expect_reference_frames(energy, force, virial, ref, tol, nframes);
}

template <typename ValueType>
void check_compute_frames_atomic(deepmd::DeepPot& dp,
                                 const deepmd_test::DeepPotRef& ref,
                                 const double tol,
                                 const int nframes) {
  const std::vector<ValueType> coord = repeat_values(
      cast_values<ValueType>(deepmd_test::deeppot_coord()), nframes);
  const std::vector<ValueType> box = repeat_values(
      cast_values<ValueType>(deepmd_test::deeppot_box()), nframes);
  const std::vector<int> atype = deepmd_test::deeppot_atype();

  std::vector<double> energy;
  std::vector<ValueType> force;
  std::vector<ValueType> virial;
  std::vector<ValueType> atomic_energy;
  std::vector<ValueType> atomic_virial;
  dp.compute(energy, force, virial, atomic_energy, atomic_virial, coord, atype,
             box);

  expect_atomic_reference_frames(energy, force, virial, atomic_energy,
                                 atomic_virial, ref, tol, nframes);
}

template <typename ValueType>
void check_lmp_nlist_frames(deepmd::DeepPot& dp,
                            const deepmd_test::DeepPotRef& ref,
                            const double tol,
                            const int nframes,
                            const bool atomic,
                            const double cutoff_scale = 1.0) {
  const int natoms = static_cast<int>(deepmd_test::deeppot_atype().size());
  const std::vector<ValueType> coord =
      cast_values<ValueType>(deepmd_test::deeppot_coord());
  const std::vector<int> atype = deepmd_test::deeppot_atype();
  const std::vector<ValueType> box_frame =
      cast_values<ValueType>(deepmd_test::deeppot_box());
  const std::vector<ValueType> box = repeat_values(box_frame, nframes);
  const float rc = static_cast<float>(dp.cutoff() * cutoff_scale);
  const int nloc = natoms;
  std::vector<ValueType> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<ValueType>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box_frame, rc);
  const int nall = static_cast<int>(coord_cpy.size() / 3);
  const std::vector<ValueType> coord_cpy_frames =
      repeat_values(coord_cpy, nframes);
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);

  for (int ago = 0; ago < 2; ++ago) {
    std::vector<double> energy;
    std::vector<ValueType> force_all;
    std::vector<ValueType> virial;
    if (atomic) {
      std::vector<ValueType> atomic_energy_all;
      std::vector<ValueType> atomic_virial_all;
      dp.compute(energy, force_all, virial, atomic_energy_all,
                 atomic_virial_all, coord_cpy_frames, atype_cpy, box,
                 nall - nloc, inlist, ago);

      std::vector<ValueType> force, atomic_energy, atomic_virial;
      _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3, nframes);
      _fold_back<ValueType>(atomic_energy, atomic_energy_all, mapping, nloc,
                            nall, 1, nframes);
      _fold_back<ValueType>(atomic_virial, atomic_virial_all, mapping, nloc,
                            nall, 9, nframes);
      expect_atomic_reference_frames(energy, force, virial, atomic_energy,
                                     atomic_virial, ref, tol, nframes);
    } else {
      dp.compute(energy, force_all, virial, coord_cpy_frames, atype_cpy, box,
                 nall - nloc, inlist, ago);

      std::vector<ValueType> force;
      _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3, nframes);
      expect_reference_frames(energy, force, virial, ref, tol, nframes);
    }
  }
}

template <typename ValueType>
void check_lmp_nlist_type_sel_frames(deepmd::DeepPot& dp,
                                     const deepmd_test::DeepPotRef& ref,
                                     const double tol,
                                     const int nframes,
                                     const bool atomic) {
  constexpr int nvir = 2;
  const std::vector<ValueType> base_coord =
      cast_values<ValueType>(deepmd_test::deeppot_coord());
  std::vector<ValueType> coord_vir(nvir * 3);
  std::copy(base_coord.begin(), base_coord.begin() + nvir * 3,
            coord_vir.begin());

  std::vector<ValueType> coord = base_coord;
  coord.insert(coord.begin(), coord_vir.begin(), coord_vir.end());
  std::vector<int> atype = deepmd_test::deeppot_atype();
  atype.insert(atype.begin(), nvir, 2);
  const std::vector<ValueType> box_frame =
      cast_values<ValueType>(deepmd_test::deeppot_box());
  const std::vector<ValueType> box = repeat_values(box_frame, nframes);

  const int natoms = static_cast<int>(atype.size());
  const int nloc = natoms;
  const float rc = static_cast<float>(dp.cutoff());
  std::vector<ValueType> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<ValueType>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box_frame, rc);
  const int nall = static_cast<int>(coord_cpy.size() / 3);
  const std::vector<ValueType> coord_cpy_frames =
      repeat_values(coord_cpy, nframes);
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);

  std::vector<double> energy;
  std::vector<ValueType> force_all;
  std::vector<ValueType> virial;
  if (atomic) {
    std::vector<ValueType> atomic_energy, atomic_virial;
    dp.compute(energy, force_all, virial, atomic_energy, atomic_virial,
               coord_cpy_frames, atype_cpy, box, nall - nloc, inlist, 0);
  } else {
    dp.compute(energy, force_all, virial, coord_cpy_frames, atype_cpy, box,
               nall - nloc, inlist, 0);
  }

  std::vector<ValueType> force;
  _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3, nframes);

  const std::vector<double> expected_virial = deepmd_test::total_virial(ref);
  std::vector<ValueType> expected_force_frame(ref.force.begin(),
                                              ref.force.end());
  expected_force_frame.insert(expected_force_frame.begin(), nvir * 3, 0.0);
  const std::vector<ValueType> expected_force =
      repeat_values(expected_force_frame, nframes);

  ASSERT_EQ(energy.size(), static_cast<size_t>(nframes));
  ASSERT_EQ(force.size(), expected_force.size());
  ASSERT_EQ(virial.size(), static_cast<size_t>(nframes * 9));
  for (int kk = 0; kk < nframes; ++kk) {
    EXPECT_NEAR(energy[kk], deepmd_test::total_energy(ref), tol);
    for (size_t ii = 0; ii < expected_force_frame.size(); ++ii) {
      EXPECT_NEAR(force[kk * expected_force_frame.size() + ii],
                  expected_force_frame[ii], tol);
    }
    for (size_t ii = 0; ii < 9; ++ii) {
      EXPECT_NEAR(virial[kk * 9 + ii], expected_virial[ii], tol);
    }
  }
}

template <typename ValueType>
void check_fparam_compute_simple(deepmd::DeepPot& dp,
                                 const deepmd_test::DeepPotRef& ref,
                                 const double tol,
                                 const std::vector<double>& fparam_values) {
  const std::vector<ValueType> coord =
      cast_values<ValueType>(deepmd_test::deeppot_coord());
  const std::vector<int> atype = deepmd_test::fparam_aparam_atype();
  const std::vector<ValueType> box =
      cast_values<ValueType>(deepmd_test::deeppot_box());
  const std::vector<ValueType> fparam = cast_values<ValueType>(fparam_values);
  const std::vector<ValueType> aparam =
      cast_values<ValueType>(deepmd_test::aparam_value());

  double energy = 0.0;
  std::vector<ValueType> force;
  std::vector<ValueType> virial;
  dp.compute(energy, force, virial, coord, atype, box, fparam, aparam);

  expect_reference(energy, force, virial, ref, tol);
}

template <typename ValueType>
void check_fparam_compute_atomic(deepmd::DeepPot& dp,
                                 const deepmd_test::DeepPotRef& ref,
                                 const double tol,
                                 const std::vector<double>& fparam_values) {
  const std::vector<ValueType> coord =
      cast_values<ValueType>(deepmd_test::deeppot_coord());
  const std::vector<int> atype = deepmd_test::fparam_aparam_atype();
  const std::vector<ValueType> box =
      cast_values<ValueType>(deepmd_test::deeppot_box());
  const std::vector<ValueType> fparam = cast_values<ValueType>(fparam_values);
  const std::vector<ValueType> aparam =
      cast_values<ValueType>(deepmd_test::aparam_value());

  double energy = 0.0;
  std::vector<ValueType> force;
  std::vector<ValueType> virial;
  std::vector<ValueType> atomic_energy;
  std::vector<ValueType> atomic_virial;
  dp.compute(energy, force, virial, atomic_energy, atomic_virial, coord, atype,
             box, fparam, aparam);

  expect_atomic_reference(energy, force, virial, atomic_energy, atomic_virial,
                          ref, tol);
}

template <typename ValueType>
void check_fparam_lmp_nlist(deepmd::DeepPot& dp,
                            const deepmd_test::DeepPotRef& ref,
                            const double tol,
                            const std::vector<double>& fparam_values,
                            const bool atomic,
                            const double cutoff_scale = 1.0,
                            const int ago_count = 2) {
  const int natoms =
      static_cast<int>(deepmd_test::fparam_aparam_atype().size());
  const std::vector<ValueType> coord =
      cast_values<ValueType>(deepmd_test::deeppot_coord());
  const std::vector<int> atype = deepmd_test::fparam_aparam_atype();
  const std::vector<ValueType> box =
      cast_values<ValueType>(deepmd_test::deeppot_box());
  const std::vector<ValueType> fparam = cast_values<ValueType>(fparam_values);
  const std::vector<ValueType> aparam =
      cast_values<ValueType>(deepmd_test::aparam_value());
  const float rc = static_cast<float>(dp.cutoff() * cutoff_scale);
  const int nloc = natoms;
  std::vector<ValueType> coord_cpy;
  std::vector<int> atype_cpy, mapping;
  std::vector<std::vector<int>> nlist_data;
  _build_nlist<ValueType>(nlist_data, coord_cpy, atype_cpy, mapping, coord,
                          atype, box, rc);
  const int nall = static_cast<int>(coord_cpy.size() / 3);
  std::vector<int> ilist(nloc), numneigh(nloc);
  std::vector<int*> firstneigh(nloc);
  deepmd::InputNlist inlist(nloc, ilist.data(), numneigh.data(),
                            firstneigh.data());
  convert_nlist(inlist, nlist_data);

  for (int ago = 0; ago < ago_count; ++ago) {
    double energy = 0.0;
    std::vector<ValueType> force_all(nall * 3, 0.0);
    std::vector<ValueType> virial(9, 0.0);
    if (atomic) {
      std::vector<ValueType> atomic_energy_all(nall, 0.0);
      std::vector<ValueType> atomic_virial_all(nall * 9, 0.0);
      dp.compute(energy, force_all, virial, atomic_energy_all,
                 atomic_virial_all, coord_cpy, atype_cpy, box, nall - nloc,
                 inlist, ago, fparam, aparam);

      std::vector<ValueType> force, atomic_energy, atomic_virial;
      _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3);
      _fold_back<ValueType>(atomic_energy, atomic_energy_all, mapping, nloc,
                            nall, 1);
      _fold_back<ValueType>(atomic_virial, atomic_virial_all, mapping, nloc,
                            nall, 9);
      expect_atomic_reference(energy, force, virial, atomic_energy,
                              atomic_virial, ref, tol);
    } else {
      dp.compute(energy, force_all, virial, coord_cpy, atype_cpy, box,
                 nall - nloc, inlist, ago, fparam, aparam);

      std::vector<ValueType> force;
      _fold_back<ValueType>(force, force_all, mapping, nloc, nall, 3);
      expect_reference(energy, force, virial, ref, tol);
    }
  }
}

TEST_P(UniversalDeepPotTest, ComputeDouble) {
  check_compute_atomic<double>(dp, *GetParam().ref, GetParam().double_tol);
}

TEST_P(UniversalDeepPotTest, ComputeFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_compute_atomic<float>(dp, *GetParam().ref, GetParam().float_tol);
}

TEST_P(UniversalDeepPotTest, ComputeSimpleDouble) {
  check_compute_simple<double>(dp, *GetParam().ref, GetParam().double_tol);
}

TEST_P(UniversalDeepPotTest, ComputeSimpleFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_compute_simple<float>(dp, *GetParam().ref, GetParam().float_tol);
}

TEST_P(UniversalDeepPotTest, FiniteDifferenceDouble) {
  check_finite_difference<double>(dp);
}

TEST_P(UniversalDeepPotTest, FiniteDifferenceFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_finite_difference<float>(dp);
}

TEST_P(UniversalDeepPotTest, LmpNlistDouble) {
  check_lmp_nlist<double>(dp, *GetParam().ref, GetParam().double_tol);
}

TEST_P(UniversalDeepPotTest, LmpNlistFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_lmp_nlist<float>(dp, *GetParam().ref, GetParam().float_tol);
}

TEST_P(UniversalDeepPotTest, LmpNlistAtomicDouble) {
  check_lmp_nlist_atomic<double>(dp, *GetParam().ref, GetParam().double_tol);
}

TEST_P(UniversalDeepPotTest, LmpNlistAtomicFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_lmp_nlist_atomic<float>(dp, *GetParam().ref, GetParam().float_tol);
}

TEST_P(UniversalDeepPotTest, LmpNlistDoubleCutoffTwice) {
  check_lmp_nlist<double>(dp, *GetParam().ref, GetParam().double_tol, 2.0);
}

TEST_P(UniversalDeepPotTest, LmpNlistFloatCutoffTwice) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_lmp_nlist<float>(dp, *GetParam().ref, GetParam().float_tol, 2.0);
}

TEST_P(UniversalDeepPotTest, LmpNlistTypeSelDouble) {
  check_lmp_nlist_type_sel<double>(dp, *GetParam().ref, GetParam().double_tol,
                                   false);
}

TEST_P(UniversalDeepPotTest, LmpNlistTypeSelFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_lmp_nlist_type_sel<float>(dp, *GetParam().ref, GetParam().float_tol,
                                  false);
}

TEST_P(UniversalDeepPotTest, LmpNlistTypeSelAtomicDouble) {
  check_lmp_nlist_type_sel<double>(dp, *GetParam().ref, GetParam().double_tol,
                                   true);
}

TEST_P(UniversalDeepPotTest, LmpNlistTypeSelAtomicFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_lmp_nlist_type_sel<float>(dp, *GetParam().ref, GetParam().float_tol,
                                  true);
}

TEST_P(UniversalDeepPotTest, ComputeNoPbcDouble) {
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute_atomic<double>(dp, *GetParam().no_pbc_ref,
                               GetParam().double_tol, false);
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
  check_compute_atomic<float>(dp, *GetParam().no_pbc_ref, GetParam().float_tol,
                              false);
}

TEST_P(UniversalDeepPotTest, ComputeSimpleNoPbcDouble) {
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute_simple<double>(dp, *GetParam().no_pbc_ref,
                               GetParam().double_tol, false);
}

TEST_P(UniversalDeepPotTest, ComputeSimpleNoPbcFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute_simple<float>(dp, *GetParam().no_pbc_ref, GetParam().float_tol,
                              false);
}

TEST_P(UniversalDeepPotTest, PrintSummary) { dp.print_summary(""); }

TEST_P(UniversalDeepPotTest, NFramesComputeDouble) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_compute_frames_simple<double>(dp, *GetParam().ref,
                                      GetParam().double_tol, 2);
}

TEST_P(UniversalDeepPotTest, NFramesComputeFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_compute_frames_simple<float>(dp, *GetParam().ref, GetParam().float_tol,
                                     2);
}

TEST_P(UniversalDeepPotTest, NFramesComputeAtomicDouble) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_compute_frames_atomic<double>(dp, *GetParam().ref,
                                      GetParam().double_tol, 2);
}

TEST_P(UniversalDeepPotTest, NFramesComputeAtomicFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_compute_frames_atomic<float>(dp, *GetParam().ref, GetParam().float_tol,
                                     2);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistDouble) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_frames<double>(dp, *GetParam().ref, GetParam().double_tol, 2,
                                 false);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_frames<float>(dp, *GetParam().ref, GetParam().float_tol, 2,
                                false);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistAtomicDouble) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_frames<double>(dp, *GetParam().ref, GetParam().double_tol, 2,
                                 true);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistAtomicFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_frames<float>(dp, *GetParam().ref, GetParam().float_tol, 2,
                                true);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistDoubleCutoffTwice) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_frames<double>(dp, *GetParam().ref, GetParam().double_tol, 2,
                                 false, 2.0);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistFloatCutoffTwice) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_frames<float>(dp, *GetParam().ref, GetParam().float_tol, 2,
                                false, 2.0);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistTypeSelDouble) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_type_sel_frames<double>(dp, *GetParam().ref,
                                          GetParam().double_tol, 2, false);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistTypeSelFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_type_sel_frames<float>(dp, *GetParam().ref,
                                         GetParam().float_tol, 2, false);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistTypeSelAtomicDouble) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_type_sel_frames<double>(dp, *GetParam().ref,
                                          GetParam().double_tol, 2, true);
}

TEST_P(UniversalDeepPotTest, NFramesLmpNlistTypeSelAtomicFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  check_lmp_nlist_type_sel_frames<float>(dp, *GetParam().ref,
                                         GetParam().float_tol, 2, true);
}

TEST_P(UniversalDeepPotTest, NFramesComputeNoPbcDouble) {
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute_frames_simple<double>(dp, *GetParam().no_pbc_ref,
                                      GetParam().double_tol, 2, false);
}

TEST_P(UniversalDeepPotTest, NFramesComputeNoPbcFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  if (!GetParam().supports_nframes) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " nframes coverage is not enabled.";
  }
  if (GetParam().no_pbc_ref == nullptr) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " NoPBC reference is not available.";
  }
  check_compute_frames_simple<float>(dp, *GetParam().no_pbc_ref,
                                     GetParam().float_tol, 2, false);
}

TEST_P(FParamAParamDeepPotTest, Metadata) {
  EXPECT_EQ(dp.dim_fparam(), 1);
  EXPECT_EQ(dp.dim_aparam(), 1);
  EXPECT_FALSE(dp.has_default_fparam());
}

TEST_P(FParamAParamDeepPotTest, ComputeDouble) {
  check_fparam_compute_atomic<double>(dp, *ref, GetParam().double_tol,
                                      deepmd_test::fparam_value());
}

TEST_P(FParamAParamDeepPotTest, ComputeFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_compute_atomic<float>(dp, *ref, GetParam().float_tol,
                                     deepmd_test::fparam_value());
}

TEST_P(FParamAParamDeepPotTest, ComputeSimpleDouble) {
  check_fparam_compute_simple<double>(dp, *ref, GetParam().double_tol,
                                      deepmd_test::fparam_value());
}

TEST_P(FParamAParamDeepPotTest, ComputeSimpleFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_compute_simple<float>(dp, *ref, GetParam().float_tol,
                                     deepmd_test::fparam_value());
}

TEST_P(FParamAParamDeepPotTest, LmpNlistDouble) {
  check_fparam_lmp_nlist<double>(dp, *ref, GetParam().double_tol,
                                 deepmd_test::fparam_value(), false);
}

TEST_P(FParamAParamDeepPotTest, LmpNlistFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_lmp_nlist<float>(dp, *ref, GetParam().float_tol,
                                deepmd_test::fparam_value(), false);
}

TEST_P(FParamAParamDeepPotTest, LmpNlistAtomicDouble) {
  check_fparam_lmp_nlist<double>(dp, *ref, GetParam().double_tol,
                                 deepmd_test::fparam_value(), true);
}

TEST_P(FParamAParamDeepPotTest, LmpNlistAtomicFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_lmp_nlist<float>(dp, *ref, GetParam().float_tol,
                                deepmd_test::fparam_value(), true);
}

TEST_P(FParamAParamDeepPotTest, LmpNlistDoubleCutoffTwice) {
  check_fparam_lmp_nlist<double>(dp, *ref, GetParam().double_tol,
                                 deepmd_test::fparam_value(), false, 2.0);
}

TEST_P(FParamAParamDeepPotTest, LmpNlistFloatCutoffTwice) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_lmp_nlist<float>(dp, *ref, GetParam().float_tol,
                                deepmd_test::fparam_value(), false, 2.0);
}

TEST_P(DefaultFParamDeepPotTest, Metadata) {
  EXPECT_EQ(dp.dim_fparam(), 1);
  EXPECT_EQ(dp.dim_aparam(), 1);
  EXPECT_TRUE(dp.has_default_fparam());
}

TEST_P(DefaultFParamDeepPotTest, ComputeWithEmptyFParamDouble) {
  check_fparam_compute_simple<double>(dp, ref, GetParam().double_tol, {});
}

TEST_P(DefaultFParamDeepPotTest, ComputeWithEmptyFParamFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_compute_simple<float>(dp, ref, GetParam().float_tol, {});
}

TEST_P(DefaultFParamDeepPotTest, ComputeWithExplicitFParamDouble) {
  check_fparam_compute_simple<double>(dp, ref, GetParam().double_tol,
                                      deepmd_test::fparam_value());
}

TEST_P(DefaultFParamDeepPotTest, ComputeWithExplicitFParamFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_compute_simple<float>(dp, ref, GetParam().float_tol,
                                     deepmd_test::fparam_value());
}

TEST_P(DefaultFParamDeepPotTest, LmpNlistWithEmptyFParamDouble) {
  check_fparam_lmp_nlist<double>(dp, ref, GetParam().double_tol, {}, false, 1.0,
                                 1);
}

TEST_P(DefaultFParamDeepPotTest, LmpNlistWithEmptyFParamFloat) {
  if (!GetParam().supports_float) {
    GTEST_SKIP() << backend_name(GetParam().backend)
                 << " does not provide float inference coverage.";
  }
  check_fparam_lmp_nlist<float>(dp, ref, GetParam().float_tol, {}, false, 1.0,
                                1);
}

INSTANTIATE_TEST_SUITE_P(
    AvailableBackends,
    UniversalDeepPotTest,
    ::testing::ValuesIn(model_cases()),
    [](const ::testing::TestParamInfo<UniversalDeepPotTest::ParamType>& info) {
      return info.param.name;
    });

INSTANTIATE_TEST_SUITE_P(
    FParamAParamBackends,
    FParamAParamDeepPotTest,
    ::testing::ValuesIn(fparam_aparam_cases()),
    [](const ::testing::TestParamInfo<FParamAParamDeepPotTest::ParamType>&
           info) { return info.param.name; });

INSTANTIATE_TEST_SUITE_P(
    DefaultFParamBackends,
    DefaultFParamDeepPotTest,
    ::testing::ValuesIn(default_fparam_cases()),
    [](const ::testing::TestParamInfo<DefaultFParamDeepPotTest::ParamType>&
           info) { return info.param.name; });

}  // namespace
