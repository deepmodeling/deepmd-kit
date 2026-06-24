// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <sys/stat.h>

#include <string>
#include <vector>

#include "../../tests/infer/deeppot_universal_data.h"
#include "DeepPotPTExpt.h"
#include "expected_ref.h"

namespace deepmd_test {
namespace universal {

enum class Backend { TensorFlow, PyTorch, PTExpt, JAX, Paddle };

struct ModelCase {
  std::string name;
  Backend backend;
  std::string model_path;
  bool convert_pbtxt;
  const DeepPotRef* ref;
  const DeepPotRef* no_pbc_ref;
  double double_tol;
  double float_tol;
  bool supports_float;
  bool supports_nframes;
  bool supports_lmp_nlist_mapping;
};

struct FParamAParamCase {
  std::string name;
  Backend backend;
  std::string model_path;
  bool convert_pbtxt;
  const DeepPotRef* builtin_ref;
  std::string ref_path;
  double double_tol;
  double float_tol;
  bool supports_float;
};

inline bool path_exists(const std::string& path) {
  struct stat statbuf;
  return stat(path.c_str(), &statbuf) == 0;
}

inline bool backend_enabled(Backend backend) {
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

inline std::string backend_name(Backend backend) {
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

inline std::vector<ModelCase> model_cases() {
  return {
      {"tensorflow_pb", Backend::TensorFlow, "../../tests/infer/deeppot.pbtxt",
       /*convert_pbtxt=*/true, &tf_deeppot_ref(), &tf_deeppot_no_pbc_ref(),
       1e-10, 1e-4,
       /*supports_float=*/true,
       /*supports_nframes=*/true,
       /*supports_lmp_nlist_mapping=*/true},
      {"pytorch_pth", Backend::PyTorch, "../../tests/infer/deeppot_sea.pth",
       /*convert_pbtxt=*/false, &sea_deeppot_ref(), &sea_deeppot_no_pbc_ref(),
       1e-10, 1e-4,
       /*supports_float=*/true,
       /*supports_nframes=*/false,
       /*supports_lmp_nlist_mapping=*/true},
      {"pytorch_pt2", Backend::PTExpt, "../../tests/infer/deeppot_sea.pt2",
       /*convert_pbtxt=*/false, &sea_deeppot_ref(), &sea_deeppot_no_pbc_ref(),
       1e-10, 1e-4,
       /*supports_float=*/true,
       /*supports_nframes=*/false,
       /*supports_lmp_nlist_mapping=*/true},
      {"jax_savedmodel", Backend::JAX,
       "../../tests/infer/deeppot_sea.savedmodel",
       /*convert_pbtxt=*/false, &sea_deeppot_ref(), nullptr, 1e-10, 1e-4,
       /*supports_float=*/true,
       /*supports_nframes=*/false,
       /*supports_lmp_nlist_mapping=*/true},
      {"paddle_json", Backend::Paddle, "../../tests/infer/deeppot_sea.json",
       /*convert_pbtxt=*/false, &sea_deeppot_ref(), nullptr, 1e-7, 1e-4,
       /*supports_float=*/false,
       /*supports_nframes=*/false,
       /*supports_lmp_nlist_mapping=*/false}};
}

inline std::vector<FParamAParamCase> fparam_aparam_cases() {
  return {
      {"tensorflow_pb", Backend::TensorFlow,
       "../../tests/infer/fparam_aparam.pbtxt",
       /*convert_pbtxt=*/true, &tf_fparam_aparam_ref(), "", 1e-10, 1e-4,
       /*supports_float=*/true},
      {"pytorch_pth", Backend::PyTorch, "../../tests/infer/fparam_aparam.pth",
       /*convert_pbtxt=*/false, nullptr,
       "../../tests/infer/fparam_aparam.expected", 1e-7, 1e-4,
       /*supports_float=*/true},
      {"pytorch_pt2", Backend::PTExpt, "../../tests/infer/fparam_aparam.pt2",
       /*convert_pbtxt=*/false, nullptr,
       "../../tests/infer/fparam_aparam.expected", 1e-7, 1e-4,
       /*supports_float=*/true}};
}

inline DeepPotRef load_fparam_ref(const std::string& ref_path) {
  ExpectedRef ref_file;
  ref_file.load(ref_path);
  DeepPotRef ref;
  ref.atomic_energy = ref_file.get<double>("default", "expected_e");
  ref.force = ref_file.get<double>("default", "expected_f");
  ref.atomic_virial = ref_file.get<double>("default", "expected_v");
  ref.numb_types = 1;
  ref.dim_fparam = 1;
  ref.dim_aparam = 1;
  ref.type_map = "O";
  return ref;
}

}  // namespace universal
}  // namespace deepmd_test
