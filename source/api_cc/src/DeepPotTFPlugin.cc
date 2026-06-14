// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_TENSORFLOW

#include <fstream>
#include <sstream>

#include "BackendPluginFactory.h"
#include "DataModifierTF.h"
#include "DeepPotTF.h"
#include "DeepSpinTF.h"
#include "DeepTensorTF.h"
#include "commonTF.h"
#include "google/protobuf/text_format.h"

namespace {

void convert_pbtxt_to_pb_impl(const char* pbtxt, const char* pb) {
  if (pbtxt == nullptr || pb == nullptr) {
    throw deepmd::deepmd_exception("pbtxt and pb paths must not be null");
  }

  std::ifstream input(pbtxt);
  if (!input.is_open()) {
    throw deepmd::deepmd_exception(std::string("Failed to open file: ") +
                                   pbtxt);
  }

  std::stringstream buffer;
  buffer << input.rdbuf();

  tensorflow::GraphDef graph_def;
  if (!tensorflow::protobuf::TextFormat::ParseFromString(buffer.str(),
                                                         &graph_def)) {
    throw deepmd::deepmd_exception(std::string("Failed to parse pbtxt: ") +
                                   pbtxt);
  }

  std::ofstream output(pb, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!output.is_open()) {
    throw deepmd::deepmd_exception(std::string("Failed to open file: ") + pb);
  }
  if (!graph_def.SerializeToOstream(&output)) {
    throw deepmd::deepmd_exception(std::string("Failed to write pb: ") + pb);
  }
}

}  // namespace

extern "C" void* deepmd_create_deeppot_backend_v1(const char* model,
                                                  int gpu_rank,
                                                  const char* file_content,
                                                  std::size_t file_content_size,
                                                  char** error_message) {
  return deepmd::plugin::create_deeppot_backend<deepmd::DeepPotTF>(
      model, gpu_rank, file_content, file_content_size, error_message);
}

extern "C" void deepmd_delete_deeppot_backend_v1(void* backend) {
  deepmd::plugin::delete_deeppot_backend(backend);
}

extern "C" void* deepmd_create_deepspin_backend_v1(
    const char* model,
    int gpu_rank,
    const char* file_content,
    std::size_t file_content_size,
    char** error_message) {
  return deepmd::plugin::create_deepspin_backend<deepmd::DeepSpinTF>(
      model, gpu_rank, file_content, file_content_size, error_message);
}

extern "C" void deepmd_delete_deepspin_backend_v1(void* backend) {
  deepmd::plugin::delete_deepspin_backend(backend);
}

extern "C" void* deepmd_create_deeptensor_backend_v1(const char* model,
                                                     int gpu_rank,
                                                     const char* name_scope,
                                                     char** error_message) {
  return deepmd::plugin::create_deeptensor_backend<deepmd::DeepTensorTF>(
      model, gpu_rank, name_scope, error_message);
}

extern "C" void deepmd_delete_deeptensor_backend_v1(void* backend) {
  deepmd::plugin::delete_deeptensor_backend(backend);
}

extern "C" void* deepmd_create_dipole_charge_modifier_backend_v1(
    const char* model,
    int gpu_rank,
    const char* name_scope,
    char** error_message) {
  return deepmd::plugin::create_dipole_charge_modifier_backend<
      deepmd::DipoleChargeModifierTF>(model, gpu_rank, name_scope,
                                      error_message);
}

extern "C" void deepmd_delete_dipole_charge_modifier_backend_v1(void* backend) {
  deepmd::plugin::delete_dipole_charge_modifier_backend(backend);
}

extern "C" int deepmd_convert_pbtxt_to_pb_v1(const char* pbtxt,
                                             const char* pb,
                                             char** error_message) {
  try {
    convert_pbtxt_to_pb_impl(pbtxt, pb);
    return 0;
  } catch (const deepmd::deepmd_exception& e) {
    deepmd::plugin::set_error_message(error_message, e.what());
  } catch (const std::exception& e) {
    deepmd::plugin::set_error_message(error_message, e.what());
  } catch (...) {
    deepmd::plugin::set_error_message(error_message,
                                      "unknown backend plugin error");
  }
  return 1;
}

extern "C" void deepmd_free_backend_error_v1(char* error_message) {
  deepmd::plugin::free_error_message(error_message);
}

#endif  // BUILD_TENSORFLOW
