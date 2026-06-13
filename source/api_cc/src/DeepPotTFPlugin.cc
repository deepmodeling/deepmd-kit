// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_TENSORFLOW

#include "BackendPluginFactory.h"
#include "DataModifierTF.h"
#include "DeepPotTF.h"
#include "DeepSpinTF.h"
#include "DeepTensorTF.h"

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

extern "C" void deepmd_free_backend_error_v1(char* error_message) {
  deepmd::plugin::free_error_message(error_message);
}

#endif  // BUILD_TENSORFLOW
