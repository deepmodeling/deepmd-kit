// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPotPTExpt.h"
#include "DeepSpinPTExpt.h"

#ifdef BUILD_PYTORCH

#include "BackendPluginFactory.h"

extern "C" void* deepmd_create_deeppot_backend_v1(const char* model,
                                                  int gpu_rank,
                                                  const char* file_content,
                                                  std::size_t file_content_size,
                                                  char** error_message) {
#if BUILD_PT_EXPT
  return deepmd::plugin::create_deeppot_backend<deepmd::DeepPotPTExpt>(
      model, gpu_rank, file_content, file_content_size, error_message);
#else
  deepmd::plugin::set_error_message(
      error_message,
      "PyTorch Exportable backend is not available (missing AOTInductor "
      "headers at build time)");
  return nullptr;
#endif
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
#if BUILD_PT_EXPT_SPIN
  return deepmd::plugin::create_deepspin_backend<deepmd::DeepSpinPTExpt>(
      model, gpu_rank, file_content, file_content_size, error_message);
#else
  deepmd::plugin::set_error_message(
      error_message,
      "PyTorch Exportable spin backend is not available (missing AOTInductor "
      "headers at build time)");
  return nullptr;
#endif
}

extern "C" void deepmd_delete_deepspin_backend_v1(void* backend) {
  deepmd::plugin::delete_deepspin_backend(backend);
}

extern "C" void deepmd_free_backend_error_v1(char* error_message) {
  deepmd::plugin::free_error_message(error_message);
}

#endif  // BUILD_PYTORCH
