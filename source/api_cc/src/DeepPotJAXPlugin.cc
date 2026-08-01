// SPDX-License-Identifier: LGPL-3.0-or-later
#if defined(BUILD_TENSORFLOW) || defined(BUILD_JAX)

#include "BackendPluginFactory.h"
#include "DeepPotJAX.h"

extern "C" void* deepmd_create_deeppot_backend_v1(const char* model,
                                                  int gpu_rank,
                                                  const char* file_content,
                                                  std::size_t file_content_size,
                                                  char** error_message) {
  return deepmd::plugin::create_deeppot_backend<deepmd::DeepPotJAX>(
      model, gpu_rank, file_content, file_content_size, error_message);
}

extern "C" void deepmd_delete_deeppot_backend_v1(void* backend) {
  deepmd::plugin::delete_deeppot_backend(backend);
}

extern "C" void deepmd_free_backend_error_v1(char* error_message) {
  deepmd::plugin::free_error_message(error_message);
}

#endif  // defined(BUILD_TENSORFLOW) || defined(BUILD_JAX)
