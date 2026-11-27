// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPotPTExpt.h"
#include "DeepSpinPTExpt.h"
#include "NativeSpinPTExpt.h"

#ifdef BUILD_PYTORCH

#include <exception>
#include <string>

#include "BackendPluginFactory.h"

#if BUILD_PT_EXPT_SPIN && BUILD_PT_EXPT_NATIVE_SPIN
#include "commonPTExpt.h"

namespace {

/**
 * @brief Whether ``NativeSpinPTExpt`` serves an archive.
 *
 * The two spin schemes are served by different classes, and only the archive
 * itself says which one it holds: every spin artifact declares its scheme in
 * ``spin_scheme``. Anything else -- the virtual-atom scheme, a spin archive
 * of the pt backend, which declares no scheme, and an archive whose
 * metadata cannot be read -- stays with ``DeepSpinPTExpt``, so the
 * diagnostic for a malformed file comes from the loader that owns the format
 * rather than from this dispatch. Both lowers of a native-spin model carry the
 * ghost-feature exchange of domain decomposition in their main program, so
 * one class serves every native-spin archive on one rank or several.
 *
 * The lower-forward schema takes no part here: it is an internal branch of
 * whichever class the scheme selects.
 */
bool native_spin_backend_serves(const char* model) {
  if (model == nullptr) {
    return false;
  }
  try {
    const auto metadata = deepmd::ptexpt::parse_json(
        deepmd::ptexpt::read_zip_entry(model, "extra/metadata.json"));
    return metadata.obj_val.count("spin_scheme") &&
           metadata["spin_scheme"].as_string() == "native";
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace
#endif

extern "C" void* deepmd_create_deeppot_backend_v3(const char* model,
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

extern "C" void deepmd_delete_deeppot_backend_v3(void* backend) {
  deepmd::plugin::delete_deeppot_backend(backend);
}

extern "C" void* deepmd_create_deepspin_backend_v3(
    const char* model,
    int gpu_rank,
    const char* file_content,
    std::size_t file_content_size,
    char** error_message) {
#if BUILD_PT_EXPT_SPIN
#if BUILD_PT_EXPT_NATIVE_SPIN
  if (native_spin_backend_serves(model)) {
    return deepmd::plugin::create_deepspin_backend<deepmd::NativeSpinPTExpt>(
        model, gpu_rank, file_content, file_content_size, error_message);
  }
#endif
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

extern "C" void deepmd_delete_deepspin_backend_v3(void* backend) {
  deepmd::plugin::delete_deepspin_backend(backend);
}

extern "C" void deepmd_free_backend_error_v1(char* error_message) {
  deepmd::plugin::free_error_message(error_message);
}

#endif  // BUILD_PYTORCH
