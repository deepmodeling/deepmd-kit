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
 * ``spin_scheme``. Anything else -- the virtual-atom scheme, an archive
 * frozen before the field existed, and an archive whose metadata cannot be
 * read -- stays with ``DeepSpinPTExpt``, so the diagnostic for a malformed
 * file comes from the loader that owns the format rather than from this
 * dispatch.
 *
 * A native-spin archive that declares ``has_comm_artifact`` also stays with
 * ``DeepSpinPTExpt``. That flag marks the nested with-comm artifact carrying
 * the per-layer ghost-feature exchange that domain decomposition needs, and
 * ``DeepSpinPTExpt`` is the only class that drives it. The conjunct states a
 * capability of the serving class rather than a property of any descriptor
 * family, so it drops out once ``NativeSpinPTExpt`` gains a with-comm route
 * of its own.
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
    const bool native_scheme = metadata.obj_val.count("spin_scheme") &&
                               metadata["spin_scheme"].as_string() == "native";
    const bool needs_with_comm = metadata.obj_val.count("has_comm_artifact") &&
                                 metadata["has_comm_artifact"].as_bool();
    return native_scheme && !needs_with_comm;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace
#endif

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

extern "C" void deepmd_delete_deepspin_backend_v1(void* backend) {
  deepmd::plugin::delete_deepspin_backend(backend);
}

extern "C" void deepmd_free_backend_error_v1(char* error_message) {
  deepmd::plugin::free_error_message(error_message);
}

#endif  // BUILD_PYTORCH
