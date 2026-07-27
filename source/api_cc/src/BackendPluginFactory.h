// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#include "BackendPlugin.h"
#include "DataModifier.h"
#include "DeepPot.h"
#include "DeepSpin.h"
#include "DeepTensor.h"
#include "errors.h"

namespace deepmd {
namespace plugin {

inline void set_error_message(char** error_message,
                              const std::string& message) {
  if (error_message == nullptr) {
    return;
  }
  char* buffer = static_cast<char*>(std::malloc(message.size() + 1));
  if (buffer == nullptr) {
    *error_message = nullptr;
    return;
  }
  std::memcpy(buffer, message.c_str(), message.size() + 1);
  *error_message = buffer;
}

template <typename Backend>
void* create_deeppot_backend(const char* model,
                             int gpu_rank,
                             const char* file_content,
                             std::size_t file_content_size,
                             char** error_message) {
  try {
    const std::string content(file_content == nullptr ? "" : file_content,
                              file_content_size);
    return static_cast<DeepPotBackend*>(
        new Backend(model == nullptr ? "" : model, gpu_rank, content));
  } catch (const deepmd::deepmd_exception& e) {
    set_error_message(error_message, e.what());
  } catch (const std::exception& e) {
    set_error_message(error_message, e.what());
  } catch (...) {
    set_error_message(error_message, "unknown backend plugin error");
  }
  return nullptr;
}

template <typename Backend>
void* create_deepspin_backend(const char* model,
                              int gpu_rank,
                              const char* file_content,
                              std::size_t file_content_size,
                              char** error_message) {
  try {
    const std::string content(file_content == nullptr ? "" : file_content,
                              file_content_size);
    return static_cast<DeepSpinBackend*>(
        new Backend(model == nullptr ? "" : model, gpu_rank, content));
  } catch (const deepmd::deepmd_exception& e) {
    set_error_message(error_message, e.what());
  } catch (const std::exception& e) {
    set_error_message(error_message, e.what());
  } catch (...) {
    set_error_message(error_message, "unknown backend plugin error");
  }
  return nullptr;
}

template <typename Backend>
void* create_deeptensor_backend(const char* model,
                                int gpu_rank,
                                const char* name_scope,
                                char** error_message) {
  try {
    return static_cast<DeepTensorBase*>(
        new Backend(model == nullptr ? "" : model, gpu_rank,
                    name_scope == nullptr ? "" : name_scope));
  } catch (const deepmd::deepmd_exception& e) {
    set_error_message(error_message, e.what());
  } catch (const std::exception& e) {
    set_error_message(error_message, e.what());
  } catch (...) {
    set_error_message(error_message, "unknown backend plugin error");
  }
  return nullptr;
}

template <typename Backend>
void* create_dipole_charge_modifier_backend(const char* model,
                                            int gpu_rank,
                                            const char* name_scope,
                                            char** error_message) {
  try {
    return static_cast<DipoleChargeModifierBase*>(
        new Backend(model == nullptr ? "" : model, gpu_rank,
                    name_scope == nullptr ? "" : name_scope));
  } catch (const deepmd::deepmd_exception& e) {
    set_error_message(error_message, e.what());
  } catch (const std::exception& e) {
    set_error_message(error_message, e.what());
  } catch (...) {
    set_error_message(error_message, "unknown backend plugin error");
  }
  return nullptr;
}

inline void delete_deeppot_backend(void* backend) {
  delete static_cast<DeepPotBackend*>(backend);
}

inline void delete_deepspin_backend(void* backend) {
  delete static_cast<DeepSpinBackend*>(backend);
}

inline void delete_deeptensor_backend(void* backend) {
  delete static_cast<DeepTensorBase*>(backend);
}

inline void delete_dipole_charge_modifier_backend(void* backend) {
  delete static_cast<DipoleChargeModifierBase*>(backend);
}

inline void free_error_message(char* error_message) {
  std::free(error_message);
}

}  // namespace plugin
}  // namespace deepmd
