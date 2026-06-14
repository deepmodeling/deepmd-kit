// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "common.h"

namespace deepmd {
class DeepPotBackend;
class DeepSpinBackend;
class DeepTensorBase;
class DipoleChargeModifierBase;

/**
 * @brief C ABI symbol names exported by backend plugin libraries.
 **/
constexpr const char* DEEPMD_DEEPPOT_PLUGIN_CREATE_SYMBOL =
    "deepmd_create_deeppot_backend_v1";
constexpr const char* DEEPMD_DEEPPOT_PLUGIN_DELETE_SYMBOL =
    "deepmd_delete_deeppot_backend_v1";
constexpr const char* DEEPMD_DEEPSPIN_PLUGIN_CREATE_SYMBOL =
    "deepmd_create_deepspin_backend_v1";
constexpr const char* DEEPMD_DEEPSPIN_PLUGIN_DELETE_SYMBOL =
    "deepmd_delete_deepspin_backend_v1";
constexpr const char* DEEPMD_DEEPTENSOR_PLUGIN_CREATE_SYMBOL =
    "deepmd_create_deeptensor_backend_v1";
constexpr const char* DEEPMD_DEEPTENSOR_PLUGIN_DELETE_SYMBOL =
    "deepmd_delete_deeptensor_backend_v1";
constexpr const char* DEEPMD_DIPOLE_CHARGE_MODIFIER_PLUGIN_CREATE_SYMBOL =
    "deepmd_create_dipole_charge_modifier_backend_v1";
constexpr const char* DEEPMD_DIPOLE_CHARGE_MODIFIER_PLUGIN_DELETE_SYMBOL =
    "deepmd_delete_dipole_charge_modifier_backend_v1";
constexpr const char* DEEPMD_CONVERT_PBTXT_TO_PB_PLUGIN_SYMBOL =
    "deepmd_convert_pbtxt_to_pb_v1";
constexpr const char* DEEPMD_BACKEND_PLUGIN_FREE_ERROR_SYMBOL =
    "deepmd_free_backend_error_v1";

/**
 * @brief C ABI function pointer types exported by backend plugin libraries.
 **/
extern "C" {
typedef void* (*deepmd_create_deeppot_backend_fn)(const char* model,
                                                  int gpu_rank,
                                                  const char* file_content,
                                                  std::size_t file_content_size,
                                                  char** error_message);
typedef void (*deepmd_delete_deeppot_backend_fn)(void* backend);
typedef void* (*deepmd_create_deepspin_backend_fn)(
    const char* model,
    int gpu_rank,
    const char* file_content,
    std::size_t file_content_size,
    char** error_message);
typedef void (*deepmd_delete_deepspin_backend_fn)(void* backend);
typedef void* (*deepmd_create_deeptensor_backend_fn)(const char* model,
                                                     int gpu_rank,
                                                     const char* name_scope,
                                                     char** error_message);
typedef void (*deepmd_delete_deeptensor_backend_fn)(void* backend);
typedef void* (*deepmd_create_dipole_charge_modifier_backend_fn)(
    const char* model,
    int gpu_rank,
    const char* name_scope,
    char** error_message);
typedef void (*deepmd_delete_dipole_charge_modifier_backend_fn)(void* backend);
typedef int (*deepmd_convert_pbtxt_to_pb_fn)(const char* pbtxt,
                                             const char* pb,
                                             char** error_message);
typedef void (*deepmd_free_backend_error_fn)(char* error_message);
}

/**
 * @brief Create a DeepPot backend through a runtime backend plugin.
 * @param[in] backend The backend plugin to load.
 * @param[in] model The name of the frozen model file.
 * @param[in] gpu_rank The GPU rank.
 * @param[in] file_content The content of the model file. If it is not empty,
 * the backend will read from the string instead of the file.
 * @return The DeepPot backend instance.
 **/
std::shared_ptr<DeepPotBackend> create_deeppot_backend_from_plugin(
    DPBackend backend,
    const std::string& model,
    const int& gpu_rank,
    const std::string& file_content);

/**
 * @brief Create a DeepSpin backend through a runtime backend plugin.
 * @param[in] backend The backend plugin to load.
 * @param[in] model The name of the frozen model file.
 * @param[in] gpu_rank The GPU rank.
 * @param[in] file_content The content of the model file. If it is not empty,
 * the backend will read from the string instead of the file.
 * @return The DeepSpin backend instance.
 **/
std::shared_ptr<DeepSpinBackend> create_deepspin_backend_from_plugin(
    DPBackend backend,
    const std::string& model,
    const int& gpu_rank,
    const std::string& file_content);

/**
 * @brief Create a DeepTensor backend through a runtime backend plugin.
 * @param[in] backend The backend plugin to load.
 * @param[in] model The name of the frozen model file.
 * @param[in] gpu_rank The GPU rank.
 * @param[in] name_scope The TensorFlow name scope.
 * @return The DeepTensor backend instance.
 **/
std::shared_ptr<DeepTensorBase> create_deeptensor_backend_from_plugin(
    DPBackend backend,
    const std::string& model,
    const int& gpu_rank,
    const std::string& name_scope);

/**
 * @brief Create a DipoleChargeModifier backend through a runtime backend
 * plugin.
 * @param[in] backend The backend plugin to load.
 * @param[in] model The name of the frozen model file.
 * @param[in] gpu_rank The GPU rank.
 * @param[in] name_scope The TensorFlow name scope.
 * @return The DipoleChargeModifier backend instance.
 **/
std::shared_ptr<DipoleChargeModifierBase>
create_dipole_charge_modifier_backend_from_plugin(
    DPBackend backend,
    const std::string& model,
    const int& gpu_rank,
    const std::string& name_scope);

/**
 * @brief Convert a TensorFlow pbtxt model to pb through the TensorFlow backend
 * plugin.
 * @param[in] fn_pb_txt The input pbtxt file.
 * @param[in] fn_pb The output pb file.
 **/
void convert_pbtxt_to_pb_from_plugin(const std::string& fn_pb_txt,
                                     const std::string& fn_pb);

/**
 * @brief Get the display name of a backend.
 * @param[in] backend The backend enum value.
 * @return The backend display name.
 **/
std::string backend_name(DPBackend backend);

}  // namespace deepmd
