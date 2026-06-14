// SPDX-License-Identifier: LGPL-3.0-or-later
#include "BackendPlugin.h"

#include <cstdlib>
#include <exception>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#include "DataModifier.h"
#include "DeepPot.h"
#include "DeepSpin.h"
#include "DeepTensor.h"

#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

struct PluginHandle {
  void* handle = nullptr;
  deepmd::deepmd_create_deeppot_backend_fn create_deeppot = nullptr;
  deepmd::deepmd_delete_deeppot_backend_fn delete_deeppot = nullptr;
  deepmd::deepmd_free_backend_error_fn free_error = nullptr;
  std::string path;
};

std::string path_separator() {
#if defined(_WIN32)
  return ";";
#else
  return ":";
#endif
}

std::vector<std::string> split_paths(const std::string& paths) {
  std::vector<std::string> result;
  const std::string sep = path_separator();
  std::string::size_type begin = 0;
  while (begin <= paths.size()) {
    const std::string::size_type end = paths.find(sep, begin);
    std::string item = paths.substr(
        begin, end == std::string::npos ? std::string::npos : end - begin);
    if (!item.empty()) {
      result.push_back(item);
    }
    if (end == std::string::npos) {
      break;
    }
    begin = end + sep.size();
  }
  return result;
}

std::string dirname_of(const std::string& path) {
  const std::string::size_type pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return "";
  }
  return path.substr(0, pos);
}

std::string join_path(const std::string& dir, const std::string& name) {
  if (dir.empty()) {
    return name;
  }
  const char last = dir[dir.size() - 1];
  if (last == '/' || last == '\\') {
    return dir + name;
  }
#if defined(_WIN32)
  return dir + "\\" + name;
#else
  return dir + "/" + name;
#endif
}

std::string current_library_dir() {
#if defined(_WIN32)
  HMODULE module = nullptr;
  if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&current_library_dir),
                          &module)) {
    return "";
  }
  char path[MAX_PATH];
  const DWORD size = GetModuleFileNameA(module, path, MAX_PATH);
  if (size == 0 || size == MAX_PATH) {
    return "";
  }
  return dirname_of(std::string(path, size));
#else
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&current_library_dir), &info) == 0 ||
      info.dli_fname == nullptr) {
    return "";
  }
  return dirname_of(info.dli_fname);
#endif
}

std::string library_file_name(const std::string& library_name) {
#if defined(_WIN32)
  return library_name + ".dll";
#elif defined(__APPLE__)
  return "lib" + library_name + ".dylib";
#else
  return "lib" + library_name + ".so";
#endif
}

std::string backend_library_name(deepmd::DPBackend backend) {
  switch (backend) {
    case deepmd::DPBackend::TensorFlow:
      return "deepmd_backend_tf";
    case deepmd::DPBackend::PyTorch:
      return "deepmd_backend_pt";
    case deepmd::DPBackend::PyTorchExportable:
      return "deepmd_backend_ptexpt";
    case deepmd::DPBackend::Paddle:
      return "deepmd_backend_pd";
    case deepmd::DPBackend::JAX:
      return "deepmd_backend_jax";
    default:
      throw deepmd::deepmd_exception("Unknown backend plugin request");
  }
}

void* open_library(const std::string& path, std::string& error) {
#if defined(_WIN32)
  HMODULE handle = LoadLibraryA(path.c_str());
  if (handle == nullptr) {
    error = "LoadLibrary failed";
  }
  return reinterpret_cast<void*>(handle);
#else
  dlerror();
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* dl_error = dlerror();
    error = dl_error == nullptr ? "dlopen failed" : dl_error;
  }
  return handle;
#endif
}

void* load_symbol(void* handle,
                  const std::string& path,
                  const char* symbol_name) {
#if defined(_WIN32)
  void* symbol = reinterpret_cast<void*>(
      GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol_name));
  if (symbol == nullptr) {
    throw deepmd::deepmd_exception("Backend plugin " + path +
                                   " does not export " + symbol_name);
  }
  return symbol;
#else
  dlerror();
  void* symbol = dlsym(handle, symbol_name);
  const char* dl_error = dlerror();
  if (dl_error != nullptr) {
    throw deepmd::deepmd_exception("Backend plugin " + path +
                                   " does not export " + symbol_name + ": " +
                                   std::string(dl_error));
  }
  return symbol;
#endif
}

template <typename FunctionType>
FunctionType load_typed_symbol(const std::shared_ptr<PluginHandle>& plugin,
                               const char* symbol_name) {
  return reinterpret_cast<FunctionType>(
      load_symbol(plugin->handle, plugin->path, symbol_name));
}

std::shared_ptr<PluginHandle> load_plugin(deepmd::DPBackend backend) {
  static std::mutex mutex;
  static std::map<deepmd::DPBackend, std::shared_ptr<PluginHandle>> plugins;

  std::lock_guard<std::mutex> lock(mutex);
  const auto iter = plugins.find(backend);
  if (iter != plugins.end()) {
    return iter->second;
  }

  const std::string backend_name = deepmd::backend_name(backend);
  const std::string library_name =
      library_file_name(backend_library_name(backend));
  std::vector<std::string> candidates;

  const char* env_plugin_path = std::getenv("DP_BACKEND_PLUGIN_PATH");
  if (env_plugin_path != nullptr) {
    for (const auto& dir : split_paths(env_plugin_path)) {
      candidates.push_back(join_path(dir, library_name));
    }
  }

  const std::string own_dir = current_library_dir();
  if (!own_dir.empty()) {
    candidates.push_back(join_path(own_dir, library_name));
  }
  candidates.push_back(library_name);

  std::ostringstream errors;
  for (const auto& candidate : candidates) {
    std::string error;
    void* handle = open_library(candidate, error);
    if (handle == nullptr) {
      errors << "\n  " << candidate << ": " << error;
      continue;
    }

    std::shared_ptr<PluginHandle> plugin(new PluginHandle);
    plugin->handle = handle;
    plugin->path = candidate;
    plugin->create_deeppot =
        reinterpret_cast<deepmd::deepmd_create_deeppot_backend_fn>(load_symbol(
            handle, candidate, deepmd::DEEPMD_DEEPPOT_PLUGIN_CREATE_SYMBOL));
    plugin->delete_deeppot =
        reinterpret_cast<deepmd::deepmd_delete_deeppot_backend_fn>(load_symbol(
            handle, candidate, deepmd::DEEPMD_DEEPPOT_PLUGIN_DELETE_SYMBOL));
    plugin->free_error = reinterpret_cast<deepmd::deepmd_free_backend_error_fn>(
        load_symbol(handle, candidate,
                    deepmd::DEEPMD_BACKEND_PLUGIN_FREE_ERROR_SYMBOL));
    plugins[backend] = plugin;
    return plugin;
  }

  throw deepmd::deepmd_exception(
      "Unable to load " + backend_name + " backend plugin (" + library_name +
      "). Set DP_BACKEND_PLUGIN_PATH or install the plugin next to "
      "libdeepmd_cc. Tried:" +
      errors.str());
}

}  // namespace

std::string deepmd::backend_name(DPBackend backend) {
  switch (backend) {
    case deepmd::DPBackend::TensorFlow:
      return "TensorFlow";
    case deepmd::DPBackend::PyTorch:
      return "PyTorch";
    case deepmd::DPBackend::PyTorchExportable:
      return "PyTorch Exportable";
    case deepmd::DPBackend::Paddle:
      return "PaddlePaddle";
    case deepmd::DPBackend::JAX:
      return "JAX";
    case deepmd::DPBackend::Unknown:
      return "Unknown";
  }
  return "Unknown";
}

static std::string take_plugin_error(
    const std::shared_ptr<PluginHandle>& plugin, char* error_message) {
  std::string message;
  if (error_message != nullptr) {
    message = error_message;
    plugin->free_error(error_message);
  }
  return message;
}

template <typename Function>
void* call_backend_create(const std::shared_ptr<PluginHandle>& plugin,
                          char*& error_message,
                          const std::string& action,
                          Function create) {
  try {
    return create();
  } catch (const std::exception& e) {
    std::string message = take_plugin_error(plugin, error_message);
    throw deepmd::deepmd_exception(action + " from " + plugin->path +
                                   ": plugin threw an exception: " + e.what() +
                                   (message.empty() ? "" : ": " + message));
  } catch (...) {
    std::string message = take_plugin_error(plugin, error_message);
    throw deepmd::deepmd_exception(action + " from " + plugin->path +
                                   ": plugin threw an unknown exception" +
                                   (message.empty() ? "" : ": " + message));
  }
}

std::shared_ptr<deepmd::DeepPotBackend>
deepmd::create_deeppot_backend_from_plugin(DPBackend backend,
                                           const std::string& model,
                                           const int& gpu_rank,
                                           const std::string& file_content) {
  std::shared_ptr<PluginHandle> plugin = load_plugin(backend);

  char* error_message = nullptr;
  const std::string action =
      "Failed to create " + backend_name(backend) + " DeepPot backend";
  void* backend_handle =
      call_backend_create(plugin, error_message, action, [&]() {
        return plugin->create_deeppot(model.c_str(), gpu_rank,
                                      file_content.data(), file_content.size(),
                                      &error_message);
      });

  if (backend_handle == nullptr) {
    std::string message = take_plugin_error(plugin, error_message);
    throw deepmd::deepmd_exception(action + " from " + plugin->path +
                                   (message.empty() ? "" : ": " + message));
  }

  if (error_message != nullptr) {
    plugin->free_error(error_message);
  }

  return std::shared_ptr<DeepPotBackend>(
      static_cast<DeepPotBackend*>(backend_handle),
      [plugin](DeepPotBackend* ptr) { plugin->delete_deeppot(ptr); });
}

std::shared_ptr<deepmd::DeepSpinBackend>
deepmd::create_deepspin_backend_from_plugin(DPBackend backend,
                                            const std::string& model,
                                            const int& gpu_rank,
                                            const std::string& file_content) {
  std::shared_ptr<PluginHandle> plugin = load_plugin(backend);
  deepmd_create_deepspin_backend_fn create_deepspin =
      load_typed_symbol<deepmd_create_deepspin_backend_fn>(
          plugin, DEEPMD_DEEPSPIN_PLUGIN_CREATE_SYMBOL);
  deepmd_delete_deepspin_backend_fn delete_deepspin =
      load_typed_symbol<deepmd_delete_deepspin_backend_fn>(
          plugin, DEEPMD_DEEPSPIN_PLUGIN_DELETE_SYMBOL);

  char* error_message = nullptr;
  const std::string action =
      "Failed to create " + backend_name(backend) + " DeepSpin backend";
  void* backend_handle =
      call_backend_create(plugin, error_message, action, [&]() {
        return create_deepspin(model.c_str(), gpu_rank, file_content.data(),
                               file_content.size(), &error_message);
      });
  if (backend_handle == nullptr) {
    std::string message = take_plugin_error(plugin, error_message);
    throw deepmd::deepmd_exception(action + " from " + plugin->path +
                                   (message.empty() ? "" : ": " + message));
  }
  if (error_message != nullptr) {
    plugin->free_error(error_message);
  }
  return std::shared_ptr<DeepSpinBackend>(
      static_cast<DeepSpinBackend*>(backend_handle),
      [plugin, delete_deepspin](DeepSpinBackend* ptr) {
        delete_deepspin(ptr);
      });
}

std::shared_ptr<deepmd::DeepTensorBase>
deepmd::create_deeptensor_backend_from_plugin(DPBackend backend,
                                              const std::string& model,
                                              const int& gpu_rank,
                                              const std::string& name_scope) {
  std::shared_ptr<PluginHandle> plugin = load_plugin(backend);
  deepmd_create_deeptensor_backend_fn create_deeptensor =
      load_typed_symbol<deepmd_create_deeptensor_backend_fn>(
          plugin, DEEPMD_DEEPTENSOR_PLUGIN_CREATE_SYMBOL);
  deepmd_delete_deeptensor_backend_fn delete_deeptensor =
      load_typed_symbol<deepmd_delete_deeptensor_backend_fn>(
          plugin, DEEPMD_DEEPTENSOR_PLUGIN_DELETE_SYMBOL);

  char* error_message = nullptr;
  const std::string action =
      "Failed to create " + backend_name(backend) + " DeepTensor backend";
  void* backend_handle =
      call_backend_create(plugin, error_message, action, [&]() {
        return create_deeptensor(model.c_str(), gpu_rank, name_scope.c_str(),
                                 &error_message);
      });
  if (backend_handle == nullptr) {
    std::string message = take_plugin_error(plugin, error_message);
    throw deepmd::deepmd_exception(action + " from " + plugin->path +
                                   (message.empty() ? "" : ": " + message));
  }
  if (error_message != nullptr) {
    plugin->free_error(error_message);
  }
  return std::shared_ptr<DeepTensorBase>(
      static_cast<DeepTensorBase*>(backend_handle),
      [plugin, delete_deeptensor](DeepTensorBase* ptr) {
        delete_deeptensor(ptr);
      });
}

std::shared_ptr<deepmd::DipoleChargeModifierBase>
deepmd::create_dipole_charge_modifier_backend_from_plugin(
    DPBackend backend,
    const std::string& model,
    const int& gpu_rank,
    const std::string& name_scope) {
  std::shared_ptr<PluginHandle> plugin = load_plugin(backend);
  deepmd_create_dipole_charge_modifier_backend_fn create_modifier =
      load_typed_symbol<deepmd_create_dipole_charge_modifier_backend_fn>(
          plugin, DEEPMD_DIPOLE_CHARGE_MODIFIER_PLUGIN_CREATE_SYMBOL);
  deepmd_delete_dipole_charge_modifier_backend_fn delete_modifier =
      load_typed_symbol<deepmd_delete_dipole_charge_modifier_backend_fn>(
          plugin, DEEPMD_DIPOLE_CHARGE_MODIFIER_PLUGIN_DELETE_SYMBOL);

  char* error_message = nullptr;
  const std::string action = "Failed to create " + backend_name(backend) +
                             " DipoleChargeModifier backend";
  void* backend_handle =
      call_backend_create(plugin, error_message, action, [&]() {
        return create_modifier(model.c_str(), gpu_rank, name_scope.c_str(),
                               &error_message);
      });
  if (backend_handle == nullptr) {
    std::string message = take_plugin_error(plugin, error_message);
    throw deepmd::deepmd_exception(action + " from " + plugin->path +
                                   (message.empty() ? "" : ": " + message));
  }
  if (error_message != nullptr) {
    plugin->free_error(error_message);
  }
  return std::shared_ptr<DipoleChargeModifierBase>(
      static_cast<DipoleChargeModifierBase*>(backend_handle),
      [plugin, delete_modifier](DipoleChargeModifierBase* ptr) {
        delete_modifier(ptr);
      });
}

void deepmd::convert_pbtxt_to_pb_from_plugin(const std::string& fn_pb_txt,
                                             const std::string& fn_pb) {
  std::shared_ptr<PluginHandle> plugin = load_plugin(DPBackend::TensorFlow);
  deepmd_convert_pbtxt_to_pb_fn convert_pbtxt_to_pb =
      load_typed_symbol<deepmd_convert_pbtxt_to_pb_fn>(
          plugin, DEEPMD_CONVERT_PBTXT_TO_PB_PLUGIN_SYMBOL);

  char* error_message = nullptr;
  int status = 1;
  try {
    status =
        convert_pbtxt_to_pb(fn_pb_txt.c_str(), fn_pb.c_str(), &error_message);
  } catch (const deepmd::deepmd_exception&) {
    throw;
  } catch (const std::exception& e) {
    throw deepmd::deepmd_exception("Backend plugin " + plugin->path +
                                   " threw an exception: " + e.what());
  } catch (...) {
    throw deepmd::deepmd_exception("Backend plugin " + plugin->path +
                                   " threw an unknown exception");
  }

  if (status != 0) {
    std::string message = take_plugin_error(plugin, error_message);
    throw deepmd::deepmd_exception(
        "Failed to convert pbtxt with TensorFlow backend plugin from " +
        plugin->path + (message.empty() ? "" : ": " + message));
  }

  if (error_message != nullptr) {
    plugin->free_error(error_message);
  }
}
