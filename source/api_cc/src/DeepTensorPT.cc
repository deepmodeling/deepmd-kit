// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include "DeepTensorPT.h"

#include <torch/csrc/jit/runtime/jit_exception.h>

#include <cstdint>
#include <numeric>  // for std::iota
#include <sstream>

#include "common.h"
#include "device.h"
#include "errors.h"

using namespace deepmd;

static torch::Tensor createNlistTensor(
    const std::vector<std::vector<int>>& data) {
  size_t total_size = 0;
  for (const auto& row : data) {
    total_size += row.size();
  }
  std::vector<int> flat_data;
  flat_data.reserve(total_size);
  for (const auto& row : data) {
    flat_data.insert(flat_data.end(), row.begin(), row.end());
  }

  torch::Tensor flat_tensor = torch::tensor(flat_data, torch::kInt32);
  int nloc = data.size();
  int nnei = nloc > 0 ? total_size / nloc : 0;
  return flat_tensor.view({1, nloc, nnei});
}

void DeepTensorPT::translate_error(std::function<void()> f) {
  try {
    f();
    // it seems that libtorch may throw different types of exceptions which are
    // inherbited from different base classes
    // https://github.com/pytorch/pytorch/blob/13316a8d4642454012d34da0d742f1ba93fc0667/torch/csrc/jit/runtime/interpreter.cpp#L924-L939
  } catch (const c10::Error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  } catch (const torch::jit::JITException& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend JIT error: " +
                                   std::string(e.what()));
  } catch (const std::runtime_error& e) {
    throw deepmd::deepmd_exception("DeePMD-kit PyTorch backend error: " +
                                   std::string(e.what()));
  }
}

DeepTensorPT::DeepTensorPT() : inited(false) {}

DeepTensorPT::DeepTensorPT(const std::string& model,
                           const int& gpu_rank,
                           const std::string& name_scope_)
    : inited(false), name_scope(name_scope_) {
  try {
    translate_error([&] { init(model, gpu_rank, name_scope_); });
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    throw;
  }
}

void DeepTensorPT::init(const std::string& model,
                        const int& gpu_rank,
                        const std::string& name_scope_) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  name_scope = name_scope_;
  deepmd::load_op_library();
  int gpu_num = torch::cuda::device_count();
  if (gpu_num > 0) {
    gpu_id = gpu_rank % gpu_num;
  } else {
    gpu_id = 0;
  }
  torch::Device device(torch::kCUDA, gpu_id);
  gpu_enabled = torch::cuda::is_available();
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
    std::cout << "load model from: " << model << " to cpu " << std::endl;
  } else {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    DPErrcheck(DPSetDevice(gpu_id));
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    std::cout << "load model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }
  std::unordered_map<std::string, std::string> metadata = {{"type", ""}};
  module = torch::jit::load(model, device, metadata);
  module.eval();

  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  if (num_inter_nthreads) {
    try {
      at::set_num_interop_threads(num_inter_nthreads);
    } catch (...) {
    }
  }
  if (num_intra_nthreads) {
    try {
      at::set_num_threads(num_intra_nthreads);
    } catch (...) {
    }
  }

  // Get model properties using run_method for C++ interface
  auto rcut_result = module.run_method("get_rcut");
  rcut = rcut_result.toDouble();

  auto ntypes_result = module.run_method("get_ntypes");
  ntypes = ntypes_result.toInt();

  // Get task dimension from model method
  auto task_dim_result = module.run_method("get_task_dim");
  odim = task_dim_result.toInt();

  // Get type map and set up sel_type
  auto type_map_result = module.run_method("get_type_map");
  auto type_map_list = type_map_result.toList();
  sel_type.clear();

  // For PyTorch models, all types are included (the backend handles exclusions
  // internally) The model always outputs all types, but some results may be
  // zero
  for (size_t i = 0; i < type_map_list.size(); ++i) {
    sel_type.push_back(i);
  }
  inited = true;
}

DeepTensorPT::~DeepTensorPT() {}

void DeepTensorPT::get_type_map(std::string& type_map) {
  auto type_map_result = module.run_method("get_type_map");
  auto type_map_list = type_map_result.toList();
  type_map.clear();
  for (const torch::IValue& element : type_map_list) {
    if (!type_map.empty()) {
      type_map += " ";
    }
    type_map += torch::str(element);
  }
}

template <typename VALUETYPE>
void DeepTensorPT::compute(std::vector<VALUETYPE>& global_tensor,
                           std::vector<VALUETYPE>& force,
                           std::vector<VALUETYPE>& virial,
                           std::vector<VALUETYPE>& atom_tensor,
                           std::vector<VALUETYPE>& atom_virial,
                           const std::vector<VALUETYPE>& coord,
                           const std::vector<int>& atype,
                           const std::vector<VALUETYPE>& box,
                           const bool request_deriv) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }

  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);

  // Convert inputs to tensors
  std::vector<VALUETYPE> coord_wrapped = coord;
  at::Tensor coord_tensor =
      torch::from_blob(coord_wrapped.data(), {1, natoms, 3}, options)
          .to(device);

  std::vector<std::int64_t> atype_64(atype.begin(), atype.end());
  at::Tensor atype_tensor =
      torch::from_blob(atype_64.data(), {1, natoms}, int_options).to(device);

  c10::optional<torch::Tensor> box_tensor;
  if (!box.empty()) {
    box_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(box.data()), {1, 9}, options)
            .to(device);
  }

  // Create input vector
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(coord_tensor);
  inputs.push_back(atype_tensor);
  inputs.push_back(box_tensor);

  // Add None for fparam and aparam (not used by tensor models)
  inputs.push_back(torch::jit::IValue());  // fparam = None
  inputs.push_back(torch::jit::IValue());  // aparam = None
  inputs.push_back(request_deriv);         // do_atomic_virial

  // Forward pass through model
  c10::Dict<c10::IValue, c10::IValue> outputs =
      module.forward(inputs).toGenericDict();

  // Extract global dipole/polar results
  c10::IValue global_out;
  if (outputs.contains("global_dipole")) {
    global_out = outputs.at("global_dipole");
  } else if (outputs.contains("global_polar")) {
    global_out = outputs.at("global_polar");
  } else {
    throw deepmd::deepmd_exception(
        "Cannot find global tensor output in model results");
  }
  torch::Tensor flat_global_ = global_out.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_global_ = flat_global_.to(torch::kCPU);
  global_tensor.assign(cpu_global_.data_ptr<VALUETYPE>(),
                       cpu_global_.data_ptr<VALUETYPE>() + cpu_global_.numel());

  // Extract atomic dipole/polar results
  c10::IValue atom_out;
  if (outputs.contains("dipole")) {
    atom_out = outputs.at("dipole");
  } else if (outputs.contains("polar")) {
    atom_out = outputs.at("polar");
  } else {
    throw deepmd::deepmd_exception(
        "Cannot find atomic tensor output in model results");
  }
  torch::Tensor flat_atom_ = atom_out.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_atom_ = flat_atom_.to(torch::kCPU);
  atom_tensor.assign(cpu_atom_.data_ptr<VALUETYPE>(),
                     cpu_atom_.data_ptr<VALUETYPE>() + cpu_atom_.numel());

  // Extract force results
  c10::IValue force_ = outputs.at("force");
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  force.assign(cpu_force_.data_ptr<VALUETYPE>(),
               cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());

  // Extract virial results
  c10::IValue virial_ = outputs.at("virial");
  torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());
  // Extract atomic virial results if requested
  if (request_deriv) {
    c10::IValue atom_virial_ = outputs.at("atom_virial");
    torch::Tensor flat_atom_virial_ =
        atom_virial_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
    atom_virial.assign(
        cpu_atom_virial_.data_ptr<VALUETYPE>(),
        cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
  } else {
    atom_virial.clear();
  }
}

template <typename VALUETYPE>
void DeepTensorPT::compute(std::vector<VALUETYPE>& global_tensor,
                           std::vector<VALUETYPE>& force,
                           std::vector<VALUETYPE>& virial,
                           std::vector<VALUETYPE>& atom_tensor,
                           std::vector<VALUETYPE>& atom_virial,
                           const std::vector<VALUETYPE>& coord,
                           const std::vector<int>& atype,
                           const std::vector<VALUETYPE>& box,
                           const int nghost,
                           const InputNlist& lmp_list,
                           const bool request_deriv) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }

  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int32_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32);
  auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

  // Select real atoms following DeepPotPT pattern
  std::vector<VALUETYPE> dcoord, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  int nframes = 1;
  std::vector<VALUETYPE> aparam;  // Empty for tensor models
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, nframes, 0, nall, false);

  std::vector<VALUETYPE> coord_wrapped = dcoord;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option).to(device);

  // Process neighbor list following DeepPotPT pattern
  nlist_data.copy_from_nlist(lmp_list, nall - nghost);
  nlist_data.shuffle_exclude_empty(fwd_map);
  nlist_data.padding();

  at::Tensor firstneigh = createNlistTensor(nlist_data.jlist);
  firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);

  bool do_atom_virial_tensor = request_deriv;
  c10::optional<torch::Tensor> fparam_tensor;
  c10::optional<torch::Tensor> aparam_tensor;
  c10::optional<torch::Tensor> mapping_tensor;

  // Use forward_lower method following DeepPotPT pattern
  c10::Dict<c10::IValue, c10::IValue> outputs =
      module
          .run_method("forward_lower", coord_wrapped_Tensor, atype_Tensor,
                      firstneigh_tensor, mapping_tensor, fparam_tensor,
                      aparam_tensor, do_atom_virial_tensor)
          .toGenericDict();

  // Extract outputs following DeepPotPT pattern
  c10::IValue global_dipole_;
  if (outputs.contains("global_dipole")) {
    global_dipole_ = outputs.at("global_dipole");
  } else if (outputs.contains("global_polar")) {
    global_dipole_ = outputs.at("global_polar");
  } else {
    throw deepmd::deepmd_exception(
        "Cannot find global tensor output in model results");
  }
  // in Python, here used double; however, in TF C++, float is used
  // for consistency, we use float
  torch::Tensor flat_global_ =
      global_dipole_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_global_ = flat_global_.to(torch::kCPU);
  global_tensor.assign(cpu_global_.data_ptr<VALUETYPE>(),
                       cpu_global_.data_ptr<VALUETYPE>() + cpu_global_.numel());

  c10::IValue force_ = outputs.at("extended_force");
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  std::vector<VALUETYPE> dforce;
  dforce.assign(cpu_force_.data_ptr<VALUETYPE>(),
                cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());

  c10::IValue virial_ = outputs.at("virial");
  torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());

  // bkw map for forces
  force.resize(static_cast<size_t>(nframes) * odim * fwd_map.size() * 3);
  for (int kk = 0; kk < odim; ++kk) {
    select_map<VALUETYPE>(force.begin() + kk * fwd_map.size() * 3,
                          dforce.begin() + kk * bkw_map.size() * 3, bkw_map, 3);
  }

  // Extract atomic dipoles/polars if available
  c10::IValue atom_tensor_output;
  int task_dim;
  if (outputs.contains("dipole")) {
    atom_tensor_output = outputs.at("dipole");
    task_dim = 3;  // dipole has 3 components
  } else if (outputs.contains("polar")) {
    atom_tensor_output = outputs.at("polar");
    task_dim = 9;  // polarizability has 9 components typically
  } else {
    throw deepmd::deepmd_exception(
        "Cannot find atomic tensor output in model results");
  }

  torch::Tensor flat_atom_tensor_ =
      atom_tensor_output.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_atom_tensor_ = flat_atom_tensor_.to(torch::kCPU);
  std::vector<VALUETYPE> datom_tensor;
  datom_tensor.assign(
      cpu_atom_tensor_.data_ptr<VALUETYPE>(),
      cpu_atom_tensor_.data_ptr<VALUETYPE>() + cpu_atom_tensor_.numel());
  atom_tensor.resize(static_cast<size_t>(nframes) * fwd_map.size() * task_dim);
  select_map<VALUETYPE>(atom_tensor, datom_tensor, bkw_map, task_dim, nframes,
                        fwd_map.size(), nall_real);

  if (request_deriv) {
    c10::IValue atom_virial_ = outputs.at("extended_virial");
    torch::Tensor flat_atom_virial_ =
        atom_virial_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
    std::vector<VALUETYPE> datom_virial;
    datom_virial.assign(
        cpu_atom_virial_.data_ptr<VALUETYPE>(),
        cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
    atom_virial.resize(static_cast<size_t>(nframes) * odim * fwd_map.size() *
                       9);
    for (int kk = 0; kk < odim; ++kk) {
      select_map<VALUETYPE>(atom_virial.begin() + kk * fwd_map.size() * 9,
                            datom_virial.begin() + kk * bkw_map.size() * 9,
                            bkw_map, 9);
    }
  }
}

// Public wrapper functions
void DeepTensorPT::computew(std::vector<double>& global_tensor,
                            std::vector<double>& force,
                            std::vector<double>& virial,
                            std::vector<double>& atom_tensor,
                            std::vector<double>& atom_virial,
                            const std::vector<double>& coord,
                            const std::vector<int>& atype,
                            const std::vector<double>& box,
                            const bool request_deriv) {
  translate_error([&] {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box, request_deriv);
  });
}

void DeepTensorPT::computew(std::vector<float>& global_tensor,
                            std::vector<float>& force,
                            std::vector<float>& virial,
                            std::vector<float>& atom_tensor,
                            std::vector<float>& atom_virial,
                            const std::vector<float>& coord,
                            const std::vector<int>& atype,
                            const std::vector<float>& box,
                            const bool request_deriv) {
  translate_error([&] {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box, request_deriv);
  });
}

void DeepTensorPT::computew(std::vector<double>& global_tensor,
                            std::vector<double>& force,
                            std::vector<double>& virial,
                            std::vector<double>& atom_tensor,
                            std::vector<double>& atom_virial,
                            const std::vector<double>& coord,
                            const std::vector<int>& atype,
                            const std::vector<double>& box,
                            const int nghost,
                            const InputNlist& inlist,
                            const bool request_deriv) {
  translate_error([&] {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box, nghost, inlist, request_deriv);
  });
}

void DeepTensorPT::computew(std::vector<float>& global_tensor,
                            std::vector<float>& force,
                            std::vector<float>& virial,
                            std::vector<float>& atom_tensor,
                            std::vector<float>& atom_virial,
                            const std::vector<float>& coord,
                            const std::vector<int>& atype,
                            const std::vector<float>& box,
                            const int nghost,
                            const InputNlist& inlist,
                            const bool request_deriv) {
  translate_error([&] {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box, nghost, inlist, request_deriv);
  });
}

#endif  // BUILD_PYTORCH
