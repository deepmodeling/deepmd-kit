// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include "DeepPotPT.h"

#include "common.h"
using namespace deepmd;
torch::Tensor createNlistTensor(const std::vector<std::vector<int>>& data) {
  std::vector<torch::Tensor> row_tensors;

  for (const auto& row : data) {
    torch::Tensor row_tensor = torch::tensor(row, torch::kInt32).unsqueeze(0);
    row_tensors.push_back(row_tensor);
  }

  torch::Tensor tensor = torch::cat(row_tensors, 0).unsqueeze(0);
  return tensor;
}
DeepPotPT::DeepPotPT() : inited(false) {}
DeepPotPT::DeepPotPT(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content)
    : inited(false) {
  try {
    init(model, gpu_rank, file_content);
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    throw;
  }
}
void DeepPotPT::init(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  gpu_id = gpu_rank;
  torch::Device device(torch::kCUDA, gpu_rank);
  gpu_enabled = torch::cuda::is_available();
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
    std::cout << "load model from: " << model << " to cpu " << gpu_rank
              << std::endl;
  } else {
    std::cout << "load model from: " << model << " to gpu " << gpu_rank
              << std::endl;
  }
  module = torch::jit::load(model, device);

  torch::jit::FusionStrategy strategy;
  strategy = {{torch::jit::FusionBehavior::DYNAMIC, 10}};
  torch::jit::setFusionStrategy(strategy);

  get_env_nthreads(num_intra_nthreads,
                   num_inter_nthreads);  // need to be fixed as
                                         // DP_INTRA_OP_PARALLELISM_THREADS
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

  auto rcut_ = module.run_method("get_rcut").toDouble();
  rcut = static_cast<double>(rcut_);
  ntypes = module.run_method("get_ntypes").toInt();
  ntypes_spin = 0;
  dfparam = module.run_method("get_dim_fparam").toInt();
  daparam = module.run_method("get_dim_aparam").toInt();
  aparam_nall = module.run_method("is_aparam_nall").toBool();
  inited = true;
}
DeepPotPT::~DeepPotPT() {}

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPT::compute(ENERGYVTYPE& ener,
                        std::vector<VALUETYPE>& force,
                        std::vector<VALUETYPE>& virial,
                        std::vector<VALUETYPE>& atom_energy,
                        std::vector<VALUETYPE>& atom_virial,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const int nghost,
                        const InputNlist& lmp_list,
                        const int& ago,
                        const std::vector<VALUETYPE>& fparam,
                        const std::vector<VALUETYPE>& aparam) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same_v<VALUETYPE, float>) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);
  auto int32_options = torch::TensorOptions().dtype(torch::kInt32);

  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam_, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  std::cout << datype.size() << std::endl;
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  std::vector<int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_options).to(device);
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
  }
  at::Tensor firstneigh = createNlistTensor(nlist_data.jlist);
  firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);
  bool do_atom_virial_tensor = true;
  c10::optional<torch::Tensor> optional_tensor;
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<long int>(fparam.size())}, options)
            .to(device);
  }
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam_.empty()) {
    aparam_tensor = torch::from_blob(
                        const_cast<VALUETYPE*>(aparam_.data()),
                        {1, lmp_list.inum,
                         static_cast<long int>(aparam_.size()) / lmp_list.inum},
                        options)
                        .to(device);
  }
  c10::Dict<c10::IValue, c10::IValue> outputs =
      module
          .run_method("forward_lower", coord_wrapped_Tensor, atype_Tensor,
                      firstneigh_tensor, optional_tensor, fparam_tensor,
                      aparam_tensor, do_atom_virial_tensor)
          .toGenericDict();
  c10::IValue energy_ = outputs.at("energy");
  c10::IValue force_ = outputs.at("extended_force");
  c10::IValue virial_ = outputs.at("virial");
  c10::IValue atom_virial_ = outputs.at("extended_virial");
  c10::IValue atom_energy_ = outputs.at("atom_energy");
  torch::Tensor flat_energy_ = energy_.toTensor().view({-1});
  torch::Tensor cpu_energy_ = flat_energy_.to(torch::kCPU);
  ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
              cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
  torch::Tensor flat_atom_energy_ =
      atom_energy_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_atom_energy_ = flat_atom_energy_.to(torch::kCPU);
  datom_energy.resize(nall_real,
                      0.0);  // resize to nall to be consistenet with TF.
  datom_energy.assign(
      cpu_atom_energy_.data_ptr<VALUETYPE>(),
      cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  dforce.assign(cpu_force_.data_ptr<VALUETYPE>(),
                cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());
  torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());
  torch::Tensor flat_atom_virial_ =
      atom_virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
  datom_virial.assign(
      cpu_atom_virial_.data_ptr<VALUETYPE>(),
      cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
  int nframes = 1;
  // bkw map
  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
  atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                        fwd_map.size(), nall_real);
  select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                        fwd_map.size(), nall_real);
}
template void DeepPotPT::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);
template void DeepPotPT::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);
template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPT::compute(ENERGYVTYPE& ener,
                        std::vector<VALUETYPE>& force,
                        std::vector<VALUETYPE>& virial,
                        std::vector<VALUETYPE>& atom_energy,
                        std::vector<VALUETYPE>& atom_virial,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const std::vector<VALUETYPE>& fparam,
                        const std::vector<VALUETYPE>& aparam) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  std::vector<VALUETYPE> coord_wrapped = coord;
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same_v<VALUETYPE, float>) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);
  std::vector<torch::jit::IValue> inputs;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, natoms, 3}, options)
          .to(device);
  inputs.push_back(coord_wrapped_Tensor);
  std::vector<int64_t> atype_64(atype.begin(), atype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, natoms}, int_options).to(device);
  inputs.push_back(atype_Tensor);
  c10::optional<torch::Tensor> box_Tensor;
  if (!box.empty()) {
    box_Tensor =
        torch::from_blob(const_cast<VALUETYPE*>(box.data()), {1, 9}, options)
            .to(device);
  }
  inputs.push_back(box_Tensor);
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<long int>(fparam.size())}, options)
            .to(device);
  }
  inputs.push_back(fparam_tensor);
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam.data()),
            {1, natoms, static_cast<long int>(aparam.size()) / natoms}, options)
            .to(device);
  }
  inputs.push_back(aparam_tensor);
  bool do_atom_virial_tensor = true;
  inputs.push_back(do_atom_virial_tensor);
  c10::Dict<c10::IValue, c10::IValue> outputs =
      module.forward(inputs).toGenericDict();
  c10::IValue energy_ = outputs.at("energy");
  c10::IValue force_ = outputs.at("force");
  c10::IValue virial_ = outputs.at("virial");
  c10::IValue atom_virial_ = outputs.at("atom_virial");
  c10::IValue atom_energy_ = outputs.at("atom_energy");
  torch::Tensor flat_energy_ = energy_.toTensor().view({-1});
  torch::Tensor cpu_energy_ = flat_energy_.to(torch::kCPU);
  ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
              cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
  torch::Tensor flat_atom_energy_ =
      atom_energy_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_atom_energy_ = flat_atom_energy_.to(torch::kCPU);
  atom_energy.assign(
      cpu_atom_energy_.data_ptr<VALUETYPE>(),
      cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  force.assign(cpu_force_.data_ptr<VALUETYPE>(),
               cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());
  torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());
  torch::Tensor flat_atom_virial_ =
      atom_virial_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
  atom_virial.assign(
      cpu_atom_virial_.data_ptr<VALUETYPE>(),
      cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
}

template void DeepPotPT::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);
template void DeepPotPT::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);
void DeepPotPT::get_type_map(std::string& type_map) {
  auto ret = module.run_method("get_type_map").toList();
  for (const torch::IValue& element : ret) {
    type_map += torch::str(element);  // Convert each element to a string
    type_map += " ";                  // Add a space between elements
  }
}

// forward to template method
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam);
}
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam);
}
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const int nghost,
                         const InputNlist& inlist,
                         const int& ago,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam);
}
void DeepPotPT::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const int nghost,
                         const InputNlist& inlist,
                         const int& ago,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam);
}
void DeepPotPT::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<double>& force,
                                    std::vector<double>& virial,
                                    std::vector<double>& atom_energy,
                                    std::vector<double>& atom_virial,
                                    const int& nframes,
                                    const std::vector<double>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<double>& box,
                                    const std::vector<double>& fparam,
                                    const std::vector<double>& aparam) {
  throw deepmd::deepmd_exception("computew_mixed_type is not implemented");
}
void DeepPotPT::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<float>& force,
                                    std::vector<float>& virial,
                                    std::vector<float>& atom_energy,
                                    std::vector<float>& atom_virial,
                                    const int& nframes,
                                    const std::vector<float>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<float>& box,
                                    const std::vector<float>& fparam,
                                    const std::vector<float>& aparam) {
  throw deepmd::deepmd_exception("computew_mixed_type is not implemented");
}
#endif
