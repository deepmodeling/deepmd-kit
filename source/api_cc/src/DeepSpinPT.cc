// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PYTORCH
#include "DeepSpinPT.h"

#include <torch/csrc/jit/runtime/jit_exception.h>

#include <cstdint>

#include "common.h"
#include "device.h"
#include "errors.h"

using namespace deepmd;

void DeepSpinPT::translate_error(std::function<void()> f) {
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

torch::Tensor createNlistTensor2(const std::vector<std::vector<int>>& data) {
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
DeepSpinPT::DeepSpinPT() : inited(false) {}
DeepSpinPT::DeepSpinPT(const std::string& model,
                       const int& gpu_rank,
                       const std::string& file_content)
    : inited(false) {
  try {
    translate_error([&] { init(model, gpu_rank, file_content); });
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    throw;
  }
}
void DeepSpinPT::init(const std::string& model,
                      const int& gpu_rank,
                      const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
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
  do_message_passing = module.run_method("has_message_passing").toBool();
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
DeepSpinPT::~DeepSpinPT() {}

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepSpinPT::compute(ENERGYVTYPE& ener,
                         std::vector<VALUETYPE>& force,
                         std::vector<VALUETYPE>& force_mag,
                         std::vector<VALUETYPE>& virial,
                         std::vector<VALUETYPE>& atom_energy,
                         std::vector<VALUETYPE>& atom_virial,
                         const std::vector<VALUETYPE>& coord,
                         const std::vector<VALUETYPE>& spin,
                         const std::vector<int>& atype,
                         const std::vector<VALUETYPE>& box,
                         const int nghost,
                         const InputNlist& lmp_list,
                         const int& ago,
                         const std::vector<VALUETYPE>& fparam,
                         const std::vector<VALUETYPE>& aparam,
                         const bool atomic) {
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
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, dforce_mag, aparam_, datom_energy,
      datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  int nframes = 1;
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  std::vector<VALUETYPE> spin_wrapped = spin;
  at::Tensor spin_wrapped_Tensor =
      torch::from_blob(spin_wrapped.data(), {1, nall_real, 3}, options)
          .to(device);
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option).to(device);
  c10::optional<torch::Tensor> mapping_tensor;
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
    if (do_message_passing) {
      int nswap = lmp_list.nswap;
      torch::Tensor sendproc_tensor =
          torch::from_blob(lmp_list.sendproc, {nswap}, int32_option);
      torch::Tensor recvproc_tensor =
          torch::from_blob(lmp_list.recvproc, {nswap}, int32_option);
      torch::Tensor firstrecv_tensor =
          torch::from_blob(lmp_list.firstrecv, {nswap}, int32_option);
      torch::Tensor recvnum_tensor =
          torch::from_blob(lmp_list.recvnum, {nswap}, int32_option);
      torch::Tensor sendnum_tensor =
          torch::from_blob(lmp_list.sendnum, {nswap}, int32_option);
      torch::Tensor communicator_tensor;
      if (lmp_list.world == 0) {
        communicator_tensor = torch::empty({1}, torch::kInt64);
      } else {
        communicator_tensor = torch::from_blob(
            const_cast<void*>(lmp_list.world), {1}, torch::kInt64);
      }
      torch::Tensor nswap_tensor = torch::tensor(nswap, int32_option);
      int total_send =
          std::accumulate(lmp_list.sendnum, lmp_list.sendnum + nswap, 0);
      torch::Tensor sendlist_tensor =
          torch::from_blob(lmp_list.sendlist, {total_send}, int32_option);
      torch::Tensor has_spin = torch::tensor({1}, int32_option);
      comm_dict.insert("send_list", sendlist_tensor);
      comm_dict.insert("send_proc", sendproc_tensor);
      comm_dict.insert("recv_proc", recvproc_tensor);
      comm_dict.insert("send_num", sendnum_tensor);
      comm_dict.insert("recv_num", recvnum_tensor);
      comm_dict.insert("communicator", communicator_tensor);
      comm_dict.insert("has_spin", has_spin);
    }
  }
  at::Tensor firstneigh = createNlistTensor2(nlist_data.jlist);
  firstneigh_tensor = firstneigh.to(torch::kInt64).to(device);
  bool do_atom_virial_tensor = atomic;
  c10::optional<torch::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())}, options)
            .to(device);
  }
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam_.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam_.data()),
            {1, lmp_list.inum,
             static_cast<std::int64_t>(aparam_.size()) / lmp_list.inum},
            options)
            .to(device);
  }
  c10::Dict<c10::IValue, c10::IValue> outputs =
      (do_message_passing)
          ? module
                .run_method("forward_lower", coord_wrapped_Tensor, atype_Tensor,
                            spin_wrapped_Tensor, firstneigh_tensor,
                            mapping_tensor, fparam_tensor, aparam_tensor,
                            do_atom_virial_tensor, comm_dict)
                .toGenericDict()
          : module
                .run_method("forward_lower", coord_wrapped_Tensor, atype_Tensor,
                            spin_wrapped_Tensor, firstneigh_tensor,
                            mapping_tensor, fparam_tensor, aparam_tensor,
                            do_atom_virial_tensor)
                .toGenericDict();
  c10::IValue energy_ = outputs.at("energy");
  c10::IValue force_ = outputs.at("extended_force");
  c10::IValue force_mag_ = outputs.at("extended_force_mag");
  // spin model not suported yet
  // c10::IValue virial_ = outputs.at("virial");
  torch::Tensor flat_energy_ = energy_.toTensor().view({-1});
  torch::Tensor cpu_energy_ = flat_energy_.to(torch::kCPU);
  ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
              cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  dforce.assign(cpu_force_.data_ptr<VALUETYPE>(),
                cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());
  torch::Tensor flat_force_mag_ =
      force_mag_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_mag_ = flat_force_mag_.to(torch::kCPU);
  dforce_mag.assign(
      cpu_force_mag_.data_ptr<VALUETYPE>(),
      cpu_force_mag_.data_ptr<VALUETYPE>() + cpu_force_mag_.numel());
  // spin model not suported yet
  // torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  // torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  // virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
  //               cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());

  // bkw map
  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  force_mag.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(force_mag, dforce_mag, bkw_map, 3, nframes,
                        fwd_map.size(), nall_real);
  if (atomic) {
    // spin model not suported yet
    // c10::IValue atom_virial_ = outputs.at("extended_virial");
    c10::IValue atom_energy_ = outputs.at("atom_energy");
    torch::Tensor flat_atom_energy_ =
        atom_energy_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = flat_atom_energy_.to(torch::kCPU);
    datom_energy.resize(nall_real,
                        0.0);  // resize to nall to be consistenet with TF.
    datom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());
    // spin model not suported yet
    // torch::Tensor flat_atom_virial_ =
    //     atom_virial_.toTensor().view({-1}).to(floatType);
    // torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
    // datom_virial.assign(
    //     cpu_atom_virial_.data_ptr<VALUETYPE>(),
    //     cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    // atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    // select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
    //                       fwd_map.size(), nall_real);
  }
}
template void DeepSpinPT::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& force_mag,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<double>& spin,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void DeepSpinPT::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& force_mag,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<float>& spin,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepSpinPT::compute(ENERGYVTYPE& ener,
                         std::vector<VALUETYPE>& force,
                         std::vector<VALUETYPE>& force_mag,
                         std::vector<VALUETYPE>& virial,
                         std::vector<VALUETYPE>& atom_energy,
                         std::vector<VALUETYPE>& atom_virial,
                         const std::vector<VALUETYPE>& coord,
                         const std::vector<VALUETYPE>& spin,
                         const std::vector<int>& atype,
                         const std::vector<VALUETYPE>& box,
                         const std::vector<VALUETYPE>& fparam,
                         const std::vector<VALUETYPE>& aparam,
                         const bool atomic) {
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  std::vector<VALUETYPE> coord_wrapped = coord;
  std::vector<VALUETYPE> spin_wrapped = spin;
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    options = torch::TensorOptions().dtype(torch::kFloat32);
    floatType = torch::kFloat32;
  }
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);
  int nframes = 1;
  std::vector<torch::jit::IValue> inputs;
  at::Tensor coord_wrapped_Tensor =
      torch::from_blob(coord_wrapped.data(), {1, natoms, 3}, options)
          .to(device);
  inputs.push_back(coord_wrapped_Tensor);
  std::vector<std::int64_t> atype_64(atype.begin(), atype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, natoms}, int_options).to(device);
  inputs.push_back(atype_Tensor);
  at::Tensor spin_wrapped_Tensor =
      torch::from_blob(spin_wrapped.data(), {1, natoms, 3}, options).to(device);
  inputs.push_back(spin_wrapped_Tensor);
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
                         {1, static_cast<std::int64_t>(fparam.size())}, options)
            .to(device);
  }
  inputs.push_back(fparam_tensor);
  c10::optional<torch::Tensor> aparam_tensor;
  if (!aparam.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam.data()),
            {1, natoms, static_cast<std::int64_t>(aparam.size()) / natoms},
            options)
            .to(device);
  }
  inputs.push_back(aparam_tensor);
  bool do_atom_virial_tensor = atomic;
  inputs.push_back(do_atom_virial_tensor);
  c10::Dict<c10::IValue, c10::IValue> outputs =
      module.forward(inputs).toGenericDict();
  c10::IValue energy_ = outputs.at("energy");
  c10::IValue force_ = outputs.at("force");
  c10::IValue force_mag_ = outputs.at("force_mag");
  // spin model not suported yet
  // c10::IValue virial_ = outputs.at("virial");
  torch::Tensor flat_energy_ = energy_.toTensor().view({-1});
  torch::Tensor cpu_energy_ = flat_energy_.to(torch::kCPU);
  ener.assign(cpu_energy_.data_ptr<ENERGYTYPE>(),
              cpu_energy_.data_ptr<ENERGYTYPE>() + cpu_energy_.numel());
  torch::Tensor flat_force_ = force_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_ = flat_force_.to(torch::kCPU);
  force.assign(cpu_force_.data_ptr<VALUETYPE>(),
               cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());
  torch::Tensor flat_force_mag_ =
      force_mag_.toTensor().view({-1}).to(floatType);
  torch::Tensor cpu_force_mag_ = flat_force_mag_.to(torch::kCPU);
  force_mag.assign(
      cpu_force_mag_.data_ptr<VALUETYPE>(),
      cpu_force_mag_.data_ptr<VALUETYPE>() + cpu_force_mag_.numel());
  // spin model not suported yet
  // torch::Tensor flat_virial_ = virial_.toTensor().view({-1}).to(floatType);
  // torch::Tensor cpu_virial_ = flat_virial_.to(torch::kCPU);
  // virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
  //               cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());
  if (atomic) {
    // c10::IValue atom_virial_ = outputs.at("atom_virial");
    c10::IValue atom_energy_ = outputs.at("atom_energy");
    torch::Tensor flat_atom_energy_ =
        atom_energy_.toTensor().view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = flat_atom_energy_.to(torch::kCPU);
    atom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());
    // torch::Tensor flat_atom_virial_ =
    //     atom_virial_.toTensor().view({-1}).to(floatType);
    // torch::Tensor cpu_atom_virial_ = flat_atom_virial_.to(torch::kCPU);
    // atom_virial.assign(
    //     cpu_atom_virial_.data_ptr<VALUETYPE>(),
    //     cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());
  }
}

template void DeepSpinPT::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& force_mag,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<double>& spin,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void DeepSpinPT::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& force_mag,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<float>& spin,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);
void DeepSpinPT::get_type_map(std::string& type_map) {
  auto ret = module.run_method("get_type_map").toList();
  for (const torch::IValue& element : ret) {
    type_map += torch::str(element);  // Convert each element to a string
    type_map += " ";                  // Add a space between elements
  }
}

// forward to template method
void DeepSpinPT::computew(std::vector<double>& ener,
                          std::vector<double>& force,
                          std::vector<double>& force_mag,
                          std::vector<double>& virial,
                          std::vector<double>& atom_energy,
                          std::vector<double>& atom_virial,
                          const std::vector<double>& coord,
                          const std::vector<double>& spin,
                          const std::vector<int>& atype,
                          const std::vector<double>& box,
                          const std::vector<double>& fparam,
                          const std::vector<double>& aparam,
                          const bool atomic) {
  translate_error([&] {
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, box, fparam, aparam, atomic);
  });
}
void DeepSpinPT::computew(std::vector<double>& ener,
                          std::vector<float>& force,
                          std::vector<float>& force_mag,
                          std::vector<float>& virial,
                          std::vector<float>& atom_energy,
                          std::vector<float>& atom_virial,
                          const std::vector<float>& coord,
                          const std::vector<float>& spin,
                          const std::vector<int>& atype,
                          const std::vector<float>& box,
                          const std::vector<float>& fparam,
                          const std::vector<float>& aparam,
                          const bool atomic) {
  translate_error([&] {
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, box, fparam, aparam, atomic);
  });
}
void DeepSpinPT::computew(std::vector<double>& ener,
                          std::vector<double>& force,
                          std::vector<double>& force_mag,
                          std::vector<double>& virial,
                          std::vector<double>& atom_energy,
                          std::vector<double>& atom_virial,
                          const std::vector<double>& coord,
                          const std::vector<double>& spin,
                          const std::vector<int>& atype,
                          const std::vector<double>& box,
                          const int nghost,
                          const InputNlist& inlist,
                          const int& ago,
                          const std::vector<double>& fparam,
                          const std::vector<double>& aparam,
                          const bool atomic) {
  translate_error([&] {
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, box, nghost, inlist, ago, fparam, aparam, atomic);
  });
}
void DeepSpinPT::computew(std::vector<double>& ener,
                          std::vector<float>& force,
                          std::vector<float>& force_mag,
                          std::vector<float>& virial,
                          std::vector<float>& atom_energy,
                          std::vector<float>& atom_virial,
                          const std::vector<float>& coord,
                          const std::vector<float>& spin,
                          const std::vector<int>& atype,
                          const std::vector<float>& box,
                          const int nghost,
                          const InputNlist& inlist,
                          const int& ago,
                          const std::vector<float>& fparam,
                          const std::vector<float>& aparam,
                          const bool atomic) {
  translate_error([&] {
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, box, nghost, inlist, ago, fparam, aparam, atomic);
  });
}
#endif
