// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PADDLE
#include "DeepPotPD.h"

#include <cstdint>
#include <numeric>

#include "common.h"
#include "device.h"
#include "errors.h"

using namespace deepmd;

std::vector<int> createNlistTensor(const std::vector<std::vector<int>>& data) {
  std::vector<int> ret;
  for (const auto& row : data) {
    ret.insert(ret.end(), row.begin(), row.end());
  }
  return ret;
}

DeepPotPD::DeepPotPD() : inited(false) {}
DeepPotPD::DeepPotPD(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content)
    : inited(false) {
  try {
    init(model, gpu_rank, file_content);
  } catch (...) {
    throw;
  }
}
void DeepPotPD::init(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  // NOTE: There is no custom operators need to be loaded now.
  // deepmd::load_op_library();

  // NOTE: Only support 1 GPU now.
  int gpu_num = 1;
  if (gpu_num > 0) {
    gpu_id = gpu_rank % gpu_num;
  } else {
    gpu_id = 0;
  }

  // initialize inference config
  config = std::make_shared<paddle_infer::Config>();
  config->DisableGlogInfo();
  config->EnableNewExecutor(true);
  config->EnableNewIR(true);

  // loading inference model
  std::string pdmodel_path;
  std::string pdiparams_path;
  if (model.find(".json") != std::string::npos) {
    pdmodel_path = model;
    pdiparams_path = model;
    pdiparams_path.replace(pdiparams_path.find(".json"), 5,
                           std::string(".pdiparams"));
  } else if (model.find(".pdmodel") != std::string::npos) {
    pdmodel_path = model;
    pdiparams_path = model;
    pdiparams_path.replace(pdiparams_path.find(".pdmodel"), 8,
                           std::string(".pdiparams"));
  } else {
    throw deepmd::deepmd_exception("Given inference model: " + model +
                                   " do not exist, please check it.");
  }
  config->SetModel(pdmodel_path, pdiparams_path);
  config->EnableUseGpu(
      4096, 0);  // annotate it if use cpu, default use gpu with 4G mem
  gpu_enabled = config->use_gpu();
  if (!gpu_enabled) {
    config->DisableGpu();
    std::cout << "load model from: " << model << " to cpu " << std::endl;
  } else {
    std::cout << "load model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }

  // NOTE: Both set to 1 now.
  // get_env_nthreads(num_intra_nthreads,
  //                  num_inter_nthreads);  // need to be fixed as
  //                                        // DP_INTRA_OP_PARALLELISM_THREADS
  // num_intra_nthreads = 1;
  num_inter_nthreads = 1;
  if (num_inter_nthreads) {
    config->SetCpuMathLibraryNumThreads(num_inter_nthreads);
  }

  predictor = paddle_infer::CreatePredictor(*config);

  // initialize hyper params from model buffers
  ntypes_spin = 0;
  DeepPotPD::get_buffer<int>("buffer_has_message_passing", do_message_passing);
  DeepPotPD::get_buffer<double>("buffer_rcut", rcut);
  DeepPotPD::get_buffer<int>("buffer_ntypes", ntypes);
  DeepPotPD::get_buffer<int>("buffer_dfparam", dfparam);
  DeepPotPD::get_buffer<int>("buffer_daparam", daparam);
  DeepPotPD::get_buffer<int>("buffer_aparam_nall", aparam_nall);
  inited = true;
}
DeepPotPD::~DeepPotPD() {}

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute(ENERGYVTYPE& ener,
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
                        const std::vector<VALUETYPE>& aparam,
                        const bool atomic) {
  int natoms = atype.size();
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam_, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  int nframes = 1;
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  auto coord_wrapped_Tensor = predictor->GetInputHandle("coord");
  coord_wrapped_Tensor->Reshape({1, nall_real, 3});
  coord_wrapped_Tensor->CopyFromCpu(coord_wrapped.data());

  auto atype_Tensor = predictor->GetInputHandle("atype");
  atype_Tensor->Reshape({1, nall_real});
  atype_Tensor->CopyFromCpu(datype.data());

  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
    if (do_message_passing == 1 && nghost > 0) {
      throw deepmd::deepmd_exception(
          "(do_message_passing == 1 && nghost > 0) is not supported yet.");
      int nswap = lmp_list.nswap;
      auto sendproc_tensor = predictor->GetInputHandle("sendproc");
      sendproc_tensor->Reshape({nswap});
      sendproc_tensor->CopyFromCpu(lmp_list.sendproc);
      auto recvproc_tensor = predictor->GetInputHandle("recvproc");
      recvproc_tensor->Reshape({nswap});
      recvproc_tensor->CopyFromCpu(lmp_list.recvproc);
      auto firstrecv_tensor = predictor->GetInputHandle("firstrecv");
      firstrecv_tensor->Reshape({nswap});
      firstrecv_tensor->CopyFromCpu(lmp_list.firstrecv);
      auto recvnum_tensor = predictor->GetInputHandle("recvnum");
      recvnum_tensor->Reshape({nswap});
      recvnum_tensor->CopyFromCpu(lmp_list.recvnum);
      auto sendnum_tensor = predictor->GetInputHandle("sendnum");
      sendnum_tensor->Reshape({nswap});
      sendnum_tensor->CopyFromCpu(lmp_list.sendnum);
      auto communicator_tensor = predictor->GetInputHandle("communicator");
      communicator_tensor->Reshape({1});
      communicator_tensor->CopyFromCpu(static_cast<int*>(lmp_list.world));
      auto sendlist_tensor = predictor->GetInputHandle("sendlist");

      int total_send =
          std::accumulate(lmp_list.sendnum, lmp_list.sendnum + nswap, 0);
    }
    if (do_message_passing == 1 && nghost == 0) {
      throw deepmd::deepmd_exception(
          "(do_message_passing == 1 && nghost == 0) is not supported yet.");
    }
  }
  std::vector<int> firstneigh = createNlistTensor(nlist_data.jlist);
  firstneigh_tensor = predictor->GetInputHandle("nlist");
  firstneigh_tensor->Reshape({1, nloc, (int)firstneigh.size() / (int)nloc});
  firstneigh_tensor->CopyFromCpu(firstneigh.data());
  bool do_atom_virial_tensor = atomic;
  std::unique_ptr<paddle_infer::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    throw deepmd::deepmd_exception("fparam is not supported as input yet.");
    // fparam_tensor = predictor->GetInputHandle("fparam");
    // fparam_tensor->Reshape({1, static_cast<int>(fparam.size())});
    // fparam_tensor->CopyFromCpu((fparam.data()));
  }
  std::unique_ptr<paddle_infer::Tensor> aparam_tensor;
  if (!aparam_.empty()) {
    throw deepmd::deepmd_exception("aparam is not supported as input yet.");
    // aparam_tensor = predictor->GetInputHandle("aparam");
    // aparam_tensor->Reshape({1, lmp_list.inum,
    //          static_cast<int>(aparam_.size()) / lmp_list.inum});
    // aparam_tensor->CopyFromCpu((aparam_.data()));
  }

  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference run failed");
  }
  auto output_names = predictor->GetOutputNames();

  auto energy_ = predictor->GetOutputHandle(output_names[1]);
  auto force_ = predictor->GetOutputHandle(output_names[2]);
  auto virial_ = predictor->GetOutputHandle(output_names[3]);
  std::vector<int> output_energy_shape = energy_->shape();
  int output_energy_size =
      std::accumulate(output_energy_shape.begin(), output_energy_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_force_shape = force_->shape();
  int output_force_size =
      std::accumulate(output_force_shape.begin(), output_force_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_virial_shape = virial_->shape();
  int output_virial_size =
      std::accumulate(output_virial_shape.begin(), output_virial_shape.end(), 1,
                      std::multiplies<int>());
  // output energy
  ener.resize(output_energy_size);
  energy_->CopyToCpu(ener.data());

  // output force
  dforce.resize(output_force_size);
  force_->CopyToCpu(dforce.data());

  // output virial
  virial.resize(output_virial_size);
  virial_->CopyToCpu(virial.data());

  // bkw map
  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  if (atomic) {
    throw "atomic virial is not supported as output yet.";
    // auto atom_virial_ = predictor->GetOutputHandle("extended_virial");
    // auto atom_energy_ = predictor->GetOutputHandle("atom_energy");
    // datom_energy.resize(nall_real,
    //                     0.0);  // resize to nall to be consistenet with TF.
    // atom_energy_->CopyToCpu(datom_energy.data());
    // atom_virial_->CopyToCpu(datom_virial.data());
    // atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    // atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    // select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
    //                       fwd_map.size(), nall_real);
    // select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
    //                       fwd_map.size(), nall_real);
  }
}
template void DeepPotPD::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
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
    const std::vector<double>& aparam_,
    const bool atomic);

template void DeepPotPD::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
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
    const std::vector<float>& aparam_,
    const bool atomic);

// ENERGYVTYPE: std::vector<ENERGYTYPE> or ENERGYTYPE
template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute(ENERGYVTYPE& ener,
                        std::vector<VALUETYPE>& force,
                        std::vector<VALUETYPE>& virial,
                        std::vector<VALUETYPE>& atom_energy,
                        std::vector<VALUETYPE>& atom_virial,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const std::vector<VALUETYPE>& fparam,
                        const std::vector<VALUETYPE>& aparam,
                        const bool atomic) {
  // select real atoms
  std::vector<VALUETYPE> coord_wrapped = coord;
  int natoms = atype.size();
  int nframes = 1;
  auto coord_wrapped_Tensor = predictor->GetInputHandle("coord");
  coord_wrapped_Tensor->Reshape({1, natoms, 3});
  coord_wrapped_Tensor->CopyFromCpu(coord_wrapped.data());

  std::vector<std::int64_t> atype_64(atype.begin(), atype.end());
  auto atype_Tensor = predictor->GetInputHandle("atype");
  atype_Tensor->Reshape({1, natoms});
  atype_Tensor->CopyFromCpu(atype_64.data());

  std::unique_ptr<paddle_infer::Tensor> box_Tensor;
  if (!box.empty()) {
    box_Tensor = predictor->GetInputHandle("box");
    box_Tensor->Reshape({1, 9});
    box_Tensor->CopyFromCpu((box.data()));
  }
  std::unique_ptr<paddle_infer::Tensor> fparam_tensor;
  if (!fparam.empty()) {
    throw deepmd::deepmd_exception("fparam is not supported as input yet.");
    // fparam_tensor = predictor->GetInputHandle("box");
    // fparam_tensor->Reshape({1, static_cast<int>(fparam.size())});
    // fparam_tensor->CopyFromCpu((fparam.data()));
  }
  std::unique_ptr<paddle_infer::Tensor> aparam_tensor;
  if (!aparam.empty()) {
    throw deepmd::deepmd_exception("fparam is not supported as input yet.");
    // aparam_tensor = predictor->GetInputHandle("box");
    // aparam_tensor->Reshape({1, natoms, static_cast<int>(aparam.size()) /
    // natoms}); aparam_tensor->CopyFromCpu((aparam.data()));
  }

  bool do_atom_virial_tensor = atomic;
  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference run failed");
  }

  auto output_names = predictor->GetOutputNames();
  auto energy_ = predictor->GetOutputHandle(output_names[1]);
  auto force_ = predictor->GetOutputHandle(output_names[2]);
  auto virial_ = predictor->GetOutputHandle(output_names[3]);

  energy_->CopyToCpu(ener.data());
  force_->CopyToCpu(force.data());
  virial_->CopyToCpu(virial.data());

  if (atomic) {
    throw deepmd::deepmd_exception(
        "atomic virial is not supported as output yet.");
    // auto atom_energy_ = predictor->GetOutputHandle(output_names[4]);
    // auto atom_virial_ = predictor->GetOutputHandle(output_names[5]);
    // atom_energy_->CopyToCpu(atom_energy.data());
    // atom_virial_->CopyToCpu(atom_virial.data());
  }
}

template void DeepPotPD::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& dforce,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& dcoord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepPotPD::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& dcoord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

/* type_map is regarded as a special string buffer
that need to be postprocessed */
void DeepPotPD::get_type_map(std::string& type_map) {
  auto type_map_tensor = predictor->GetOutputHandle("buffer_type_map");
  auto type_map_shape = type_map_tensor->shape();
  int type_map_size = std::accumulate(
      type_map_shape.begin(), type_map_shape.end(), 1, std::multiplies<int>());

  std::vector<int> type_map_arr(type_map_size, 0);
  type_map_tensor->CopyToCpu(type_map_arr.data());
  for (auto char_c : type_map_arr) {
    type_map += std::string(1, char_c);
  }
}

/* general function except for string buffer */
template <typename BUFFERTYPE>
void DeepPotPD::get_buffer(const std::string& buffer_name,
                           std::vector<BUFFERTYPE>& buffer_array) {
  auto buffer_tensor = predictor->GetOutputHandle(buffer_name);
  auto buffer_shape = buffer_tensor->shape();
  int buffer_size = std::accumulate(buffer_shape.begin(), buffer_shape.end(), 1,
                                    std::multiplies<int>());
  buffer_array.resize(buffer_size);
  buffer_tensor->CopyToCpu(buffer_array.data());
}

template <typename BUFFERTYPE>
void DeepPotPD::get_buffer(const std::string& buffer_name,
                           BUFFERTYPE& buffer_scalar) {
  std::vector<BUFFERTYPE> buffer_array(1);
  DeepPotPD::get_buffer<BUFFERTYPE>(buffer_name, buffer_array);
  buffer_scalar = buffer_array[0];
}

// forward to template method
void DeepPotPD::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam, atomic);
}
void DeepPotPD::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam, atomic);
}
void DeepPotPD::computew(std::vector<double>& ener,
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
                         const std::vector<double>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam, atomic);
}
void DeepPotPD::computew(std::vector<double>& ener,
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
                         const std::vector<float>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam, atomic);
}
void DeepPotPD::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<double>& force,
                                    std::vector<double>& virial,
                                    std::vector<double>& atom_energy,
                                    std::vector<double>& atom_virial,
                                    const int& nframes,
                                    const std::vector<double>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<double>& box,
                                    const std::vector<double>& fparam,
                                    const std::vector<double>& aparam,
                                    const bool atomic) {
  throw deepmd::deepmd_exception(
      "computew_mixed_type is not implemented in paddle backend yet");
}
void DeepPotPD::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<float>& force,
                                    std::vector<float>& virial,
                                    std::vector<float>& atom_energy,
                                    std::vector<float>& atom_virial,
                                    const int& nframes,
                                    const std::vector<float>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<float>& box,
                                    const std::vector<float>& fparam,
                                    const std::vector<float>& aparam,
                                    const bool atomic) {
  throw deepmd::deepmd_exception(
      "computew_mixed_type is not implemented in paddle backend yet");
}
#endif
