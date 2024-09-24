// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PADDLE
#include "DeepPotPD.h"

#include <cstdint>
#include <stdexcept>
#include <numeric>

#include "AtomMap.h"
#include "device.h"
#include "common.h"
#include "paddle/include/paddle_inference_api.h"

using namespace deepmd;

DeepPotPD::DeepPotPD() : inited(false) {}

DeepPotPD::DeepPotPD(const std::string& model,
                 const int& gpu_rank,
                 const std::string& file_content)
    : inited(false) {
  init(model, gpu_rank, file_content);
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
  // deepmd::load_op_library();
  int gpu_num = 1; // hard code here
  if (gpu_num > 0) {
    gpu_id = gpu_rank % gpu_num;
  } else {
    gpu_id = 0;
  }

  std::string pdmodel_path;
  std::string pdiparams_path;
  bool use_paddle_inference = false;
  bool use_pir = false;

  if (model.find(".json") != std::string::npos) {
    use_pir = true;
    pdmodel_path = model;
    std::string tmp = model;
    pdiparams_path = tmp.replace(model.find(".json"), 5, std::string(".pdiparams"));
    use_paddle_inference = true;
  } else if (model.find(".pdmodel") != std::string::npos){
    pdmodel_path = model;
    std::string tmp = model;
    pdiparams_path = tmp.replace(model.find(".pdmodel"), 8, std::string(".pdiparams"));
    use_paddle_inference = true;
  } else {
    throw "[Error] Not found any inference model in";
  }

  int math_lib_num_threads = 1;

  if (use_paddle_inference) {
    config = std::make_shared<paddle_infer::Config>();
    config->DisableGlogInfo();
    // config->SwitchIrDebug(true);
    if (use_pir) {
      config->EnableNewExecutor(true);
      config->EnableNewIR(true);
    }
    config->SetModel(pdmodel_path, pdiparams_path);
    // config->SwitchIrOptim(true);
    config->EnableUseGpu(8192, 0);
    // config->EnableMKLDNN();
    // config->EnableMemoryOptim();
    // config->EnableProfile();
    predictor = paddle_infer::CreatePredictor(*config);
  }
  rcut = double(6.0);
  ntypes = 2;
  ntypes_spin = 0;
  dfparam = 0;
  daparam = 0;
  aparam_nall = false;

  inited = true;
}

DeepPotPD::~DeepPotPD() {}

template <typename VALUETYPE>
void DeepPotPD::validate_fparam_aparam(
    const int nframes,
    const int& nloc,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam) const {
  if (fparam.size() != dfparam && fparam.size() != nframes * dfparam) {
    throw deepmd::deepmd_exception(
        "the dim of frame parameter provided is not consistent with what the "
        "model uses");
  }

  if (aparam.size() != daparam * nloc &&
      aparam.size() != nframes * daparam * nloc) {
    throw deepmd::deepmd_exception(
        "the dim of atom parameter provided is not consistent with what the "
        "model uses");
  }
}

std::vector<int> createNlistTensor(const std::vector<std::vector<int>>& data) {
  std::vector<int> ret;

  for (const auto& row : data) {
    ret.insert(ret.end(), row.begin(), row.end());
  }

  return ret;
}

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
        "do_message_passing == 1 && nghost == 0"
      );
    }
  }
  std::vector<int> firstneigh = createNlistTensor(nlist_data.jlist);
  auto firstneigh_tensor = predictor->GetInputHandle("nlist");
  firstneigh_tensor->Reshape({1, nloc, (int)firstneigh.size() / (int)nloc});
  firstneigh_tensor->CopyFromCpu(firstneigh.data());
  bool do_atom_virial_tensor = atomic;
  // paddle_infer::Tensor fparam_tensor;
  // if (!fparam.empty()) {
  //   fparam_tensor = predictor->GetInputHandle("fparam");
  //   fparam_tensor->Reshape({1, static_cast<int>(fparam.size())});
  //   fparam_tensor->CopyFromCpu((fparam.data()));
  // }
  // paddle_infer::Tensor aparam_tensor;
  // if (!aparam_.empty()) {
  //   aparam_tensor = predictor->GetInputHandle("aparam");
  //   aparam_tensor->Reshape({1, lmp_list.inum,
  //            static_cast<int>(aparam_.size()) / lmp_list.inum});
  //   aparam_tensor->CopyFromCpu((aparam_.data()));
  // }

  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference failed");
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
    auto atom_virial_ = predictor->GetOutputHandle("extended_virial");
    auto atom_energy_ = predictor->GetOutputHandle("atom_energy");
    datom_energy.resize(nall_real,
                        0.0);  // resize to nall to be consistenet with TF.
    atom_energy_->CopyToCpu(datom_energy.data());
    atom_virial_->CopyToCpu(datom_virial.data());
    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                          fwd_map.size(), nall_real);
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
    fparam_tensor = predictor->GetInputHandle("box");
    fparam_tensor->Reshape({1, static_cast<int>(fparam.size())});
    fparam_tensor->CopyFromCpu((fparam.data()));
  }
  std::unique_ptr<paddle_infer::Tensor> aparam_tensor;
  if (!fparam.empty()) {
    aparam_tensor = predictor->GetInputHandle("box");
    aparam_tensor->Reshape({1, natoms, static_cast<int>(aparam.size()) / natoms});
    aparam_tensor->CopyFromCpu((aparam.data()));
  }

  bool do_atom_virial_tensor = atomic;
  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference failed");
  }

  auto output_names = predictor->GetOutputNames();
  auto energy_ = predictor->GetOutputHandle(output_names[1]);
  auto force_ = predictor->GetOutputHandle(output_names[2]);
  auto virial_ = predictor->GetOutputHandle(output_names[3]);

  energy_->CopyToCpu(ener.data());
  force_->CopyToCpu(force.data());
  virial_->CopyToCpu(virial.data());

  if (atomic) {
    auto atom_energy_ = predictor->GetOutputHandle(output_names[4]);
    auto atom_virial_ = predictor->GetOutputHandle(output_names[5]);
    atom_energy_->CopyToCpu(atom_energy.data());
    atom_virial_->CopyToCpu(atom_virial.data());
  }
}

template void DeepPotPD::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>&dforce,
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

// mixed type

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute_mixed_type(ENERGYVTYPE& dener,
                                   std::vector<VALUETYPE>& dforce_,
                                   std::vector<VALUETYPE>& dvirial,
                                   std::vector<VALUETYPE>& datom_energy_,
                                   std::vector<VALUETYPE>& datom_virial_,
                                   const int& nframes,
                                   const std::vector<VALUETYPE>& dcoord_,
                                   const std::vector<int>& datype_,
                                   const std::vector<VALUETYPE>& dbox,
                                   const std::vector<VALUETYPE>& fparam_,
                                   const std::vector<VALUETYPE>& aparam_,
                                   const bool atomic) {
  // int nloc = datype_.size() / nframes;
  // // here atommap only used to get nloc
  // atommap = deepmd::AtomMap(datype_.begin(), datype_.begin() + nloc);
  // std::vector<VALUETYPE> fparam;
  // std::vector<VALUETYPE> aparam;
  // validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  // tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  // tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  // if (dtype == paddle_infer::DataType::FLOAT64) {
  //   int nloc = predictor_input_tensors_mixed_type<double>(
  //       predictor, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
  //       fparam, aparam, atommap, aparam_nall);
  //   if (atomic) {
  //     run_model<double>(dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor,
  //                       atommap, nframes);
  //   } else {
  //     run_model<double>(dener, dforce_, dvirial, predictor,
  //                       atommap, nframes);
  //   }
  // } else {
  //   int nloc = predictor_input_tensors_mixed_type<double>(
  //       predictor, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
  //       fparam, aparam, atommap, aparam_nall);
  //   if (atomic) {
  //     run_model<float>(dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor,
  //                      atommap, nframes);
  //   } else {
  //     run_model<float>(dener, dforce_, dvirial, predictor, atommap,
  //                      nframes);
  //   }
  // }
}

template void DeepPotPD::compute_mixed_type<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const int& nframes,
    const std::vector<double>& coord,
    const std::vector<int>& dtype,
    const std::vector<double>& box,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepPotPD::compute_mixed_type<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
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
    const bool atomic);


template <class VT>
VT DeepPotPD::get_scalar(const std::string& name) const {
  return predictor_get_scalar<VT>(predictor, name);
}

void DeepPotPD::get_type_map(std::string& type_map) {
  type_map = "O H ";
  // type_map = predictor_get_scalar<std::string>(predictor, "type_map");
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
  compute_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                     coord, atype, box, fparam, aparam, atomic);
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
  compute_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                     coord, atype, box, fparam, aparam, atomic);
}
#endif
