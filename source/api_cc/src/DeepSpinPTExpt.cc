// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepSpinPTExpt.h"

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT_SPIN
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>

#include "SimulationRegion.h"
#include "common.h"
#include "commonPT.h"
#include "commonPTExpt.h"
#include "device.h"
#include "errors.h"
#include "neighbor_list.h"

using deepmd::ptexpt::parse_json;
using deepmd::ptexpt::read_zip_entry;

using namespace deepmd;

void DeepSpinPTExpt::translate_error(std::function<void()> f) {
  try {
    f();
  } catch (const c10::Error& e) {
    throw deepmd::deepmd_exception(
        "DeePMD-kit PyTorch Exportable backend error: " +
        std::string(e.what()));
  } catch (const deepmd::deepmd_exception&) {
    throw;
  } catch (const std::exception& e) {
    throw deepmd::deepmd_exception(
        "DeePMD-kit PyTorch Exportable backend error: " +
        std::string(e.what()));
  }
}

DeepSpinPTExpt::DeepSpinPTExpt() : inited(false) {}

DeepSpinPTExpt::DeepSpinPTExpt(const std::string& model,
                               const int& gpu_rank,
                               const std::string& file_content)
    : inited(false) {
  try {
    translate_error([&] { init(model, gpu_rank, file_content); });
  } catch (...) {
    throw;
  }
}

void DeepSpinPTExpt::init(const std::string& model,
                          const int& gpu_rank,
                          const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }

  // Load libdeepmd_op_pt.so so deepmd_export::* schemas are visible
  // to torch's dispatcher before the AOTI module loads.  See
  // DeepPotPTExpt::init for the full rationale.
  deepmd::load_op_library();

  if (!file_content.empty()) {
    throw deepmd::deepmd_exception(
        "In-memory file_content loading is not supported for .pt2 models. "
        "Please provide a file path instead.");
  }

  int gpu_num = torch::cuda::device_count();
  gpu_id = (gpu_num > 0) ? (gpu_rank % gpu_num) : 0;
  gpu_enabled = torch::cuda::is_available();

  std::string device_str;
  if (!gpu_enabled) {
    device_str = "cpu";
    std::cout << "load model from: " << model << " to cpu" << std::endl;
  } else {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    DPErrcheck(DPSetDevice(gpu_id));
#endif
    device_str = "cuda:" + std::to_string(gpu_id);
    std::cout << "load model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }

  // Read metadata from the .pt2 ZIP archive
  std::string metadata_json = read_zip_entry(model, "extra/metadata.json");

  auto metadata = parse_json(metadata_json);
  rcut = metadata["rcut"].as_double();
  ntypes = static_cast<int>(metadata["type_map"].as_array().size());
  dfparam = metadata["dim_fparam"].as_int();
  daparam = metadata["dim_aparam"].as_int();
  aparam_nall = false;

  // Spin-specific metadata
  if (metadata.obj_val.count("ntypes_spin")) {
    ntypes_spin = metadata["ntypes_spin"].as_int();
  } else {
    ntypes_spin = 0;
  }

  use_spin_.clear();
  if (metadata.obj_val.count("use_spin")) {
    for (const auto& v : metadata["use_spin"].as_array()) {
      use_spin_.push_back(v.as_bool());
    }
  }

  if (metadata.obj_val.count("has_default_fparam")) {
    has_default_fparam_ = metadata["has_default_fparam"].as_bool();
  } else {
    has_default_fparam_ = false;
  }
  if (has_default_fparam_) {
    if (metadata.obj_val.count("default_fparam")) {
      default_fparam_.clear();
      for (const auto& v : metadata["default_fparam"].as_array()) {
        default_fparam_.push_back(v.as_double());
      }
      if (static_cast<int>(default_fparam_.size()) != dfparam) {
        throw deepmd::deepmd_exception(
            "default_fparam length (" + std::to_string(default_fparam_.size()) +
            ") does not match dim_fparam (" + std::to_string(dfparam) + ").");
      }
    } else {
      std::cerr << "WARNING: Model has has_default_fparam=true but "
                   "default_fparam values are missing from metadata."
                << std::endl;
    }
  }

  if (metadata.obj_val.count("do_atomic_virial")) {
    do_atomic_virial = metadata["do_atomic_virial"].as_bool();
  } else {
    // Older models without this field were exported with do_atomic_virial=True
    do_atomic_virial = true;
  }

  // Read expected nnei (= sum(sel)) — the .pt2 graph has this dimension static.
  if (metadata.obj_val.count("nnei")) {
    nnei = metadata["nnei"].as_int();
  } else {
    // Fallback: compute from sel array
    nnei = 0;
    for (const auto& v : metadata["sel"].as_array()) {
      nnei += v.as_int();
    }
  }

  type_map.clear();
  for (const auto& v : metadata["type_map"].as_array()) {
    type_map.push_back(v.as_string());
  }

  output_keys.clear();
  for (const auto& v : metadata["output_keys"].as_array()) {
    output_keys.push_back(v.as_string());
  }

  // Load the AOTInductor model package
  loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
      model, "model", false, 1,
      gpu_enabled ? static_cast<c10::DeviceIndex>(gpu_id)
                  : static_cast<c10::DeviceIndex>(-1));

  // Phase 4: load the optional with-comm artifact for multi-rank GNN
  // spin inference.  Mirrors DeepPotPTExpt; see its init() comment.
  has_comm_artifact_ = metadata.obj_val.count("has_comm_artifact") &&
                       metadata["has_comm_artifact"].as_bool();
  if (has_comm_artifact_) {
    with_comm_tempfile_ = std::make_unique<deepmd::ptexpt::TempFile>(
        deepmd::ptexpt::TempFile::from_zip_entry(
            model, "extra/forward_lower_with_comm.pt2"));
    with_comm_loader =
        std::make_unique<torch::inductor::AOTIModelPackageLoader>(
            with_comm_tempfile_->path(), "model", false, 1,
            gpu_enabled ? static_cast<c10::DeviceIndex>(gpu_id)
                        : static_cast<c10::DeviceIndex>(-1));
  }

  int num_intra_nthreads, num_inter_nthreads;
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

  inited = true;
}

DeepSpinPTExpt::~DeepSpinPTExpt() {}

std::vector<torch::Tensor> DeepSpinPTExpt::run_model(
    const torch::Tensor& coord,
    const torch::Tensor& atype,
    const torch::Tensor& spin,
    const torch::Tensor& nlist,
    const torch::Tensor& mapping,
    const torch::Tensor& fparam,
    const torch::Tensor& aparam) {
  // Spin model has 7 positional args: coord, atype, spin, nlist, mapping,
  // fparam, aparam Only include fparam/aparam if the model was exported with
  // them.
  std::vector<torch::Tensor> inputs = {coord, atype, spin, nlist, mapping};
  if (dfparam > 0) {
    inputs.push_back(fparam);
  }
  if (daparam > 0) {
    inputs.push_back(aparam);
  }
  return loader->run(inputs);
}

std::vector<torch::Tensor> DeepSpinPTExpt::run_model_with_comm(
    const torch::Tensor& coord,
    const torch::Tensor& atype,
    const torch::Tensor& spin,
    const torch::Tensor& nlist,
    const torch::Tensor& mapping,
    const torch::Tensor& fparam,
    const torch::Tensor& aparam,
    const std::vector<at::Tensor>& comm_tensors) {
  if (!with_comm_loader) {
    throw deepmd::deepmd_exception(
        "DeepSpinPTExpt::run_model_with_comm called but the .pt2 has no "
        "with-comm artifact.");
  }
  if (comm_tensors.size() != 8) {
    throw deepmd::deepmd_exception(
        "DeepSpinPTExpt::run_model_with_comm: comm_tensors must contain "
        "exactly 8 tensors. Got " +
        std::to_string(comm_tensors.size()) + ".");
  }
  std::vector<torch::Tensor> inputs = {coord, atype, spin, nlist, mapping};
  if (dfparam > 0) {
    inputs.push_back(fparam);
  }
  if (daparam > 0) {
    inputs.push_back(aparam);
  }
  for (const auto& t : comm_tensors) {
    inputs.push_back(t);
  }
  return with_comm_loader->run(inputs);
}

void DeepSpinPTExpt::extract_outputs(
    std::map<std::string, torch::Tensor>& output_map,
    const std::vector<torch::Tensor>& flat_outputs) {
  if (flat_outputs.size() != output_keys.size()) {
    throw deepmd::deepmd_exception(
        "Model returned " + std::to_string(flat_outputs.size()) +
        " outputs but expected " + std::to_string(output_keys.size()) +
        " (from metadata.json)");
  }
  for (size_t i = 0; i < output_keys.size(); ++i) {
    output_map[output_keys[i]] = flat_outputs[i];
  }
}

// ============================================================================
// LAMMPS path: compute with pre-built neighbor list
// ============================================================================

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepSpinPTExpt::compute(ENERGYVTYPE& ener,
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
  // Fail fast before allocating any tensors.
  if (atomic && !do_atomic_virial) {
    throw deepmd::deepmd_exception(
        "Atomic virial was requested (e.g. by LAMMPS compute */atom/virial) "
        "but this .pt2 model was exported without it (metadata field "
        "do_atomic_virial=False). Atomic virial adds ~2.5x inference cost "
        "and is off by default for .pt2. To enable it, regenerate with: "
        "dp convert-backend --atomic-virial INPUT.pth OUTPUT.pt2");
  }
  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }
  int natoms = atype.size();
  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    floatType = torch::kFloat32;
  }
  auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);

  // Select real atoms (filter NULL-type atoms)
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

  // Build spin tensor for real atoms using bkw_map
  std::vector<VALUETYPE> dspin(static_cast<size_t>(nall_real) * 3);
  for (int ii = 0; ii < nall_real; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dspin[static_cast<size_t>(ii) * 3 + dd] =
          spin[static_cast<size_t>(bkw_map[ii]) * 3 + dd];
    }
  }

  // Convert coord and spin to float64
  std::vector<double> coord_d(dcoord.begin(), dcoord.end());
  std::vector<double> spin_d(dspin.begin(), dspin.end());
  at::Tensor coord_Tensor =
      torch::from_blob(coord_d.data(), {1, nall_real, 3}, options)
          .clone()
          .to(device);
  at::Tensor spin_Tensor =
      torch::from_blob(spin_d.data(), {1, nall_real, 3}, options)
          .clone()
          .to(device);
  std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall_real}, int_option)
          .clone()
          .to(device);

  // LAMMPS sets ago=0 on every nlist rebuild, so ago>0 implies the cached
  // mapping and nlist tensors are still valid — see DeepPotPTExpt.cc for
  // the same rationale.
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();

    // Rebuild mapping tensor
    if (lmp_list.mapping) {
      std::vector<std::int64_t> mapping(nall_real);
      for (int ii = 0; ii < nall_real; ii++) {
        mapping[ii] = fwd_map[lmp_list.mapping[bkw_map[ii]]];
      }
      mapping_tensor =
          torch::from_blob(mapping.data(), {1, nall_real}, int_option)
              .clone()
              .to(device);
    } else {
      std::vector<std::int64_t> mapping(nall_real);
      for (int ii = 0; ii < nall_real; ii++) {
        mapping[ii] = ii;
      }
      mapping_tensor =
          torch::from_blob(mapping.data(), {1, nall_real}, int_option)
              .clone()
              .to(device);
    }

    // Flatten raw nlist — the .pt2 model sorts by distance on-device.
    firstneigh_tensor =
        createNlistTensor(nlist_data.jlist, nnei).to(torch::kInt64).to(device);
  }

  // Build fparam/aparam tensors
  auto valuetype_options = std::is_same<VALUETYPE, float>::value
                               ? torch::TensorOptions().dtype(torch::kFloat32)
                               : torch::TensorOptions().dtype(torch::kFloat64);
  at::Tensor fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())},
                         valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else if (has_default_fparam_ && !default_fparam_.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<double*>(default_fparam_.data()),
                         {1, static_cast<std::int64_t>(default_fparam_.size())},
                         options)
            .clone()
            .to(device);
  } else if (has_default_fparam_) {
    throw deepmd::deepmd_exception(
        "fparam is empty and default_fparam values are missing from the .pt2 "
        "metadata. Please regenerate the model or provide fparam explicitly.");
  } else {
    fparam_tensor = torch::zeros({0}, options).to(device);
  }

  at::Tensor aparam_tensor;
  if (!aparam_.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam_.data()),
            {1, nloc, static_cast<std::int64_t>(aparam_.size()) / nloc},
            valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else {
    aparam_tensor = torch::zeros({0}, options).to(device);
  }

  // Phase 4 dispatch: route to with-comm artifact in multi-rank mode.
  // ``has_spin=tensor([1])`` is baked into the with-comm graph at
  // trace time (Phase 3, spin_model.forward_common_lower_exportable
  // _with_comm), so C++ supplies the same 8 comm tensors as the
  // non-spin path. ``nlocal``/``nghost`` carry the real-atom counts
  // (pre atom-doubling); the spin override halves them internally.
  std::vector<torch::Tensor> flat_outputs;
  bool use_with_comm = has_comm_artifact_ && lmp_list.nswap > 0;
  std::vector<std::vector<int>> remapped_sendlist;
  std::vector<int*> remapped_sendlist_ptrs;
  std::vector<int> remapped_sendnum, remapped_recvnum;
  if (use_with_comm) {
    bool has_null_atoms = (nall_real < nall);
    std::vector<at::Tensor> comm_tensors;
    if (has_null_atoms) {
      comm_tensors =
          deepmd::ptexpt::build_comm_tensors_positional_with_virtual_atoms(
              lmp_list, fwd_map, nloc, nghost_real, remapped_sendlist,
              remapped_sendlist_ptrs, remapped_sendnum, remapped_recvnum);
    } else {
      comm_tensors = deepmd::ptexpt::build_comm_tensors_positional(
          lmp_list, lmp_list.sendlist, lmp_list.sendnum, lmp_list.recvnum, nloc,
          nghost_real);
    }
    flat_outputs = run_model_with_comm(
        coord_Tensor, atype_Tensor, spin_Tensor, firstneigh_tensor,
        mapping_tensor, fparam_tensor, aparam_tensor, comm_tensors);
  } else {
    flat_outputs =
        run_model(coord_Tensor, atype_Tensor, spin_Tensor, firstneigh_tensor,
                  mapping_tensor, fparam_tensor, aparam_tensor);
  }

  std::map<std::string, torch::Tensor> output_map;
  extract_outputs(output_map, flat_outputs);

  // Extract energy
  torch::Tensor flat_energy_ =
      output_map["energy_redu"].view({-1}).to(torch::kCPU);
  ener.assign(flat_energy_.data_ptr<ENERGYTYPE>(),
              flat_energy_.data_ptr<ENERGYTYPE>() + flat_energy_.numel());

  // Extract force: energy_derv_r (nf, nall, 1, 3) -> (nf, nall, 3)
  torch::Tensor force_tensor =
      output_map["energy_derv_r"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_force_ = force_tensor.to(torch::kCPU);
  dforce.assign(cpu_force_.data_ptr<VALUETYPE>(),
                cpu_force_.data_ptr<VALUETYPE>() + cpu_force_.numel());

  // Extract force_mag: energy_derv_r_mag (nf, nall, 1, 3) -> (nf, nall, 3)
  torch::Tensor force_mag_tensor =
      output_map["energy_derv_r_mag"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_force_mag_ = force_mag_tensor.to(torch::kCPU);
  dforce_mag.assign(
      cpu_force_mag_.data_ptr<VALUETYPE>(),
      cpu_force_mag_.data_ptr<VALUETYPE>() + cpu_force_mag_.numel());

  // Extract virial
  torch::Tensor virial_tensor =
      output_map["energy_derv_c_redu"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = virial_tensor.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());

  // bkw map: map force from real atoms back to full atom list
  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  force_mag.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(force_mag, dforce_mag, bkw_map, 3, nframes,
                        fwd_map.size(), nall_real);

  if (atomic) {
    torch::Tensor atom_energy_tensor =
        output_map["energy"].view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = atom_energy_tensor.to(torch::kCPU);
    datom_energy.resize(nall_real, 0.0);
    datom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());

    torch::Tensor atom_virial_tensor =
        output_map["energy_derv_c"].squeeze(-2).view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ = atom_virial_tensor.to(torch::kCPU);
    datom_virial.assign(
        cpu_atom_virial_.data_ptr<VALUETYPE>(),
        cpu_atom_virial_.data_ptr<VALUETYPE>() + cpu_atom_virial_.numel());

    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                          fwd_map.size(), nall_real);
  }
}

template void DeepSpinPTExpt::compute<double, std::vector<ENERGYTYPE>>(
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
template void DeepSpinPTExpt::compute<float, std::vector<ENERGYTYPE>>(
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

// ============================================================================
// Standalone path: compute without pre-built neighbor list
// ============================================================================

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepSpinPTExpt::compute(ENERGYVTYPE& ener,
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
  // Fail fast before allocating any tensors.
  if (atomic && !do_atomic_virial) {
    throw deepmd::deepmd_exception(
        "Atomic virial was requested (e.g. by LAMMPS compute */atom/virial) "
        "but this .pt2 model was exported without it (metadata field "
        "do_atomic_virial=False). Atomic virial adds ~2.5x inference cost "
        "and is off by default for .pt2. To enable it, regenerate with: "
        "dp convert-backend --atomic-virial INPUT.pth OUTPUT.pt2");
  }
  int natoms = atype.size();

  torch::Device device(torch::kCUDA, gpu_id);
  if (!gpu_enabled) {
    device = torch::Device(torch::kCPU);
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat64);
  torch::ScalarType floatType = torch::kFloat64;
  if (std::is_same<VALUETYPE, float>::value) {
    floatType = torch::kFloat32;
  }
  auto int_options = torch::TensorOptions().dtype(torch::kInt64);
  int nframes = 1;

  // 1. Handle box: if empty (NoPbc), create a fake box large enough
  std::vector<double> coord_d(coord.begin(), coord.end());
  std::vector<double> spin_d(spin.begin(), spin.end());
  std::vector<double> box_d(box.begin(), box.end());
  if (box_d.empty()) {
    // Create a fake orthorhombic box that contains all atoms with margin
    double min_x = coord_d[0], max_x = coord_d[0];
    double min_y = coord_d[1], max_y = coord_d[1];
    double min_z = coord_d[2], max_z = coord_d[2];
    for (int ii = 1; ii < natoms; ++ii) {
      min_x = std::min(min_x, coord_d[ii * 3 + 0]);
      max_x = std::max(max_x, coord_d[ii * 3 + 0]);
      min_y = std::min(min_y, coord_d[ii * 3 + 1]);
      max_y = std::max(max_y, coord_d[ii * 3 + 1]);
      min_z = std::min(min_z, coord_d[ii * 3 + 2]);
      max_z = std::max(max_z, coord_d[ii * 3 + 2]);
    }
    // Shift coords so minimum is at rcut (ensures all atoms are in [0, L))
    double shift_x = rcut - min_x;
    double shift_y = rcut - min_y;
    double shift_z = rcut - min_z;
    for (int ii = 0; ii < natoms; ++ii) {
      coord_d[ii * 3 + 0] += shift_x;
      coord_d[ii * 3 + 1] += shift_y;
      coord_d[ii * 3 + 2] += shift_z;
    }
    box_d.resize(9, 0.0);
    box_d[0] = (max_x - min_x) + 2.0 * rcut;
    box_d[4] = (max_y - min_y) + 2.0 * rcut;
    box_d[8] = (max_z - min_z) + 2.0 * rcut;
  }

  // 2. Extend coords with ghosts
  std::vector<double> coord_cpy_d;
  std::vector<int> atype_cpy, mapping_vec;
  std::vector<int> ncell, ngcell;
  {
    SimulationRegion<double> region;
    region.reinitBox(&box_d[0]);
    copy_coord(coord_cpy_d, atype_cpy, mapping_vec, ncell, ngcell, coord_d,
               atype, static_cast<float>(rcut), region);
  }

  int nloc = natoms;
  int nall = coord_cpy_d.size() / 3;

  // 2b. Extend spin to ghost atoms using mapping
  std::vector<double> spin_cpy_d(static_cast<size_t>(nall) * 3, 0.0);
  for (int ii = 0; ii < nloc; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      spin_cpy_d[static_cast<size_t>(ii) * 3 + dd] =
          spin_d[static_cast<size_t>(ii) * 3 + dd];
    }
  }
  for (int ii = nloc; ii < nall; ++ii) {
    int li = mapping_vec[ii];
    for (int dd = 0; dd < 3; ++dd) {
      spin_cpy_d[static_cast<size_t>(ii) * 3 + dd] =
          spin_d[static_cast<size_t>(li) * 3 + dd];
    }
  }

  // 3. Build neighbor list on extended coords
  std::vector<std::vector<int>> nlist_raw, nlist_r_cpy;
  {
    SimulationRegion<double> region;
    region.reinitBox(&box_d[0]);
    std::vector<int> nat_stt(3, 0), ext_stt(3), ext_end(3);
    for (int dd = 0; dd < 3; ++dd) {
      ext_stt[dd] = -ngcell[dd];
      ext_end[dd] = ncell[dd] + ngcell[dd];
    }
    build_nlist(nlist_raw, nlist_r_cpy, coord_cpy_d, nloc, rcut, rcut, nat_stt,
                ncell, ext_stt, ext_end, region, ncell);
  }

  // 4. Convert to tensors
  at::Tensor coord_Tensor =
      torch::from_blob(coord_cpy_d.data(), {1, nall, 3}, options)
          .clone()
          .to(device);
  at::Tensor spin_Tensor =
      torch::from_blob(spin_cpy_d.data(), {1, nall, 3}, options)
          .clone()
          .to(device);
  std::vector<std::int64_t> atype_64(atype_cpy.begin(), atype_cpy.end());
  at::Tensor atype_Tensor =
      torch::from_blob(atype_64.data(), {1, nall}, int_options)
          .clone()
          .to(device);
  // Flatten raw nlist — the .pt2 model sorts by distance on-device.
  at::Tensor nlist_tensor =
      createNlistTensor(nlist_raw, nnei).to(torch::kInt64).to(device);
  std::vector<std::int64_t> mapping_64(mapping_vec.begin(), mapping_vec.end());
  at::Tensor mapping_tensor =
      torch::from_blob(mapping_64.data(), {1, nall}, int_options)
          .clone()
          .to(device);

  // Build fparam/aparam tensors
  auto valuetype_options = std::is_same<VALUETYPE, float>::value
                               ? torch::TensorOptions().dtype(torch::kFloat32)
                               : torch::TensorOptions().dtype(torch::kFloat64);
  at::Tensor fparam_tensor;
  if (!fparam.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<VALUETYPE*>(fparam.data()),
                         {1, static_cast<std::int64_t>(fparam.size())},
                         valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else if (has_default_fparam_ && !default_fparam_.empty()) {
    fparam_tensor =
        torch::from_blob(const_cast<double*>(default_fparam_.data()),
                         {1, static_cast<std::int64_t>(default_fparam_.size())},
                         options)
            .clone()
            .to(device);
  } else if (has_default_fparam_) {
    throw deepmd::deepmd_exception(
        "fparam is empty and default_fparam values are missing from the .pt2 "
        "metadata. Please regenerate the model or provide fparam explicitly.");
  } else {
    fparam_tensor = torch::zeros({0}, options).to(device);
  }

  at::Tensor aparam_tensor;
  if (!aparam.empty()) {
    aparam_tensor =
        torch::from_blob(
            const_cast<VALUETYPE*>(aparam.data()),
            {1, natoms, static_cast<std::int64_t>(aparam.size()) / natoms},
            valuetype_options)
            .to(torch::kFloat64)
            .to(device);
  } else {
    aparam_tensor = torch::zeros({0}, options).to(device);
  }

  // 5. Run the .pt2 model (7 args for spin)
  auto flat_outputs =
      run_model(coord_Tensor, atype_Tensor, spin_Tensor, nlist_tensor,
                mapping_tensor, fparam_tensor, aparam_tensor);

  // 6. Extract outputs
  std::map<std::string, torch::Tensor> output_map;
  extract_outputs(output_map, flat_outputs);

  // 7. Extract energy
  torch::Tensor flat_energy_ =
      output_map["energy_redu"].view({-1}).to(torch::kCPU);
  ener.assign(flat_energy_.data_ptr<ENERGYTYPE>(),
              flat_energy_.data_ptr<ENERGYTYPE>() + flat_energy_.numel());

  // 8. Extract virial
  torch::Tensor virial_tensor =
      output_map["energy_derv_c_redu"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_virial_ = virial_tensor.to(torch::kCPU);
  virial.assign(cpu_virial_.data_ptr<VALUETYPE>(),
                cpu_virial_.data_ptr<VALUETYPE>() + cpu_virial_.numel());

  // 9. Extract force and fold back: energy_derv_r (nf, nall, 1, 3)
  torch::Tensor force_ext =
      output_map["energy_derv_r"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_force_ext = force_ext.to(torch::kCPU);
  std::vector<VALUETYPE> extended_force(
      cpu_force_ext.data_ptr<VALUETYPE>(),
      cpu_force_ext.data_ptr<VALUETYPE>() + cpu_force_ext.numel());
  fold_back(force, extended_force, mapping_vec, nloc, nall, 3, nframes);

  // 10. Extract force_mag and fold back: energy_derv_r_mag (nf, nall, 1, 3)
  torch::Tensor force_mag_ext =
      output_map["energy_derv_r_mag"].squeeze(-2).view({-1}).to(floatType);
  torch::Tensor cpu_force_mag_ext = force_mag_ext.to(torch::kCPU);
  std::vector<VALUETYPE> extended_force_mag(
      cpu_force_mag_ext.data_ptr<VALUETYPE>(),
      cpu_force_mag_ext.data_ptr<VALUETYPE>() + cpu_force_mag_ext.numel());
  fold_back(force_mag, extended_force_mag, mapping_vec, nloc, nall, 3, nframes);

  if (atomic) {
    // atom_energy: energy (nf, nloc, 1)
    torch::Tensor atom_energy_tensor =
        output_map["energy"].view({-1}).to(floatType);
    torch::Tensor cpu_atom_energy_ = atom_energy_tensor.to(torch::kCPU);
    atom_energy.assign(
        cpu_atom_energy_.data_ptr<VALUETYPE>(),
        cpu_atom_energy_.data_ptr<VALUETYPE>() + cpu_atom_energy_.numel());

    // atom_virial: energy_derv_c (nf, nall, 1, 9) -> fold back
    torch::Tensor atom_virial_ext =
        output_map["energy_derv_c"].squeeze(-2).view({-1}).to(floatType);
    torch::Tensor cpu_atom_virial_ext = atom_virial_ext.to(torch::kCPU);
    std::vector<VALUETYPE> extended_atom_virial(
        cpu_atom_virial_ext.data_ptr<VALUETYPE>(),
        cpu_atom_virial_ext.data_ptr<VALUETYPE>() +
            cpu_atom_virial_ext.numel());
    fold_back(atom_virial, extended_atom_virial, mapping_vec, nloc, nall, 9,
              nframes);
  }
}

template void DeepSpinPTExpt::compute<double, std::vector<ENERGYTYPE>>(
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
template void DeepSpinPTExpt::compute<float, std::vector<ENERGYTYPE>>(
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

void DeepSpinPTExpt::get_type_map(std::string& type_map_str) {
  for (const auto& t : type_map) {
    type_map_str += t;
    type_map_str += " ";
  }
}

// forward to template method
void DeepSpinPTExpt::computew(std::vector<double>& ener,
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
void DeepSpinPTExpt::computew(std::vector<double>& ener,
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
void DeepSpinPTExpt::computew(std::vector<double>& ener,
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
void DeepSpinPTExpt::computew(std::vector<double>& ener,
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
