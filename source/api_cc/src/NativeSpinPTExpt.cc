// SPDX-License-Identifier: LGPL-3.0-or-later
#include "NativeSpinPTExpt.h"

#if defined(BUILD_PYTORCH) && BUILD_PT_EXPT_NATIVE_SPIN
#include <c10/core/DeviceGuard.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "SimulationRegion.h"
#include "common.h"
#include "commonPT.h"
#include "commonPTExpt.h"
#include "device.h"
#include "errors.h"
#include "neighbor_list.h"

using deepmd::ptexpt::check_call_charge_spin;
using deepmd::ptexpt::parse_json;
using deepmd::ptexpt::read_default_chg_spin;
using deepmd::ptexpt::read_zip_entry;

using namespace deepmd;

namespace {

void synchronize_current_accelerator_stream() {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  DPErrcheck(gpuDeviceSynchronize());
#else
  throw deepmd::deepmd_exception(
      "GPU-resident inference requires a GPU-enabled DeePMD-kit build.");
#endif
}

/**
 * @brief Reject conditioning inputs the loaded artifact has no slot for.
 *
 * A declared width of zero means the traced lower carries no such slot, so
 * silently dropping a value the caller supplied would hide a configuration
 * error rather than surface it.
 */
template <typename VALUETYPE>
void reject_unsupported_parametric_inputs(const std::vector<VALUETYPE>& fparam,
                                          const std::vector<VALUETYPE>& aparam,
                                          const int dim_fparam,
                                          const int dim_aparam) {
  if (dim_fparam == 0 && !fparam.empty()) {
    throw deepmd::deepmd_exception(
        "a frame parameter was supplied, but this native-spin artifact "
        "declares dim_fparam=0 and its forward has no slot for one.");
  }
  if (dim_aparam == 0 && !aparam.empty()) {
    throw deepmd::deepmd_exception(
        "an atomic parameter was supplied, but this native-spin artifact "
        "declares dim_aparam=0 and its forward has no slot for one.");
  }
}

/**
 * @brief Build the frame-parameter input of the conditional graph tail.
 *
 * The artifact consumes it in double precision, shaped ``(1, dim_fparam)``.
 * A model that carries a default supplies it whenever the caller passes none,
 * which is how LAMMPS drives such a model. A zero width means the forward has
 * no such slot, so the returned tensor is undefined and never marshalled.
 */
torch::Tensor make_fparam_tensor(const std::vector<double>& fparam,
                                 const std::vector<double>& default_fparam,
                                 const int dim_fparam,
                                 const torch::Device& device) {
  if (dim_fparam == 0) {
    return torch::Tensor();
  }
  const std::vector<double>& values = fparam.empty() ? default_fparam : fparam;
  if (static_cast<int>(values.size()) != dim_fparam) {
    throw deepmd::deepmd_exception(
        "fparam holds " + std::to_string(values.size()) +
        " values but the model expects dim_fparam=" +
        std::to_string(dim_fparam) +
        "; provide it explicitly or freeze the model with a default.");
  }
  return torch::from_blob(const_cast<double*>(values.data()), {1, dim_fparam},
                          torch::TensorOptions().dtype(torch::kFloat64))
      .clone()
      .to(device);
}

/**
 * @brief Build the atomic-parameter input of the conditional graph tail.
 *
 * The graph ABI carries the atomic parameter flat on the node axis. The
 * caller supplies the owned rows; an extended-region graph has its halo rows
 * zero-padded, which the owned-node mask makes inert. A zero width means the
 * forward has no such slot, so the returned tensor is undefined and never
 * marshalled.
 */
torch::Tensor make_aparam_tensor(const std::vector<double>& aparam,
                                 const int dim_aparam,
                                 const std::int64_t node_count,
                                 const std::int64_t nloc,
                                 const torch::Device& device) {
  if (dim_aparam == 0) {
    return torch::Tensor();
  }
  const auto f64_options = torch::TensorOptions().dtype(torch::kFloat64);
  const at::Tensor owned =
      aparam.empty()
          ? torch::zeros({0}, f64_options).to(device)
          : torch::from_blob(const_cast<double*>(aparam.data()),
                             {static_cast<std::int64_t>(aparam.size())},
                             f64_options)
                .clone()
                .to(device);
  return deepmd::extend_graph_aparam(owned, node_count, nloc, dim_aparam);
}

/**
 * @brief Build the charge/spin input of the conditional graph tail.
 *
 * The artifact consumes the condition in double precision, shaped
 * ``(1, dim_chg_spin)``. A zero width means the forward has no such slot --
 * either because the model carries no condition at all, or because
 * compression moved it into frozen tables -- so the returned tensor is
 * undefined and never marshalled.
 */
torch::Tensor make_chg_spin_tensor(const std::vector<double>& charge_spin,
                                   const int dim_chg_spin,
                                   const torch::Device& device) {
  if (dim_chg_spin == 0) {
    return torch::Tensor();
  }
  if (static_cast<int>(charge_spin.size()) != dim_chg_spin) {
    throw deepmd::deepmd_exception(
        "the charge/spin condition holds " +
        std::to_string(charge_spin.size()) +
        " values but the model expects dim_chg_spin=" +
        std::to_string(dim_chg_spin) + ".");
  }
  return torch::from_blob(const_cast<double*>(charge_spin.data()),
                          {1, dim_chg_spin},
                          torch::TensorOptions().dtype(torch::kFloat64))
      .clone()
      .to(device);
}

}  // namespace

void NativeSpinPTExpt::translate_error(std::function<void()> f) {
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

NativeSpinPTExpt::NativeSpinPTExpt() : inited(false) {}

NativeSpinPTExpt::NativeSpinPTExpt(const std::string& model,
                                   const int& gpu_rank,
                                   const std::string& file_content)
    : inited(false) {
  translate_error([&] { init(model, gpu_rank, file_content); });
}

NativeSpinPTExpt::~NativeSpinPTExpt() {}

void NativeSpinPTExpt::init(const std::string& model,
                            const int& gpu_rank,
                            const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }

  // Register the deepmd_export::* schemas with torch's dispatcher before the
  // AOTI module resolves the operators its compiled graph calls into.
  deepmd::load_op_library(deepmd::DPBackend::PyTorchExportable);

  if (!file_content.empty()) {
    throw deepmd::deepmd_exception(
        "In-memory file_content loading is not supported for .pt2 models. "
        "Please provide a file path instead.");
  }

  const int gpu_num = torch::cuda::device_count();
  gpu_id = (gpu_num > 0) ? (gpu_rank % gpu_num) : 0;
  gpu_enabled = torch::cuda::is_available();
  if (!gpu_enabled) {
    std::cout << "load model from: " << model << " to cpu" << std::endl;
  } else {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    DPErrcheck(DPSetDevice(gpu_id));
#endif
    std::cout << "load model from: " << model << " to gpu " << gpu_id
              << std::endl;
  }

  const auto metadata =
      parse_json(read_zip_entry(model, "extra/metadata.json"));

  // The lower kind selects an input schema, not a backend: both native-spin
  // schemas are served here, and every other kind belongs to a different
  // scheme or a different model class.
  const std::string lower_input_kind =
      metadata.obj_val.count("lower_input_kind")
          ? metadata["lower_input_kind"].as_string()
          : std::string();
  if (lower_input_kind == "graph") {
    canonical_abi_ = false;
  } else if (lower_input_kind == "dpa4c_canonical") {
    canonical_abi_ = true;
  } else if (lower_input_kind == "dpa1_canonical") {
    throw deepmd::deepmd_exception(
        "the dpa1_canonical compact artifact has no moment slot; freeze a "
        "native-spin model as 'graph' or 'dpa4c_canonical'.");
  } else {
    throw deepmd::deepmd_exception(
        "a native-spin artifact must declare lower_input_kind 'graph' or "
        "'dpa4c_canonical', but this archive declares '" +
        lower_input_kind + "'.");
  }

  const int declared_fparam = metadata["dim_fparam"].as_int();
  const int declared_aparam = metadata["dim_aparam"].as_int();
  const int declared_chg_spin = metadata.obj_val.count("dim_chg_spin")
                                    ? metadata["dim_chg_spin"].as_int()
                                    : 0;
  dfparam = declared_fparam;
  daparam = declared_aparam;
  dchgspin = declared_chg_spin;
  // A model whose lower reads the condition as an input accepts exactly that
  // condition; the charge-state fold, loaded below, widens this for a
  // compressed model, whose lower has no conditioning input at all.
  settable_chgspin = dchgspin;
  default_chg_spin_ = read_default_chg_spin(metadata, dchgspin);
  has_default_fparam_ = metadata.obj_val.count("has_default_fparam") &&
                        metadata["has_default_fparam"].as_bool();
  default_fparam_.clear();
  if (has_default_fparam_ && metadata.obj_val.count("default_fparam")) {
    for (const auto& v : metadata["default_fparam"].as_array()) {
      default_fparam_.push_back(v.as_double());
    }
  }

  graph_edge_fp32_ = metadata.obj_val.count("graph_edge_dtype") &&
                     metadata["graph_edge_dtype"].as_string() == "float32";
  if (canonical_abi_) {
    if (!graph_edge_fp32_) {
      throw deepmd::deepmd_exception(
          "compact canonical graph artifacts require float32 edge vectors.");
    }
    if (!metadata.obj_val.count("canonical_index_dtype") ||
        metadata["canonical_index_dtype"].as_string() != "uint32") {
      throw deepmd::deepmd_exception(
          "compact canonical graph artifacts require uint32 topology; "
          "re-freeze the model with the current DeePMD-kit version.");
    }
    // The compact lower is traced with the nine graph and moment inputs
    // alone, so a model declaring any conditioning width has no slot to
    // receive it.
    if (dfparam > 0 || daparam > 0) {
      throw deepmd::deepmd_exception(
          "the compact canonical native-spin ABI has no fparam / aparam slot, "
          "but this model declares dim_fparam=" +
          std::to_string(dfparam) + ", dim_aparam=" + std::to_string(daparam) +
          "; freeze it with the graph lower instead.");
    }
  }
  // The per-atom virial is a structural part of both contracts: the
  // device-resident entry point returns it unconditionally and the host paths
  // reduce it into the global virial.
  if (!metadata.obj_val.count("do_atomic_virial") ||
      !metadata["do_atomic_virial"].as_bool()) {
    throw deepmd::deepmd_exception(
        "native-spin graph artifacts must be exported with the per-atom "
        "virial.");
  }
  has_message_passing_ = metadata.obj_val.count("has_message_passing") &&
                         metadata["has_message_passing"].as_bool();

  rcut = metadata["rcut"].as_double();
  ntypes = metadata.obj_val.count("ntypes")
               ? metadata["ntypes"].as_int()
               : static_cast<int>(metadata["type_map"].as_array().size());
  ntypes_spin = metadata.obj_val.count("ntypes_spin")
                    ? metadata["ntypes_spin"].as_int()
                    : 0;

  use_spin_.clear();
  if (metadata.obj_val.count("use_spin")) {
    for (const auto& v : metadata["use_spin"].as_array()) {
      use_spin_.push_back(v.as_bool());
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

  loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
      model, "model", false, 1,
      gpu_enabled ? static_cast<c10::DeviceIndex>(gpu_id)
                  : static_cast<c10::DeviceIndex>(-1));

  // Model-level pair-type exclusion keeps its table on the model device for
  // the lifetime of the backend, so the per-step graph build indexes it
  // without a host round trip.
  {
    std::vector<std::pair<int, int>> pair_exclude_types;
    if (metadata.obj_val.count("pair_exclude_types")) {
      for (const auto& v : metadata["pair_exclude_types"].as_array()) {
        pair_exclude_types.emplace_back(v[0].as_int(), v[1].as_int());
      }
    }
    std::vector<int> table =
        deepmd::buildPairExcludeTable(ntypes, pair_exclude_types);
    if (!table.empty()) {
      const torch::Device device = gpu_enabled
                                       ? torch::Device(torch::kCUDA, gpu_id)
                                       : torch::Device(torch::kCPU);
      pair_exclude_table_ =
          torch::from_blob(table.data(),
                           {static_cast<std::int64_t>(table.size())},
                           torch::TensorOptions().dtype(torch::kInt32))
              .clone()
              .to(device);
    }
  }

  // Charge-state fold.  Unlike the with-comm artifact, a failure here is not
  // a degraded mode to defer: the metadata field is the archive's claim that
  // the fold ships with it, so an archive that declares the constants but
  // cannot supply the fold is malformed.
  charge_state_fold_ = deepmd::ptexpt::ChargeStateFold::load(
      model, metadata, gpu_enabled, gpu_id);
  if (charge_state_fold_) {
    // The condition of a compressed model reaches the compiled lower only
    // through the constants the fold rebuilds, so the argument list carries
    // none and ``dchgspin`` is zero.  What the model accepts is the state the
    // snapshot was frozen against, which is also the layout the fold consumes.
    settable_chgspin =
        metadata.obj_val.count("default_chg_spin")
            ? static_cast<int>(metadata["default_chg_spin"].as_array().size())
            : 0;
    if (settable_chgspin == 0) {
      throw deepmd::deepmd_exception(
          "the archive ships a charge-state fold but names no "
          "default_chg_spin, so the width of a charge state is unknown");
    }
    // The constants were frozen against the archive's own charge state, so
    // that is the state in force until ``set_charge_spin`` installs another.
    default_chg_spin_ = read_default_chg_spin(metadata, settable_chgspin);
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

void NativeSpinPTExpt::set_charge_spin(const std::vector<double>& charge_spin) {
  assert(inited);
  if (settable_chgspin == 0) {
    throw deepmd::deepmd_exception(
        "this model was not frozen with a charge/spin condition");
  }
  if (static_cast<int>(charge_spin.size()) != settable_chgspin) {
    throw deepmd::deepmd_exception("the charge/spin condition carries " +
                                   std::to_string(charge_spin.size()) +
                                   " values but the model expects " +
                                   std::to_string(settable_chgspin));
  }
  // Route one: the condition of every later forward pass that is not given
  // one explicitly.  This is the whole mechanism for an uncompressed model,
  // which reads the condition as an ordinary input.
  default_chg_spin_ = charge_spin;
  // Route two: a compressed descriptor has folded the condition into frozen
  // tables that the lower holds as constants, so serving another condition
  // means rebuilding those tables and writing them over the constants.
  if (charge_state_fold_) {
    charge_state_fold_->apply(charge_spin,
                              gpu_enabled ? torch::Device(torch::kCUDA, gpu_id)
                                          : torch::Device(torch::kCPU),
                              *loader);
  }
}

void NativeSpinPTExpt::get_type_map(std::string& type_map_str) {
  type_map_str.clear();
  for (const auto& t : type_map) {
    if (!type_map_str.empty()) {
      type_map_str += " ";
    }
    type_map_str += t;
  }
}

std::vector<torch::Tensor> NativeSpinPTExpt::run_model_canonical(
    const CanonicalGraphTensorPack& graph, const torch::Tensor& spin) {
  // The moment shares the float32 precision of the compact geometry; the cast
  // is a no-op for a caller that already holds a float32 tensor.
  return loader->run({graph.atype, graph.n_node, graph.n_local, graph.source,
                      graph.edge_vec, graph.destination_row_ptr,
                      graph.source_row_ptr, graph.source_order,
                      spin.to(torch::kFloat32)});
}

std::vector<torch::Tensor> NativeSpinPTExpt::run_model_graph(
    const GraphTensorPack& graph,
    const torch::Tensor& spin,
    const torch::Tensor& fparam,
    const torch::Tensor& aparam,
    const torch::Tensor& charge_spin) {
  deepmd::check_graph_aparam_flat(aparam, daparam,
                                  "NativeSpinPTExpt::run_model_graph");
  std::vector<torch::Tensor> inputs = {
      graph.atype,
      graph.n_node,
      graph.n_local,
      graph.edge_index,
      graph_edge_fp32_ ? graph.edge_vec.to(torch::kFloat32) : graph.edge_vec,
      graph.edge_mask,
      graph.destination_order,
      graph.destination_row_ptr,
      graph.source_order,
      graph.source_row_ptr,
      spin};
  if (dfparam > 0) {
    inputs.push_back(fparam);
  }
  if (daparam > 0) {
    inputs.push_back(aparam);
  }
  if (dchgspin > 0) {
    inputs.push_back(charge_spin);
  }
  return loader->run(inputs);
}

std::map<std::string, torch::Tensor> NativeSpinPTExpt::run_graph_payload(
    GraphTensorPack& graph,
    const std::int64_t node_count,
    const std::int64_t nloc,
    const torch::Tensor& spin,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam) {
  graph.edge_mask =
      deepmd::applyPairExclusion(graph.edge_index, graph.edge_mask, graph.atype,
                                 pair_exclude_table_, ntypes);
  canonicalizeGraphPayload(graph, node_count);
  std::map<std::string, torch::Tensor> output_map;
  if (canonical_abi_) {
    extract_outputs(output_map,
                    run_model_canonical(compactCanonicalGraph(graph), spin));
    // The compact lower reports the per-atom virial with the fitting axis
    // still in place; the graph lower already drops it.
    deepmd::flatten_canonical_atom_virial(output_map);
  } else {
    const torch::Device device = graph.atype.device();
    extract_outputs(
        output_map,
        run_model_graph(
            graph, spin,
            make_fparam_tensor(fparam, default_fparam_, dfparam, device),
            make_aparam_tensor(aparam, daparam, node_count, nloc, device),
            make_chg_spin_tensor(default_chg_spin_, dchgspin, device)));
  }
  return output_map;
}

void NativeSpinPTExpt::extract_outputs(
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
// LAMMPS path: compute with a pre-built neighbor list
// ============================================================================

template <typename VALUETYPE, typename ENERGYVTYPE>
void NativeSpinPTExpt::compute(ENERGYVTYPE& ener,
                               std::vector<VALUETYPE>& force,
                               std::vector<VALUETYPE>& force_mag,
                               std::vector<VALUETYPE>& virial,
                               std::vector<VALUETYPE>& atom_energy,
                               std::vector<VALUETYPE>& atom_virial,
                               const std::vector<VALUETYPE>& coord,
                               const std::vector<VALUETYPE>& spin,
                               const std::vector<int>& atype,
                               const int nghost,
                               const InputNlist& lmp_list,
                               const int& ago,
                               const std::vector<VALUETYPE>& fparam,
                               const std::vector<VALUETYPE>& aparam,
                               const bool atomic) {
  const torch::Device device = gpu_enabled ? torch::Device(torch::kCUDA, gpu_id)
                                           : torch::Device(torch::kCPU);
  const auto f64_options = torch::TensorOptions().dtype(torch::kFloat64);
  const auto int_option =
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
  const torch::ScalarType float_type =
      std::is_same<VALUETYPE, float>::value ? torch::kFloat32 : torch::kFloat64;
  const int nall = static_cast<int>(atype.size());
  const int nframes = 1;

  // Drop the atoms whose LAMMPS type maps to NULL: the model never sees them,
  // and select_map scatters the results back onto the full atom list.
  std::vector<VALUETYPE> dcoord, dforce, dforce_mag, aparam_real, datom_energy,
      datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam_real, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, nframes, daparam, nall,
                          /*aparam_nall=*/false);
  const int nloc = nall_real - nghost_real;

  // Domain decomposition keeps the extended local-plus-ghost node set so that
  // ghost force and magnetic force rows survive for the reverse
  // communication; a single rank folds ghosts onto their local owners, which
  // needs the LAMMPS atom map to resolve an owner.
  const bool multi_rank = (lmp_list.nprocs > 1);
  if (!multi_rank && nghost > 0 && lmp_list.mapping == nullptr) {
    throw deepmd::deepmd_exception(
        "single-rank inference folds ghost neighbours onto their local owners "
        "through the LAMMPS atom map; add 'atom_modify map yes' to the input, "
        "or populate InputNlist.mapping before calling compute().");
  }
  // A ghost node of the extended graph carries no owner on this rank, so a
  // descriptor that reads intermediate features of neighbouring nodes has no
  // source for them and would silently evaluate a truncated environment. The
  // backend factory normally keeps such an archive with ``DeepSpinPTExpt``,
  // which owns the with-comm route, so this guard stands for a caller that
  // constructs this class directly.
  if (multi_rank && has_message_passing_) {
    throw deepmd::deepmd_exception(
        "this native-spin artifact reads intermediate features of "
        "neighbouring nodes, which domain decomposition cannot supply for a "
        "ghost node; run it on a single MPI rank.");
  }

  if (nall_real == 0) {
    // A rank holding neither a real local atom nor a real ghost contributes
    // nothing, while the exported graph requires at least one node.
    ener.assign(nframes, static_cast<ENERGYTYPE>(0));
    force.assign(static_cast<size_t>(nframes) * fwd_map.size() * 3,
                 static_cast<VALUETYPE>(0));
    force_mag.assign(static_cast<size_t>(nframes) * fwd_map.size() * 3,
                     static_cast<VALUETYPE>(0));
    virial.assign(static_cast<size_t>(nframes) * 9, static_cast<VALUETYPE>(0));
    if (atomic) {
      atom_energy.assign(static_cast<size_t>(nframes) * fwd_map.size(),
                         static_cast<VALUETYPE>(0));
      atom_virial.assign(static_cast<size_t>(nframes) * fwd_map.size() * 9,
                         static_cast<VALUETYPE>(0));
    }
    return;
  }

  const std::vector<double> coord_d(dcoord.begin(), dcoord.end());
  std::vector<double> spin_d(static_cast<size_t>(nall_real) * 3, 0.0);
  for (int ii = 0; ii < nall_real; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      spin_d[static_cast<size_t>(ii) * 3 + dd] =
          static_cast<double>(spin[static_cast<size_t>(bkw_map[ii]) * 3 + dd]);
    }
  }
  const at::Tensor coord_Tensor =
      torch::from_blob(const_cast<double*>(coord_d.data()), {nall_real, 3},
                       f64_options)
          .clone()
          .to(device);
  const at::Tensor spin_Tensor =
      torch::from_blob(spin_d.data(), {nall_real, 3}, f64_options)
          .clone()
          .to(device);
  const std::vector<std::int64_t> atype_64(datype.begin(), datype.end());
  const at::Tensor atype_Tensor =
      torch::from_blob(const_cast<std::int64_t*>(atype_64.data()), {nall_real},
                       int_option)
          .clone()
          .to(device);

  // LAMMPS sets ago == 0 on every neighbor-list rebuild, so a positive ago
  // means the cached skin topology is still valid. The model-cutoff edge set
  // is recomputed from it on-device every step.
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    mapping_.resize(nall_real);
    if (lmp_list.mapping) {
      for (int ii = 0; ii < nall_real; ++ii) {
        mapping_[ii] = fwd_map[lmp_list.mapping[bkw_map[ii]]];
      }
    } else {
      for (int ii = 0; ii < nall_real; ++ii) {
        mapping_[ii] = ii;
      }
    }
    const EdgeTensorPack topology = createEdgeTensors(
        nlist_data.jlist, dcoord, mapping_, nloc, nall_real, device,
        /*with_geometry=*/false, /*row_centers=*/&nlist_data.ilist,
        /*fold_to_local=*/!multi_rank);
    edge_index_tensor = topology.edge_index;
    edge_index_ext_tensor = topology.edge_index_ext;
  }

  const std::int64_t node_count = multi_rank ? nall_real : nloc;
  const EdgeTensorPack edges =
      compactEdgeTensors(edge_index_tensor, edge_index_ext_tensor, coord_Tensor,
                         static_cast<double>(rcut));
  GraphTensorPack graph;
  graph.atype = atype_Tensor.slice(0, 0, node_count);
  graph.n_node = torch::full({1}, node_count, int_option).to(device);
  graph.n_local =
      torch::full({1}, static_cast<std::int64_t>(nloc), int_option).to(device);
  graph.edge_index = edges.edge_index;
  graph.edge_vec = edges.edge_vec;
  graph.edge_mask = edges.edge_mask;
  std::map<std::string, torch::Tensor> output_map = run_graph_payload(
      graph, node_count, nloc, spin_Tensor.slice(0, 0, node_count),
      std::vector<double>(fparam.begin(), fparam.end()),
      std::vector<double>(aparam_real.begin(), aparam_real.end()));

  // The forward emits flat per-node public keys; rewrite them into the dense
  // internal-key layout the extraction below reads. The extended node set
  // already carries one row per extended atom and must not be padded.
  if (multi_rank) {
    deepmd::remap_graph_spin_outputs_to_dense_keys_extended(output_map, nloc,
                                                            nall_real, atomic);
  } else {
    deepmd::remap_graph_spin_outputs_to_dense_keys(output_map, nloc, nall_real,
                                                   atomic);
  }

  const torch::Tensor cpu_energy =
      output_map["energy_redu"].view({-1}).to(torch::kCPU);
  ener.assign(cpu_energy.data_ptr<ENERGYTYPE>(),
              cpu_energy.data_ptr<ENERGYTYPE>() + cpu_energy.numel());

  const torch::Tensor cpu_force = output_map["energy_derv_r"]
                                      .squeeze(-2)
                                      .view({-1})
                                      .to(float_type)
                                      .to(torch::kCPU);
  dforce.assign(cpu_force.data_ptr<VALUETYPE>(),
                cpu_force.data_ptr<VALUETYPE>() + cpu_force.numel());
  const torch::Tensor cpu_force_mag = output_map["energy_derv_r_mag"]
                                          .squeeze(-2)
                                          .view({-1})
                                          .to(float_type)
                                          .to(torch::kCPU);
  dforce_mag.assign(
      cpu_force_mag.data_ptr<VALUETYPE>(),
      cpu_force_mag.data_ptr<VALUETYPE>() + cpu_force_mag.numel());
  const torch::Tensor cpu_virial = output_map["energy_derv_c_redu"]
                                       .squeeze(-2)
                                       .view({-1})
                                       .to(float_type)
                                       .to(torch::kCPU);
  virial.assign(cpu_virial.data_ptr<VALUETYPE>(),
                cpu_virial.data_ptr<VALUETYPE>() + cpu_virial.numel());

  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  force_mag.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(force_mag, dforce_mag, bkw_map, 3, nframes,
                        fwd_map.size(), nall_real);

  if (atomic) {
    const torch::Tensor cpu_atom_energy =
        output_map["energy"].view({-1}).to(float_type).to(torch::kCPU);
    datom_energy.assign(
        cpu_atom_energy.data_ptr<VALUETYPE>(),
        cpu_atom_energy.data_ptr<VALUETYPE>() + cpu_atom_energy.numel());
    const torch::Tensor cpu_atom_virial = output_map["energy_derv_c"]
                                              .squeeze(-2)
                                              .view({-1})
                                              .to(float_type)
                                              .to(torch::kCPU);
    datom_virial.assign(
        cpu_atom_virial.data_ptr<VALUETYPE>(),
        cpu_atom_virial.data_ptr<VALUETYPE>() + cpu_atom_virial.numel());

    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                          fwd_map.size(), nall_real);
  }
}

template void NativeSpinPTExpt::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& force_mag,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<double>& spin,
    const std::vector<int>& atype,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void NativeSpinPTExpt::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& force_mag,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<float>& spin,
    const std::vector<int>& atype,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

// ============================================================================
// Standalone path: compute without a pre-built neighbor list
// ============================================================================

template <typename VALUETYPE, typename ENERGYVTYPE>
void NativeSpinPTExpt::compute(ENERGYVTYPE& ener,
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
  const torch::Device device = gpu_enabled ? torch::Device(torch::kCUDA, gpu_id)
                                           : torch::Device(torch::kCPU);
  const auto f64_options = torch::TensorOptions().dtype(torch::kFloat64);
  const torch::ScalarType float_type =
      std::is_same<VALUETYPE, float>::value ? torch::kFloat32 : torch::kFloat64;
  const int nloc = static_cast<int>(atype.size());
  const int nframes = 1;

  // === Step 1. Supply a box when the caller has none ===
  // An isolated cluster is embedded in an orthorhombic cell wide enough that
  // no atom sees a periodic image of another.
  std::vector<double> coord_d(coord.begin(), coord.end());
  const std::vector<double> spin_d(spin.begin(), spin.end());
  std::vector<double> box_d(box.begin(), box.end());
  if (box_d.empty()) {
    double min_x = coord_d[0], max_x = coord_d[0];
    double min_y = coord_d[1], max_y = coord_d[1];
    double min_z = coord_d[2], max_z = coord_d[2];
    for (int ii = 1; ii < nloc; ++ii) {
      min_x = std::min(min_x, coord_d[ii * 3 + 0]);
      max_x = std::max(max_x, coord_d[ii * 3 + 0]);
      min_y = std::min(min_y, coord_d[ii * 3 + 1]);
      max_y = std::max(max_y, coord_d[ii * 3 + 1]);
      min_z = std::min(min_z, coord_d[ii * 3 + 2]);
      max_z = std::max(max_z, coord_d[ii * 3 + 2]);
    }
    for (int ii = 0; ii < nloc; ++ii) {
      coord_d[ii * 3 + 0] += rcut - min_x;
      coord_d[ii * 3 + 1] += rcut - min_y;
      coord_d[ii * 3 + 2] += rcut - min_z;
    }
    box_d.assign(9, 0.0);
    box_d[0] = (max_x - min_x) + 2.0 * rcut;
    box_d[4] = (max_y - min_y) + 2.0 * rcut;
    box_d[8] = (max_z - min_z) + 2.0 * rcut;
  }

  // === Step 2. Extend with ghosts and build the neighbor list ===
  std::vector<double> coord_cpy_d;
  std::vector<int> atype_cpy, mapping_vec, ncell, ngcell;
  {
    SimulationRegion<double> region;
    region.reinitBox(&box_d[0]);
    copy_coord(coord_cpy_d, atype_cpy, mapping_vec, ncell, ngcell, coord_d,
               atype, static_cast<float>(rcut), region);
  }
  const int nall = static_cast<int>(coord_cpy_d.size()) / 3;

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

  // === Step 3. Run the forward on the folded node set ===
  // build_nlist keys row i to center i and already cuts at rcut, so the graph
  // needs no row remapping. The path is single-rank by construction: ghosts
  // fold onto their local owners and the graph carries nloc nodes.
  const std::vector<std::int64_t> mapping_64(mapping_vec.begin(),
                                             mapping_vec.end());
  GraphTensorPack graph =
      buildGraphTensors(nlist_raw, coord_cpy_d, atype_cpy, mapping_64, nloc,
                        nall, static_cast<double>(rcut), device);
  const at::Tensor spin_Tensor =
      torch::from_blob(const_cast<double*>(spin_d.data()), {nloc, 3},
                       f64_options)
          .clone()
          .to(device);
  std::map<std::string, torch::Tensor> output_map =
      run_graph_payload(graph, nloc, nloc, spin_Tensor,
                        std::vector<double>(fparam.begin(), fparam.end()),
                        std::vector<double>(aparam.begin(), aparam.end()));
  deepmd::remap_graph_spin_outputs_to_dense_keys(output_map, nloc, nall,
                                                 atomic);

  // === Step 4. Read the outputs and fold ghost rows onto their owners ===
  const torch::Tensor cpu_energy =
      output_map["energy_redu"].view({-1}).to(torch::kCPU);
  ener.assign(cpu_energy.data_ptr<ENERGYTYPE>(),
              cpu_energy.data_ptr<ENERGYTYPE>() + cpu_energy.numel());

  const torch::Tensor cpu_virial = output_map["energy_derv_c_redu"]
                                       .squeeze(-2)
                                       .view({-1})
                                       .to(float_type)
                                       .to(torch::kCPU);
  virial.assign(cpu_virial.data_ptr<VALUETYPE>(),
                cpu_virial.data_ptr<VALUETYPE>() + cpu_virial.numel());

  const torch::Tensor cpu_force = output_map["energy_derv_r"]
                                      .squeeze(-2)
                                      .view({-1})
                                      .to(float_type)
                                      .to(torch::kCPU);
  const std::vector<VALUETYPE> extended_force(
      cpu_force.data_ptr<VALUETYPE>(),
      cpu_force.data_ptr<VALUETYPE>() + cpu_force.numel());
  fold_back(force, extended_force, mapping_vec, nloc, nall, 3, nframes);

  const torch::Tensor cpu_force_mag = output_map["energy_derv_r_mag"]
                                          .squeeze(-2)
                                          .view({-1})
                                          .to(float_type)
                                          .to(torch::kCPU);
  const std::vector<VALUETYPE> extended_force_mag(
      cpu_force_mag.data_ptr<VALUETYPE>(),
      cpu_force_mag.data_ptr<VALUETYPE>() + cpu_force_mag.numel());
  fold_back(force_mag, extended_force_mag, mapping_vec, nloc, nall, 3, nframes);

  if (atomic) {
    const torch::Tensor cpu_atom_energy =
        output_map["energy"].view({-1}).to(float_type).to(torch::kCPU);
    atom_energy.assign(
        cpu_atom_energy.data_ptr<VALUETYPE>(),
        cpu_atom_energy.data_ptr<VALUETYPE>() + cpu_atom_energy.numel());

    const torch::Tensor cpu_atom_virial = output_map["energy_derv_c"]
                                              .squeeze(-2)
                                              .view({-1})
                                              .to(float_type)
                                              .to(torch::kCPU);
    const std::vector<VALUETYPE> extended_atom_virial(
        cpu_atom_virial.data_ptr<VALUETYPE>(),
        cpu_atom_virial.data_ptr<VALUETYPE>() + cpu_atom_virial.numel());
    fold_back(atom_virial, extended_atom_virial, mapping_vec, nloc, nall, 9,
              nframes);
  }
}

template void NativeSpinPTExpt::compute<double, std::vector<ENERGYTYPE>>(
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
template void NativeSpinPTExpt::compute<float, std::vector<ENERGYTYPE>>(
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

// ============================================================================
// Public wrappers
// ============================================================================

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
    reject_unsupported_parametric_inputs(fparam, aparam, dfparam, daparam);
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, box, fparam, aparam, atomic);
  });
}

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
    reject_unsupported_parametric_inputs(fparam, aparam, dfparam, daparam);
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, box, fparam, aparam, atomic);
  });
}

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
    reject_unsupported_parametric_inputs(fparam, aparam, dfparam, daparam);
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, nghost, inlist, ago, fparam, aparam, atomic);
  });
}

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
    reject_unsupported_parametric_inputs(fparam, aparam, dfparam, daparam);
    compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
            spin, atype, nghost, inlist, ago, fparam, aparam, atomic);
  });
}

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
                                const std::vector<double>& charge_spin,
                                const bool atomic) {
  check_call_charge_spin(charge_spin, 1, settable_chgspin,
                         /*applied_per_call=*/false, default_chg_spin_);
  computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
           spin, atype, box, fparam, aparam, atomic);
}

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
                                const std::vector<double>& charge_spin,
                                const bool atomic) {
  check_call_charge_spin(charge_spin, 1, settable_chgspin,
                         /*applied_per_call=*/false, default_chg_spin_);
  computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
           spin, atype, box, fparam, aparam, atomic);
}

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
                                const std::vector<double>& charge_spin,
                                const bool atomic) {
  check_call_charge_spin(charge_spin, 1, settable_chgspin,
                         /*applied_per_call=*/false, default_chg_spin_);
  computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
           spin, atype, box, nghost, inlist, ago, fparam, aparam, atomic);
}

void NativeSpinPTExpt::computew(std::vector<double>& ener,
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
                                const std::vector<double>& charge_spin,
                                const bool atomic) {
  check_call_charge_spin(charge_spin, 1, settable_chgspin,
                         /*applied_per_call=*/false, default_chg_spin_);
  computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
           spin, atype, box, nghost, inlist, ago, fparam, aparam, atomic);
}

void NativeSpinPTExpt::compute_canonical_graph_gpu(
    double* d_atom_energy,
    double* d_force,
    double* d_force_mag,
    double* d_atom_virial,
    const std::int64_t* d_atype,
    const std::uint32_t* d_source,
    const float* d_edge_vec,
    const std::int64_t* d_destination_row_ptr,
    const std::int64_t* d_source_row_ptr,
    const std::uint32_t* d_source_order,
    const float* d_spin,
    const int nloc,
    const int nall_nodes,
    const std::int64_t edge_storage) {
  if (!canonical_abi_) {
    throw deepmd::deepmd_exception(
        "device-resident inference consumes the compact canonical graph; this "
        "archive declares the NeighborGraph lower, which the host entry "
        "points serve.");
  }
  if (!gpu_enabled) {
    throw deepmd::deepmd_exception(
        "compute_canonical_graph_gpu requires a CUDA device.");
  }
  if (nloc < 0 || nall_nodes <= 0 || nloc > nall_nodes || edge_storage < 2 ||
      static_cast<std::uint64_t>(edge_storage) >
          std::numeric_limits<std::uint32_t>::max()) {
    throw deepmd::deepmd_exception(
        "invalid compact canonical graph dimensions.");
  }

  translate_error([&] {
    const torch::Device device(torch::kCUDA, gpu_id);
    const c10::DeviceGuard device_guard(device);
    const auto opt_f32 =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const auto opt_f64 =
        torch::TensorOptions().dtype(torch::kFloat64).device(device);
    const auto opt_i64 =
        torch::TensorOptions().dtype(torch::kInt64).device(device);
    const auto opt_u32 =
        torch::TensorOptions().dtype(torch::kUInt32).device(device);
    CanonicalGraphTensorPack graph;
    graph.atype = torch::from_blob(const_cast<std::int64_t*>(d_atype),
                                   {nall_nodes}, opt_i64);
    graph.n_node = torch::full({1}, nall_nodes, opt_i64);
    graph.n_local = torch::full({1}, nloc, opt_i64);
    graph.source = torch::from_blob(const_cast<std::uint32_t*>(d_source),
                                    {edge_storage}, opt_u32);
    graph.edge_vec = torch::from_blob(const_cast<float*>(d_edge_vec),
                                      {edge_storage, 3}, opt_f32);
    graph.destination_row_ptr =
        torch::from_blob(const_cast<std::int64_t*>(d_destination_row_ptr),
                         {nall_nodes + 1}, opt_i64);
    graph.source_row_ptr = torch::from_blob(
        const_cast<std::int64_t*>(d_source_row_ptr), {nall_nodes + 1}, opt_i64);
    graph.source_order = torch::from_blob(
        const_cast<std::uint32_t*>(d_source_order), {edge_storage}, opt_u32);
    const auto spin =
        torch::from_blob(const_cast<float*>(d_spin), {nall_nodes, 3}, opt_f32);

    std::map<std::string, torch::Tensor> output;
    extract_outputs(output, run_model_canonical(graph, spin));
    auto atom_energy = output["atom_energy"]
                           .reshape({nall_nodes})
                           .slice(0, 0, nloc)
                           .contiguous();
    auto force = output["force"].reshape({nall_nodes, 3}).contiguous();
    auto force_mag = output["force_mag"].reshape({nall_nodes, 3}).contiguous();
    auto atom_virial =
        output["atom_virial"].reshape({nall_nodes, 9}).contiguous();
    if (nloc > 0) {
      torch::from_blob(d_atom_energy, {nloc}, opt_f64).copy_(atom_energy);
    }
    torch::from_blob(d_force, {nall_nodes, 3}, opt_f64).copy_(force);
    torch::from_blob(d_force_mag, {nall_nodes, 3}, opt_f64).copy_(force_mag);
    torch::from_blob(d_atom_virial, {nall_nodes, 9}, opt_f64)
        .copy_(atom_virial);
    synchronize_current_accelerator_stream();
  });
}

bool NativeSpinPTExpt::uses_canonical_graph_inference() const {
  return canonical_abi_;
}

// The archive reaches this class only through the scheme dispatch, which
// admits nothing but ``spin_scheme == "native"``; both schemas served here
// belong to that scheme.
bool NativeSpinPTExpt::uses_native_spin_scheme() const { return true; }

#endif
