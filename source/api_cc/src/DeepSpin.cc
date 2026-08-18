// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepSpin.h"

#include <memory>
#include <stdexcept>

#include "AtomMap.h"
#include "BackendPlugin.h"
#include "common.h"
#include "device.h"

using namespace deepmd;

DeepSpin::DeepSpin() { inited = false; }

DeepSpin::DeepSpin(const std::string& model,
                   const int& gpu_rank,
                   const std::string& file_content) {
  inited = false;
  init(model, gpu_rank, file_content);
}

DeepSpin::~DeepSpin() {}

void DeepSpin::init(const std::string& model,
                    const int& gpu_rank,
                    const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  const DPBackend backend = get_backend(model);
  if (deepmd::DPBackend::TensorFlow == backend ||
      deepmd::DPBackend::PyTorch == backend ||
      deepmd::DPBackend::PyTorchExportable == backend) {
    dp = create_deepspin_backend_from_plugin(backend, model, gpu_rank,
                                             file_content);
  } else if (deepmd::DPBackend::Paddle == backend) {
    throw deepmd::deepmd_exception("PaddlePaddle backend is not supported yet");
  } else if (deepmd::DPBackend::JAX == backend) {
    throw deepmd::deepmd_exception("JAX backend is not supported yet");
  } else {
    throw deepmd::deepmd_exception("Unknown file type");
  }
  inited = true;
  dpbase = dp;  // make sure the base funtions work
}

// support spin
// no nlist, no atomic : nframe
template <typename VALUETYPE>
void DeepSpin::compute(ENERGYTYPE& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam_,
                       const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               charge_spin, false);
  dener = dener_[0];
}

template <typename VALUETYPE>
void DeepSpin::compute(std::vector<ENERGYTYPE>& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam_,
                       const std::vector<double>& charge_spin) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               charge_spin, false);
}

// no nlist, no atomic : nframe * precision
template void DeepSpin::compute<double>(ENERGYTYPE& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(ENERGYTYPE& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam,
                                       const std::vector<double>& charge_spin);

template void DeepSpin::compute<double>(std::vector<ENERGYTYPE>& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam,
                                       const std::vector<double>& charge_spin);

// support spin
// nlist, no atomic : nframe
template <typename VALUETYPE>
void DeepSpin::compute(ENERGYTYPE& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const int nghost,
                       const InputNlist& lmp_list,
                       const int& ago,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam__,
                       const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, charge_spin, false);
  dener = dener_[0];
}

template <typename VALUETYPE>
void DeepSpin::compute(std::vector<ENERGYTYPE>& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const int nghost,
                       const InputNlist& lmp_list,
                       const int& ago,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam__,
                       const std::vector<double>& charge_spin) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, charge_spin, false);
}

// nlist, no atomic : nframe * precision
template void DeepSpin::compute<double>(ENERGYTYPE& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const int nghost,
                                        const InputNlist& lmp_list,
                                        const int& ago,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam_,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(ENERGYTYPE& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam_,
                                       const std::vector<double>& charge_spin);

template void DeepSpin::compute<double>(std::vector<ENERGYTYPE>& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const int nghost,
                                        const InputNlist& lmp_list,
                                        const int& ago,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam_,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam_,
                                       const std::vector<double>& charge_spin);

// support spin
// no nlist, atomic : nframe
template <typename VALUETYPE>
void DeepSpin::compute(ENERGYTYPE& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       std::vector<VALUETYPE>& datom_energy_,
                       std::vector<VALUETYPE>& datom_virial_,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam_,
                       const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               charge_spin, true);
  dener = dener_[0];
}
template <typename VALUETYPE>
void DeepSpin::compute(std::vector<ENERGYTYPE>& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       std::vector<VALUETYPE>& datom_energy_,
                       std::vector<VALUETYPE>& datom_virial_,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam_,
                       const std::vector<double>& charge_spin) {
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               charge_spin, true);
}
// no nlist, atomic : nframe * precision
template void DeepSpin::compute<double>(ENERGYTYPE& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        std::vector<double>& datom_energy_,
                                        std::vector<double>& datom_virial_,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(ENERGYTYPE& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       std::vector<float>& datom_energy_,
                                       std::vector<float>& datom_virial_,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam,
                                       const std::vector<double>& charge_spin);

template void DeepSpin::compute<double>(std::vector<ENERGYTYPE>& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        std::vector<double>& datom_energy_,
                                        std::vector<double>& datom_virial_,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       std::vector<float>& datom_energy_,
                                       std::vector<float>& datom_virial_,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam,
                                       const std::vector<double>& charge_spin);

// support spin
// nlist, atomic : nframe
template <typename VALUETYPE>
void DeepSpin::compute(ENERGYTYPE& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       std::vector<VALUETYPE>& datom_energy_,
                       std::vector<VALUETYPE>& datom_virial_,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const int nghost,
                       const InputNlist& lmp_list,
                       const int& ago,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam__,
                       const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, charge_spin, true);
  dener = dener_[0];
}
template <typename VALUETYPE>
void DeepSpin::compute(std::vector<ENERGYTYPE>& dener,
                       std::vector<VALUETYPE>& dforce_,
                       std::vector<VALUETYPE>& dforce_mag_,
                       std::vector<VALUETYPE>& dvirial,
                       std::vector<VALUETYPE>& datom_energy_,
                       std::vector<VALUETYPE>& datom_virial_,
                       const std::vector<VALUETYPE>& dcoord_,
                       const std::vector<VALUETYPE>& dspin_,
                       const std::vector<int>& datype_,
                       const std::vector<VALUETYPE>& dbox,
                       const int nghost,
                       const InputNlist& lmp_list,
                       const int& ago,
                       const std::vector<VALUETYPE>& fparam_,
                       const std::vector<VALUETYPE>& aparam__,
                       const std::vector<double>& charge_spin) {
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, charge_spin, true);
}
// nlist, atomic : nframe * precision
template void DeepSpin::compute<double>(ENERGYTYPE& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        std::vector<double>& datom_energy_,
                                        std::vector<double>& datom_virial_,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const int nghost,
                                        const InputNlist& lmp_list,
                                        const int& ago,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam_,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(ENERGYTYPE& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       std::vector<float>& datom_energy_,
                                       std::vector<float>& datom_virial_,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam_,
                                       const std::vector<double>& charge_spin);

template void DeepSpin::compute<double>(std::vector<ENERGYTYPE>& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        std::vector<double>& datom_energy_,
                                        std::vector<double>& datom_virial_,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const int nghost,
                                        const InputNlist& lmp_list,
                                        const int& ago,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam_,
                                        const std::vector<double>& charge_spin);

template void DeepSpin::compute<float>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       std::vector<float>& datom_energy_,
                                       std::vector<float>& datom_virial_,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam_,
                                       const std::vector<double>& charge_spin);

int DeepSpin::dim_chg_spin() const { return dp->dim_chg_spin(); }

void DeepSpin::set_charge_spin(const std::vector<double>& charge_spin) {
  dp->set_charge_spin(charge_spin);
}

std::vector<bool> DeepSpin::get_use_spin() const {
  if (dp) {
    return dp->get_use_spin();
  }
  return {};
}

void DeepSpinBackend::compute_canonical_graph_gpu(
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
  (void)d_atom_energy;
  (void)d_force;
  (void)d_force_mag;
  (void)d_atom_virial;
  (void)d_atype;
  (void)d_source;
  (void)d_edge_vec;
  (void)d_destination_row_ptr;
  (void)d_source_row_ptr;
  (void)d_source_order;
  (void)d_spin;
  (void)nloc;
  (void)nall_nodes;
  (void)edge_storage;
  throw deepmd::deepmd_exception(
      "compact canonical graph inference is only supported by a compatible "
      "PyTorch Exportable backend.");
}

bool DeepSpinBackend::uses_canonical_graph_inference() const { return false; }

bool DeepSpinBackend::uses_native_spin_scheme() const { return false; }

void DeepSpin::compute_canonical_graph_gpu(
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
  // Backend-agnostic dispatch: backends that implement device-resident
  // canonical inference override the hook, while the others inherit the
  // throwing default. ``libdeepmd_cc`` does not link any backend, so the
  // dispatch stays virtual rather than casting to a concrete backend type.
  dp->compute_canonical_graph_gpu(
      d_atom_energy, d_force, d_force_mag, d_atom_virial, d_atype, d_source,
      d_edge_vec, d_destination_row_ptr, d_source_row_ptr, d_source_order,
      d_spin, nloc, nall_nodes, edge_storage);
}

bool DeepSpin::uses_canonical_graph_inference() const {
  return dp->uses_canonical_graph_inference();
}

bool DeepSpin::uses_native_spin_scheme() const {
  return dp->uses_native_spin_scheme();
}

DeepSpinModelDevi::DeepSpinModelDevi() {
  inited = false;
  numb_models = 0;
}

DeepSpinModelDevi::DeepSpinModelDevi(
    const std::vector<std::string>& models,
    const int& gpu_rank,
    const std::vector<std::string>& file_contents) {
  inited = false;
  numb_models = 0;
  init(models, gpu_rank, file_contents);
}

DeepSpinModelDevi::~DeepSpinModelDevi() {}

void DeepSpinModelDevi::init(const std::vector<std::string>& models,
                             const int& gpu_rank,
                             const std::vector<std::string>& file_contents) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  numb_models = models.size();
  if (numb_models == 0) {
    throw deepmd::deepmd_exception("no model is specified");
  }
  dps.resize(numb_models);
  dpbases.resize(numb_models);
  for (unsigned int ii = 0; ii < numb_models; ++ii) {
    dps[ii] = std::make_shared<deepmd::DeepSpin>();
    dps[ii]->init(models[ii], gpu_rank,
                  file_contents.size() > ii ? file_contents[ii] : "");
    dpbases[ii] = dps[ii];
  }
  inited = true;
}

template <typename VALUETYPE>
void DeepSpinModelDevi::compute(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<VALUETYPE>>& all_force,
    std::vector<std::vector<VALUETYPE>>& all_force_mag,
    std::vector<std::vector<VALUETYPE>>& all_virial,
    const std::vector<VALUETYPE>& dcoord_,
    const std::vector<VALUETYPE>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam_,
    const std::vector<double>& charge_spin) {
  // without nlist
  if (numb_models == 0) {
    return;
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_force_mag.resize(numb_models);
  all_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_force_mag[ii],
                     all_virial[ii], dcoord_, dspin_, datype_, dbox, fparam,
                     aparam_, charge_spin);
  }
}

template void DeepSpinModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_force_mag,
    std::vector<std::vector<double>>& all_virial,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepSpinModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_force_mag,
    std::vector<std::vector<float>>& all_virial,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepSpinModelDevi::compute(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<VALUETYPE>>& all_force,
    std::vector<std::vector<VALUETYPE>>& all_force_mag,
    std::vector<std::vector<VALUETYPE>>& all_virial,
    std::vector<std::vector<VALUETYPE>>& all_atom_energy,
    std::vector<std::vector<VALUETYPE>>& all_atom_virial,
    const std::vector<VALUETYPE>& dcoord_,
    const std::vector<VALUETYPE>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam_,
    const std::vector<double>& charge_spin) {
  if (numb_models == 0) {
    return;
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_force_mag.resize(numb_models);
  all_virial.resize(numb_models);
  all_atom_energy.resize(numb_models);
  all_atom_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_force_mag[ii],
                     all_virial[ii], all_atom_energy[ii], all_atom_virial[ii],
                     dcoord_, dspin_, datype_, dbox, fparam, aparam_,
                     charge_spin);
  }
}

template void DeepSpinModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_force_mag,
    std::vector<std::vector<double>>& all_virial,
    std::vector<std::vector<double>>& all_atom_energy,
    std::vector<std::vector<double>>& all_atom_virial,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepSpinModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_force_mag,
    std::vector<std::vector<float>>& all_virial,
    std::vector<std::vector<float>>& all_atom_energy,
    std::vector<std::vector<float>>& all_atom_virial,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

// support spin
// nlist, no atomic
template <typename VALUETYPE>
void DeepSpinModelDevi::compute(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<VALUETYPE>>& all_force,
    std::vector<std::vector<VALUETYPE>>& all_force_mag,
    std::vector<std::vector<VALUETYPE>>& all_virial,
    const std::vector<VALUETYPE>& dcoord_,
    const std::vector<VALUETYPE>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam_,
    const std::vector<double>& charge_spin) {
  if (numb_models == 0) {
    return;
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_force_mag.resize(numb_models);
  all_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_force_mag[ii],
                     all_virial[ii], dcoord_, dspin_, datype_, dbox, nghost,
                     lmp_list, ago, fparam, aparam_, charge_spin);
  }
}

// nlist, no atomic: precision
template void DeepSpinModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_force_mag,
    std::vector<std::vector<double>>& all_virial,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepSpinModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_force_mag,
    std::vector<std::vector<float>>& all_virial,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

// support spin
// nlist, atomic
template <typename VALUETYPE>
void DeepSpinModelDevi::compute(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<VALUETYPE>>& all_force,
    std::vector<std::vector<VALUETYPE>>& all_force_mag,
    std::vector<std::vector<VALUETYPE>>& all_virial,
    std::vector<std::vector<VALUETYPE>>& all_atom_energy,
    std::vector<std::vector<VALUETYPE>>& all_atom_virial,
    const std::vector<VALUETYPE>& dcoord_,
    const std::vector<VALUETYPE>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam_,
    const std::vector<double>& charge_spin) {
  if (numb_models == 0) {
    return;
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_force_mag.resize(numb_models);
  all_virial.resize(numb_models);
  all_atom_energy.resize(numb_models);
  all_atom_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_force_mag[ii],
                     all_virial[ii], all_atom_energy[ii], all_atom_virial[ii],
                     dcoord_, dspin_, datype_, dbox, nghost, lmp_list, ago,
                     fparam, aparam_, charge_spin);
  }
}

// nlist, atomic : precision
template void DeepSpinModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_force_mag,
    std::vector<std::vector<double>>& all_virial,
    std::vector<std::vector<double>>& all_atom_energy,
    std::vector<std::vector<double>>& all_atom_virial,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepSpinModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_force_mag,
    std::vector<std::vector<float>>& all_virial,
    std::vector<std::vector<float>>& all_atom_energy,
    std::vector<std::vector<float>>& all_atom_virial,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

std::vector<bool> DeepSpinModelDevi::get_use_spin() const {
  if (!dps.empty()) {
    return dps[0]->get_use_spin();
  }
  return {};
}
