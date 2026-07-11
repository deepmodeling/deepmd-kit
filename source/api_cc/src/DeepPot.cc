// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPot.h"

#include <memory>
#include <stdexcept>

#include "AtomMap.h"
#include "BackendPlugin.h"
#include "common.h"
#include "device.h"

using namespace deepmd;

DeepPot::DeepPot() { inited = false; }

DeepPot::DeepPot(const std::string& model,
                 const int& gpu_rank,
                 const std::string& file_content) {
  inited = false;
  init(model, gpu_rank, file_content);
}

DeepPot::~DeepPot() {}

void DeepPot::init(const std::string& model,
                   const int& gpu_rank,
                   const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  const DPBackend backend = get_backend(model);
  if (deepmd::DPBackend::Unknown == backend) {
    throw deepmd::deepmd_exception("Unknown file type");
  }
  dp = create_deeppot_backend_from_plugin(backend, model, gpu_rank,
                                          file_content);
  inited = true;
  dpbase = dp;  // make sure the base funtions work
}

template <typename VALUETYPE>
void DeepPot::compute(ENERGYTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam_,
                      const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_, charge_spin, false);
  dener = dener_[0];
}

template <typename VALUETYPE>
void DeepPot::compute(std::vector<ENERGYTYPE>& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam_,
                      const std::vector<double>& charge_spin) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_, charge_spin, false);
}

template void DeepPot::compute<double>(ENERGYTYPE& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(ENERGYTYPE& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam,
                                      const std::vector<double>& charge_spin);

template void DeepPot::compute<double>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(std::vector<ENERGYTYPE>& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam,
                                      const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepPot::compute(ENERGYTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      const std::vector<VALUETYPE>& dcoord_,
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
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__,
               charge_spin, false);
  dener = dener_[0];
}

template <typename VALUETYPE>
void DeepPot::compute(std::vector<ENERGYTYPE>& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const int nghost,
                      const InputNlist& lmp_list,
                      const int& ago,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam__,
                      const std::vector<double>& charge_spin) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__,
               charge_spin, false);
}

template void DeepPot::compute<double>(ENERGYTYPE& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam_,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(ENERGYTYPE& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const int nghost,
                                      const InputNlist& lmp_list,
                                      const int& ago,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam_,
                                      const std::vector<double>& charge_spin);

template void DeepPot::compute<double>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam_,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(std::vector<ENERGYTYPE>& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const int nghost,
                                      const InputNlist& lmp_list,
                                      const int& ago,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam_,
                                      const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepPot::compute(ENERGYTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      std::vector<VALUETYPE>& datom_energy_,
                      std::vector<VALUETYPE>& datom_virial_,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam_,
                      const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_, charge_spin, true);
  dener = dener_[0];
}
template <typename VALUETYPE>
void DeepPot::compute(std::vector<ENERGYTYPE>& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      std::vector<VALUETYPE>& datom_energy_,
                      std::vector<VALUETYPE>& datom_virial_,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam_,
                      const std::vector<double>& charge_spin) {
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_, charge_spin, true);
}

template void DeepPot::compute<double>(ENERGYTYPE& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       std::vector<double>& datom_energy_,
                                       std::vector<double>& datom_virial_,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(ENERGYTYPE& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      std::vector<float>& datom_energy_,
                                      std::vector<float>& datom_virial_,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam,
                                      const std::vector<double>& charge_spin);

template void DeepPot::compute<double>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       std::vector<double>& datom_energy_,
                                       std::vector<double>& datom_virial_,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(std::vector<ENERGYTYPE>& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      std::vector<float>& datom_energy_,
                                      std::vector<float>& datom_virial_,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam,
                                      const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepPot::compute(ENERGYTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      std::vector<VALUETYPE>& datom_energy_,
                      std::vector<VALUETYPE>& datom_virial_,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const int nghost,
                      const InputNlist& lmp_list,
                      const int& ago,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam__,
                      const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__,
               charge_spin, true);
  dener = dener_[0];
}
template <typename VALUETYPE>
void DeepPot::compute(std::vector<ENERGYTYPE>& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      std::vector<VALUETYPE>& datom_energy_,
                      std::vector<VALUETYPE>& datom_virial_,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const int nghost,
                      const InputNlist& lmp_list,
                      const int& ago,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam__,
                      const std::vector<double>& charge_spin) {
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__,
               charge_spin, true);
}

template void DeepPot::compute<double>(ENERGYTYPE& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       std::vector<double>& datom_energy_,
                                       std::vector<double>& datom_virial_,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam_,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(ENERGYTYPE& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      std::vector<float>& datom_energy_,
                                      std::vector<float>& datom_virial_,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const int nghost,
                                      const InputNlist& lmp_list,
                                      const int& ago,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam_,
                                      const std::vector<double>& charge_spin);

template void DeepPot::compute<double>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       std::vector<double>& datom_energy_,
                                       std::vector<double>& datom_virial_,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const int nghost,
                                       const InputNlist& lmp_list,
                                       const int& ago,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam_,
                                       const std::vector<double>& charge_spin);

template void DeepPot::compute<float>(std::vector<ENERGYTYPE>& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      std::vector<float>& datom_energy_,
                                      std::vector<float>& datom_virial_,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const int nghost,
                                      const InputNlist& lmp_list,
                                      const int& ago,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam_,
                                      const std::vector<double>& charge_spin);

// mixed type
template <typename VALUETYPE>
void DeepPot::compute_mixed_type(ENERGYTYPE& dener,
                                 std::vector<VALUETYPE>& dforce_,
                                 std::vector<VALUETYPE>& dvirial,
                                 const int& nframes,
                                 const std::vector<VALUETYPE>& dcoord_,
                                 const std::vector<int>& datype_,
                                 const std::vector<VALUETYPE>& dbox,
                                 const std::vector<VALUETYPE>& fparam_,
                                 const std::vector<VALUETYPE>& aparam_,
                                 const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew_mixed_type(dener_, dforce_, dvirial, datom_energy_,
                          datom_virial_, nframes, dcoord_, datype_, dbox,
                          fparam_, aparam_, charge_spin, false);
  dener = dener_[0];
}
template <typename VALUETYPE>
void DeepPot::compute_mixed_type(std::vector<ENERGYTYPE>& dener,
                                 std::vector<VALUETYPE>& dforce_,
                                 std::vector<VALUETYPE>& dvirial,
                                 const int& nframes,
                                 const std::vector<VALUETYPE>& dcoord_,
                                 const std::vector<int>& datype_,
                                 const std::vector<VALUETYPE>& dbox,
                                 const std::vector<VALUETYPE>& fparam_,
                                 const std::vector<VALUETYPE>& aparam_,
                                 const std::vector<double>& charge_spin) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew_mixed_type(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                          nframes, dcoord_, datype_, dbox, fparam_, aparam_,
                          charge_spin, false);
}

template void DeepPot::compute_mixed_type<double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPot::compute_mixed_type<float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPot::compute_mixed_type<double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPot::compute_mixed_type<float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepPot::compute_mixed_type(ENERGYTYPE& dener,
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
                                 const std::vector<double>& charge_spin) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew_mixed_type(dener_, dforce_, dvirial, datom_energy_,
                          datom_virial_, nframes, dcoord_, datype_, dbox,
                          fparam_, aparam_, charge_spin, true);
  dener = dener_[0];
}
template <typename VALUETYPE>
void DeepPot::compute_mixed_type(std::vector<ENERGYTYPE>& dener,
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
                                 const std::vector<double>& charge_spin) {
  dp->computew_mixed_type(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                          nframes, dcoord_, datype_, dbox, fparam_, aparam_,
                          charge_spin, true);
}

template void DeepPot::compute_mixed_type<double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPot::compute_mixed_type<float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPot::compute_mixed_type<double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPot::compute_mixed_type<float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

void DeepPotBackend::compute_edges_gpu(double* d_atom_energy,
                                       double* d_force,
                                       double* d_atom_virial,
                                       const double* d_coord,
                                       const int* d_atype,
                                       const int* d_edge_index,
                                       const double* d_edge_vec,
                                       const int nloc,
                                       const int nedge,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam,
                                       const int nall_nodes,
                                       const InputNlist* comm_nlist) {
  (void)d_atom_energy;
  (void)d_force;
  (void)d_atom_virial;
  (void)d_coord;
  (void)d_atype;
  (void)d_edge_index;
  (void)d_edge_vec;
  (void)comm_nlist;
  (void)nloc;
  (void)nedge;
  (void)fparam;
  (void)aparam;
  (void)nall_nodes;
  throw deepmd::deepmd_exception(
      "compute_edges_gpu (GPU-resident edge inference) is only supported by "
      "the "
      "PyTorch Exportable (.pt2) backend.");
}

void DeepPotBackend::compute_edges_gpu(double* d_atom_energy,
                                       double* d_force,
                                       double* d_atom_virial,
                                       const double* d_coord,
                                       const int* d_atype,
                                       const int* d_edge_index,
                                       const float* d_edge_vec,
                                       const int nloc,
                                       const int nedge,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam,
                                       const int nall_nodes,
                                       const InputNlist* comm_nlist) {
  (void)d_atom_energy;
  (void)d_force;
  (void)d_atom_virial;
  (void)d_coord;
  (void)d_atype;
  (void)d_edge_index;
  (void)d_edge_vec;
  (void)comm_nlist;
  (void)nloc;
  (void)nedge;
  (void)fparam;
  (void)aparam;
  (void)nall_nodes;
  throw deepmd::deepmd_exception(
      "compute_edges_gpu with float32 edge vectors is only supported by a "
      "compatible PyTorch Exportable (.pt2) backend.");
}

void DeepPotBackend::compute_canonical_graph_gpu(
    double* d_atom_energy,
    double* d_force,
    double* d_atom_virial,
    const std::int64_t* d_atype,
    const std::int64_t* d_source,
    const float* d_edge_vec,
    const std::int64_t* d_destination_row_ptr,
    const std::int64_t* d_source_row_ptr,
    const std::int64_t* d_source_order,
    const int nloc,
    const int nall_nodes,
    const std::int64_t edge_storage) {
  (void)d_atom_energy;
  (void)d_force;
  (void)d_atom_virial;
  (void)d_atype;
  (void)d_source;
  (void)d_edge_vec;
  (void)d_destination_row_ptr;
  (void)d_source_row_ptr;
  (void)d_source_order;
  (void)nloc;
  (void)nall_nodes;
  (void)edge_storage;
  throw deepmd::deepmd_exception(
      "compact canonical graph inference is only supported by a compatible "
      "PyTorch Exportable backend.");
}

bool DeepPotBackend::uses_fp32_edge_vectors() const { return false; }

bool DeepPotBackend::supports_device_edge_inference() const { return false; }

bool DeepPotBackend::uses_canonical_graph_inference() const { return false; }

void DeepPot::compute_edges_gpu(double* d_atom_energy,
                                double* d_force,
                                double* d_atom_virial,
                                const double* d_coord,
                                const int* d_atype,
                                const int* d_edge_index,
                                const double* d_edge_vec,
                                const int nloc,
                                const int nedge) {
  // Parameter-free path: no runtime fparam/aparam, so the model defaults apply.
  compute_edges_gpu(d_atom_energy, d_force, d_atom_virial, d_coord, d_atype,
                    d_edge_index, d_edge_vec, nloc, nedge, {}, {}, 0, nullptr);
}

void DeepPot::compute_edges_gpu(double* d_atom_energy,
                                double* d_force,
                                double* d_atom_virial,
                                const double* d_coord,
                                const int* d_atype,
                                const int* d_edge_index,
                                const double* d_edge_vec,
                                const int nloc,
                                const int nedge,
                                const std::vector<double>& fparam,
                                const std::vector<double>& aparam,
                                const int nall_nodes,
                                const InputNlist* comm_nlist) {
  // Backend-agnostic dispatch: backends that implement device edge inference
  // override ``compute_edges_gpu``, while the others inherit the throwing
  // default. ``libdeepmd_cc`` does not link any backend, so the dispatch stays
  // virtual rather than casting to a concrete backend type.
  dp->compute_edges_gpu(d_atom_energy, d_force, d_atom_virial, d_coord, d_atype,
                        d_edge_index, d_edge_vec, nloc, nedge, fparam, aparam,
                        nall_nodes, comm_nlist);
}

void DeepPot::compute_edges_gpu(double* d_atom_energy,
                                double* d_force,
                                double* d_atom_virial,
                                const double* d_coord,
                                const int* d_atype,
                                const int* d_edge_index,
                                const float* d_edge_vec,
                                const int nloc,
                                const int nedge,
                                const std::vector<double>& fparam,
                                const std::vector<double>& aparam,
                                const int nall_nodes,
                                const InputNlist* comm_nlist) {
  dp->compute_edges_gpu(d_atom_energy, d_force, d_atom_virial, d_coord, d_atype,
                        d_edge_index, d_edge_vec, nloc, nedge, fparam, aparam,
                        nall_nodes, comm_nlist);
}

void DeepPot::compute_canonical_graph_gpu(
    double* d_atom_energy,
    double* d_force,
    double* d_atom_virial,
    const std::int64_t* d_atype,
    const std::int64_t* d_source,
    const float* d_edge_vec,
    const std::int64_t* d_destination_row_ptr,
    const std::int64_t* d_source_row_ptr,
    const std::int64_t* d_source_order,
    const int nloc,
    const int nall_nodes,
    const std::int64_t edge_storage) {
  dp->compute_canonical_graph_gpu(
      d_atom_energy, d_force, d_atom_virial, d_atype, d_source, d_edge_vec,
      d_destination_row_ptr, d_source_row_ptr, d_source_order, nloc, nall_nodes,
      edge_storage);
}

bool DeepPot::uses_fp32_edge_vectors() const {
  return dp->uses_fp32_edge_vectors();
}

bool DeepPot::supports_device_edge_inference() const {
  return dp->supports_device_edge_inference();
}

bool DeepPot::uses_canonical_graph_inference() const {
  return dp->uses_canonical_graph_inference();
}

int DeepPot::dim_chg_spin() const { return dp->dim_chg_spin(); }

DeepPotModelDevi::DeepPotModelDevi() {
  inited = false;
  numb_models = 0;
}

DeepPotModelDevi::DeepPotModelDevi(
    const std::vector<std::string>& models,
    const int& gpu_rank,
    const std::vector<std::string>& file_contents) {
  inited = false;
  numb_models = 0;
  init(models, gpu_rank, file_contents);
}

DeepPotModelDevi::~DeepPotModelDevi() {}

void DeepPotModelDevi::init(const std::vector<std::string>& models,
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
    dps[ii] = std::make_shared<deepmd::DeepPot>();
    dps[ii]->init(models[ii], gpu_rank,
                  file_contents.size() > ii ? file_contents[ii] : "");
    dpbases[ii] = dps[ii];
  }
  inited = true;
}

template <typename VALUETYPE>
void DeepPotModelDevi::compute(std::vector<ENERGYTYPE>& all_energy,
                               std::vector<std::vector<VALUETYPE>>& all_force,
                               std::vector<std::vector<VALUETYPE>>& all_virial,
                               const std::vector<VALUETYPE>& dcoord_,
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
  all_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii], dcoord_,
                     datype_, dbox, fparam, aparam_, charge_spin);
  }
}

template void DeepPotModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_virial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPotModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_virial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepPotModelDevi::compute(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<VALUETYPE>>& all_force,
    std::vector<std::vector<VALUETYPE>>& all_virial,
    std::vector<std::vector<VALUETYPE>>& all_atom_energy,
    std::vector<std::vector<VALUETYPE>>& all_atom_virial,
    const std::vector<VALUETYPE>& dcoord_,
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
  all_virial.resize(numb_models);
  all_atom_energy.resize(numb_models);
  all_atom_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii],
                     all_atom_energy[ii], all_atom_virial[ii], dcoord_, datype_,
                     dbox, fparam, aparam_, charge_spin);
  }
}

template void DeepPotModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_virial,
    std::vector<std::vector<double>>& all_atom_energy,
    std::vector<std::vector<double>>& all_atom_virial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPotModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_virial,
    std::vector<std::vector<float>>& all_atom_energy,
    std::vector<std::vector<float>>& all_atom_virial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepPotModelDevi::compute(std::vector<ENERGYTYPE>& all_energy,
                               std::vector<std::vector<VALUETYPE>>& all_force,
                               std::vector<std::vector<VALUETYPE>>& all_virial,
                               const std::vector<VALUETYPE>& dcoord_,
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
  all_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii], dcoord_,
                     datype_, dbox, nghost, lmp_list, ago, fparam, aparam_,
                     charge_spin);
  }
}

template void DeepPotModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_virial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPotModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_virial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);

template <typename VALUETYPE>
void DeepPotModelDevi::compute(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<VALUETYPE>>& all_force,
    std::vector<std::vector<VALUETYPE>>& all_virial,
    std::vector<std::vector<VALUETYPE>>& all_atom_energy,
    std::vector<std::vector<VALUETYPE>>& all_atom_virial,
    const std::vector<VALUETYPE>& dcoord_,
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
  all_virial.resize(numb_models);
  all_atom_energy.resize(numb_models);
  all_atom_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii],
                     all_atom_energy[ii], all_atom_virial[ii], dcoord_, datype_,
                     dbox, nghost, lmp_list, ago, fparam, aparam_, charge_spin);
  }
}

template void DeepPotModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_virial,
    std::vector<std::vector<double>>& all_atom_energy,
    std::vector<std::vector<double>>& all_atom_virial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const std::vector<double>& charge_spin);

template void DeepPotModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_virial,
    std::vector<std::vector<float>>& all_atom_energy,
    std::vector<std::vector<float>>& all_atom_virial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const std::vector<double>& charge_spin);
