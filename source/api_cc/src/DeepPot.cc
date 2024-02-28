// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPot.h"

#include "common.h"
// TODO: only include when TF backend is built
#include <memory>
#include <stdexcept>

#include "AtomMap.h"
#ifdef BUILD_TENSORFLOW
#include "DeepPotTF.h"
#endif
#ifdef BUILD_PYTORCH
#include "DeepPotPT.h"
#endif
#include "device.h"

using namespace deepmd;

DeepPot::DeepPot() : inited(false) {}

DeepPot::DeepPot(const std::string& model,
                 const int& gpu_rank,
                 const std::string& file_content)
    : inited(false) {
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
  DPBackend backend;
  if (model.length() >= 4 && model.substr(model.length() - 4) == ".pth") {
    backend = deepmd::DPBackend::PyTorch;
  } else if (model.length() >= 3 && model.substr(model.length() - 3) == ".pb") {
    backend = deepmd::DPBackend::TensorFlow;
  } else {
    throw deepmd::deepmd_exception("Unsupported model file format");
  }
  if (deepmd::DPBackend::TensorFlow == backend) {
#ifdef BUILD_TENSORFLOW
    dp = std::make_shared<deepmd::DeepPotTF>(model, gpu_rank, file_content);
#else
    throw deepmd::deepmd_exception("TensorFlow backend is not built");
#endif
  } else if (deepmd::DPBackend::PyTorch == backend) {
#ifdef BUILD_PYTORCH
    dp = std::make_shared<deepmd::DeepPotPT>(model, gpu_rank, file_content);
#else
    throw deepmd::deepmd_exception("PyTorch backend is not built");
#endif
  } else if (deepmd::DPBackend::Paddle == backend) {
    throw deepmd::deepmd_exception("PaddlePaddle backend is not supported yet");
  } else {
    throw deepmd::deepmd_exception("Unknown file type");
  }
  inited = true;
}

void DeepPot::print_summary(const std::string& pre) const {
  deepmd::print_summary(pre);
}

template <typename VALUETYPE>
void DeepPot::compute(ENERGYTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam_) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_);
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
                      const std::vector<VALUETYPE>& aparam_) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_);
}

template void DeepPot::compute<double>(ENERGYTYPE& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam);

template void DeepPot::compute<float>(ENERGYTYPE& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam);

template void DeepPot::compute<double>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam);

template void DeepPot::compute<float>(std::vector<ENERGYTYPE>& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam);

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
                      const std::vector<VALUETYPE>& aparam__) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__);
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
                      const std::vector<VALUETYPE>& aparam__) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__);
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
                                       const std::vector<double>& aparam_);

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
                                      const std::vector<float>& aparam_);

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
                                       const std::vector<double>& aparam_);

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
                                      const std::vector<float>& aparam_);

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
                      const std::vector<VALUETYPE>& aparam_) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_);
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
                      const std::vector<VALUETYPE>& aparam_) {
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_);
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
                                       const std::vector<double>& aparam);

template void DeepPot::compute<float>(ENERGYTYPE& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      std::vector<float>& datom_energy_,
                                      std::vector<float>& datom_virial_,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam);

template void DeepPot::compute<double>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<double>& dforce_,
                                       std::vector<double>& dvirial,
                                       std::vector<double>& datom_energy_,
                                       std::vector<double>& datom_virial_,
                                       const std::vector<double>& dcoord_,
                                       const std::vector<int>& datype_,
                                       const std::vector<double>& dbox,
                                       const std::vector<double>& fparam,
                                       const std::vector<double>& aparam);

template void DeepPot::compute<float>(std::vector<ENERGYTYPE>& dener,
                                      std::vector<float>& dforce_,
                                      std::vector<float>& dvirial,
                                      std::vector<float>& datom_energy_,
                                      std::vector<float>& datom_virial_,
                                      const std::vector<float>& dcoord_,
                                      const std::vector<int>& datype_,
                                      const std::vector<float>& dbox,
                                      const std::vector<float>& fparam,
                                      const std::vector<float>& aparam);

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
                      const std::vector<VALUETYPE>& aparam__) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__);
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
                      const std::vector<VALUETYPE>& aparam__) {
  dp->computew(dener, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__);
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
                                       const std::vector<double>& aparam_);

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
                                      const std::vector<float>& aparam_);

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
                                       const std::vector<double>& aparam_);

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
                                      const std::vector<float>& aparam_);

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
                                 const std::vector<VALUETYPE>& aparam_) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew_mixed_type(dener_, dforce_, dvirial, datom_energy_,
                          datom_virial_, nframes, dcoord_, datype_, dbox,
                          fparam_, aparam_);
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
                                 const std::vector<VALUETYPE>& aparam_) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew_mixed_type(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                          nframes, dcoord_, datype_, dbox, fparam_, aparam_);
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
    const std::vector<double>& aparam);

template void DeepPot::compute_mixed_type<float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template void DeepPot::compute_mixed_type<double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute_mixed_type<float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

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
                                 const std::vector<VALUETYPE>& aparam_) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew_mixed_type(dener_, dforce_, dvirial, datom_energy_,
                          datom_virial_, nframes, dcoord_, datype_, dbox,
                          fparam_, aparam_);
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
                                 const std::vector<VALUETYPE>& aparam_) {
  dp->computew_mixed_type(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                          nframes, dcoord_, datype_, dbox, fparam_, aparam_);
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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);

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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);

double DeepPot::cutoff() const { return dp->cutoff(); }

int DeepPot::numb_types() const { return dp->numb_types(); }

int DeepPot::numb_types_spin() const { return dp->numb_types_spin(); }

int DeepPot::dim_fparam() const { return dp->dim_fparam(); }

int DeepPot::dim_aparam() const { return dp->dim_aparam(); }

void DeepPot::get_type_map(std::string& type_map) {
  dp->get_type_map(type_map);
}

bool DeepPot::is_aparam_nall() const { return dp->is_aparam_nall(); }

DeepPotModelDevi::DeepPotModelDevi() : inited(false), numb_models(0) {}

DeepPotModelDevi::DeepPotModelDevi(
    const std::vector<std::string>& models,
    const int& gpu_rank,
    const std::vector<std::string>& file_contents)
    : inited(false), numb_models(0) {
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
  for (unsigned int ii = 0; ii < numb_models; ++ii) {
    dps[ii].init(models[ii], gpu_rank,
                 file_contents.size() > ii ? file_contents[ii] : "");
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
                               const int nghost,
                               const InputNlist& lmp_list,
                               const int& ago,
                               const std::vector<VALUETYPE>& fparam,
                               const std::vector<VALUETYPE>& aparam_) {
  if (numb_models == 0) {
    return;
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii].compute(all_energy[ii], all_force[ii], all_virial[ii], dcoord_,
                    datype_, dbox, nghost, lmp_list, ago, fparam, aparam_);
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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);

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
    const std::vector<VALUETYPE>& aparam_) {
  if (numb_models == 0) {
    return;
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_virial.resize(numb_models);
  all_atom_energy.resize(numb_models);
  all_atom_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii].compute(all_energy[ii], all_force[ii], all_virial[ii],
                    all_atom_energy[ii], all_atom_virial[ii], dcoord_, datype_,
                    dbox, nghost, lmp_list, ago, fparam, aparam_);
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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_avg(VALUETYPE& dener,
                                   const std::vector<VALUETYPE>& all_energy) {
  assert(all_energy.size() == numb_models);
  if (numb_models == 0) {
    return;
  }

  dener = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dener += all_energy[ii];
  }
  dener /= (VALUETYPE)(numb_models);
}

template void DeepPotModelDevi::compute_avg<double>(
    double& dener, const std::vector<double>& all_energy);

template void DeepPotModelDevi::compute_avg<float>(
    float& dener, const std::vector<float>& all_energy);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_avg(
    std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  assert(xx.size() == numb_models);
  if (numb_models == 0) {
    return;
  }

  avg.resize(xx[0].size());
  fill(avg.begin(), avg.end(), VALUETYPE(0.));

  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0; jj < avg.size(); ++jj) {
      avg[jj] += xx[ii][jj];
    }
  }

  for (unsigned jj = 0; jj < avg.size(); ++jj) {
    avg[jj] /= VALUETYPE(numb_models);
  }
}

template void DeepPotModelDevi::compute_avg<double>(
    std::vector<double>& avg, const std::vector<std::vector<double>>& xx);

template void DeepPotModelDevi::compute_avg<float>(
    std::vector<float>& avg, const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_std(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx,
    const int& stride) {
  assert(xx.size() == numb_models);
  if (numb_models == 0) {
    return;
  }

  unsigned ndof = avg.size();
  unsigned nloc = ndof / stride;
  assert(nloc * stride == ndof);

  std.resize(nloc);
  fill(std.begin(), std.end(), VALUETYPE(0.));

  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0; jj < nloc; ++jj) {
      const VALUETYPE* tmp_f = &(xx[ii][static_cast<size_t>(jj) * stride]);
      const VALUETYPE* tmp_avg = &(avg[static_cast<size_t>(jj) * stride]);
      for (unsigned dd = 0; dd < stride; ++dd) {
        VALUETYPE vdiff = tmp_f[dd] - tmp_avg[dd];
        std[jj] += vdiff * vdiff;
      }
    }
  }

  for (unsigned jj = 0; jj < nloc; ++jj) {
    std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
  }
}

template void DeepPotModelDevi::compute_std<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx,
    const int& stride);

template void DeepPotModelDevi::compute_std<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx,
    const int& stride);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_std_e(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  compute_std(std, avg, xx, 1);
}

template void DeepPotModelDevi::compute_std_e<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx);

template void DeepPotModelDevi::compute_std_e<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_std_f(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  compute_std(std, avg, xx, 3);
}

template void DeepPotModelDevi::compute_std_f<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx);

template void DeepPotModelDevi::compute_std_f<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_relative_std(std::vector<VALUETYPE>& std,
                                            const std::vector<VALUETYPE>& avg,
                                            const VALUETYPE eps,
                                            const int& stride) {
  unsigned ndof = avg.size();
  unsigned nloc = std.size();
  assert(nloc * stride == ndof);

  for (unsigned ii = 0; ii < nloc; ++ii) {
    const VALUETYPE* tmp_avg = &(avg[static_cast<size_t>(ii) * stride]);
    VALUETYPE f_norm = 0.0;
    for (unsigned dd = 0; dd < stride; ++dd) {
      f_norm += tmp_avg[dd] * tmp_avg[dd];
    }
    f_norm = sqrt(f_norm);
    std[ii] /= f_norm + eps;
  }
}

template void DeepPotModelDevi::compute_relative_std<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const double eps,
    const int& stride);

template void DeepPotModelDevi::compute_relative_std<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const float eps,
    const int& stride);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_relative_std_f(std::vector<VALUETYPE>& std,
                                              const std::vector<VALUETYPE>& avg,
                                              const VALUETYPE eps) {
  compute_relative_std(std, avg, eps, 3);
}

template void DeepPotModelDevi::compute_relative_std_f<double>(
    std::vector<double>& std, const std::vector<double>& avg, const double eps);

template void DeepPotModelDevi::compute_relative_std_f<float>(
    std::vector<float>& std, const std::vector<float>& avg, const float eps);
