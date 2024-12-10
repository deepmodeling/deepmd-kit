// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepPot.h"

#include <memory>
#include <stdexcept>

#include "AtomMap.h"
#include "common.h"
#ifdef BUILD_TENSORFLOW
#include "DeepPotTF.h"
#endif
#ifdef BUILD_PYTORCH
#include "DeepPotPT.h"
#endif
#if defined(BUILD_TENSORFLOW) || defined(BUILD_JAX)
#include "DeepPotJAX.h"
#endif
#ifdef BUILD_PADDLE
#include "DeepPotPD.h"
#endif
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
#ifdef BUILD_PADDLE
    dp = std::make_shared<deepmd::DeepPotPD>(model, gpu_rank, file_content);
#else
    throw deepmd::deepmd_exception("PaddlePaddle backend is not supported yet");
#endif
  } else if (deepmd::DPBackend::JAX == backend) {
#if defined(BUILD_TENSORFLOW) || defined(BUILD_JAX)
    dp = std::make_shared<deepmd::DeepPotJAX>(model, gpu_rank, file_content);
#else
    throw deepmd::deepmd_exception(
        "TensorFlow backend is not built, which is used to load JAX2TF "
        "SavedModels");
#endif
  } else {
    throw deepmd::deepmd_exception("Unknown file type");
  }
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
                      const std::vector<VALUETYPE>& aparam_) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dvirial, datom_energy_, datom_virial_, dcoord_,
               datype_, dbox, fparam_, aparam_, false);
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
               datype_, dbox, fparam_, aparam_, false);
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
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__, false);
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
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__, false);
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
               datype_, dbox, fparam_, aparam_, true);
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
               datype_, dbox, fparam_, aparam_, true);
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
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__, true);
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
               datype_, dbox, nghost, lmp_list, ago, fparam_, aparam__, true);
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
                          fparam_, aparam_, false);
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
                          nframes, dcoord_, datype_, dbox, fparam_, aparam_,
                          false);
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
                          fparam_, aparam_, true);
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
                          nframes, dcoord_, datype_, dbox, fparam_, aparam_,
                          true);
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
                               const std::vector<VALUETYPE>& aparam_) {
  // without nlist
  if (numb_models == 0) {
    return;
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_virial.resize(numb_models);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii], dcoord_,
                     datype_, dbox, fparam, aparam_);
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
    const std::vector<double>& aparam);

template void DeepPotModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_virial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
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
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii],
                     all_atom_energy[ii], all_atom_virial[ii], dcoord_, datype_,
                     dbox, fparam, aparam_);
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
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

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
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii], dcoord_,
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
    dps[ii]->compute(all_energy[ii], all_force[ii], all_virial[ii],
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
