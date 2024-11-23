// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepSpin.h"

#include <memory>
#include <stdexcept>

#include "AtomMap.h"
#include "common.h"
#ifdef BUILD_TENSORFLOW
#include "DeepSpinTF.h"
#endif
#ifdef BUILD_PYTORCH
#include "DeepSpinPT.h"
#endif
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
  if (deepmd::DPBackend::TensorFlow == backend) {
#ifdef BUILD_TENSORFLOW
    dp = std::make_shared<deepmd::DeepSpinTF>(model, gpu_rank, file_content);
#else
    throw deepmd::deepmd_exception("TensorFlow backend is not built");
#endif
  } else if (deepmd::DPBackend::PyTorch == backend) {
#ifdef BUILD_PYTORCH
    dp = std::make_shared<deepmd::DeepSpinPT>(model, gpu_rank, file_content);
#else
    throw deepmd::deepmd_exception("PyTorch backend is not built");
#endif
  } else if (deepmd::DPBackend::Paddle == backend) {
    throw deepmd::deepmd_exception("PaddlePaddle backend is not supported yet");
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
                       const std::vector<VALUETYPE>& aparam_) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               false);
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
                       const std::vector<VALUETYPE>& aparam_) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               false);
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
                                        const std::vector<double>& aparam);

template void DeepSpin::compute<float>(ENERGYTYPE& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam);

template void DeepSpin::compute<double>(std::vector<ENERGYTYPE>& dener,
                                        std::vector<double>& dforce_,
                                        std::vector<double>& dforce_mag_,
                                        std::vector<double>& dvirial,
                                        const std::vector<double>& dcoord_,
                                        const std::vector<double>& dspin_,
                                        const std::vector<int>& datype_,
                                        const std::vector<double>& dbox,
                                        const std::vector<double>& fparam,
                                        const std::vector<double>& aparam);

template void DeepSpin::compute<float>(std::vector<ENERGYTYPE>& dener,
                                       std::vector<float>& dforce_,
                                       std::vector<float>& dforce_mag_,
                                       std::vector<float>& dvirial,
                                       const std::vector<float>& dcoord_,
                                       const std::vector<float>& dspin_,
                                       const std::vector<int>& datype_,
                                       const std::vector<float>& dbox,
                                       const std::vector<float>& fparam,
                                       const std::vector<float>& aparam);

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
                       const std::vector<VALUETYPE>& aparam__) {
  std::vector<ENERGYTYPE> dener_;
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, false);
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
                       const std::vector<VALUETYPE>& aparam__) {
  std::vector<VALUETYPE> datom_energy_, datom_virial_;
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, false);
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
                                        const std::vector<double>& aparam_);

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
                                       const std::vector<float>& aparam_);

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
                                        const std::vector<double>& aparam_);

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
                                       const std::vector<float>& aparam_);

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
                       const std::vector<VALUETYPE>& aparam_) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               true);
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
                       const std::vector<VALUETYPE>& aparam_) {
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, fparam_, aparam_,
               true);
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
                                        const std::vector<double>& aparam);

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
                                       const std::vector<float>& aparam);

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
                                        const std::vector<double>& aparam);

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
                                       const std::vector<float>& aparam);

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
                       const std::vector<VALUETYPE>& aparam__) {
  std::vector<ENERGYTYPE> dener_;
  dp->computew(dener_, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, true);
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
                       const std::vector<VALUETYPE>& aparam__) {
  dp->computew(dener, dforce_, dforce_mag_, dvirial, datom_energy_,
               datom_virial_, dcoord_, dspin_, datype_, dbox, nghost, lmp_list,
               ago, fparam_, aparam__, true);
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
                                        const std::vector<double>& aparam_);

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
                                       const std::vector<float>& aparam_);

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
                                        const std::vector<double>& aparam_);

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
                                       const std::vector<float>& aparam_);

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
    const std::vector<VALUETYPE>& aparam_) {
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
                     aparam_);
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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);

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
    const std::vector<VALUETYPE>& aparam_) {
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
                     dcoord_, dspin_, datype_, dbox, fparam, aparam_);
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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);

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
    const std::vector<VALUETYPE>& aparam_) {
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
                     lmp_list, ago, fparam, aparam_);
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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);

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
    const std::vector<VALUETYPE>& aparam_) {
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
                     fparam, aparam_);
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
    const std::vector<double>& aparam);

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
    const std::vector<float>& aparam);
