// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DataModifier.h"

#ifdef BUILD_TENSORFLOW
#include "DataModifierTF.h"
#endif
#include "common.h"

using namespace deepmd;

DipoleChargeModifier::DipoleChargeModifier() : inited(false) {}

DipoleChargeModifier::DipoleChargeModifier(const std::string& model,
                                           const int& gpu_rank,
                                           const std::string& name_scope_)
    : inited(false) {
  init(model, gpu_rank, name_scope_);
}

DipoleChargeModifier::~DipoleChargeModifier(){};

void DipoleChargeModifier::init(const std::string& model,
                                const int& gpu_rank,
                                const std::string& name_scope_) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  // TODO: To implement detect_backend
  DPBackend backend = deepmd::DPBackend::TensorFlow;
  if (deepmd::DPBackend::TensorFlow == backend) {
#ifdef BUILD_TENSORFLOW
    dcm = std::make_shared<deepmd::DipoleChargeModifierTF>(model, gpu_rank,
                                                           name_scope_);
#else
    throw deepmd::deepmd_exception("TensorFlow backend is not built");
#endif
  } else if (deepmd::DPBackend::PyTorch == backend) {
    throw deepmd::deepmd_exception("PyTorch backend is not supported yet");
  } else if (deepmd::DPBackend::Paddle == backend) {
    throw deepmd::deepmd_exception("PaddlePaddle backend is not supported yet");
  } else {
    throw deepmd::deepmd_exception("Unknown file type");
  }
  inited = true;
}

template <typename VALUETYPE>
void DipoleChargeModifier::compute(
    std::vector<VALUETYPE>& dfcorr_,
    std::vector<VALUETYPE>& dvcorr_,
    const std::vector<VALUETYPE>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<VALUETYPE>& delef_,
    const int nghost,
    const InputNlist& lmp_list) {
  dcm->computew(dfcorr_, dvcorr_, dcoord_, datype_, dbox, pairs, delef_, nghost,
                lmp_list);
}

template void DipoleChargeModifier::compute<double>(
    std::vector<double>& dfcorr_,
    std::vector<double>& dvcorr_,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<double>& delef_,
    const int nghost,
    const InputNlist& lmp_list);

template void DipoleChargeModifier::compute<float>(
    std::vector<float>& dfcorr_,
    std::vector<float>& dvcorr_,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<float>& delef_,
    const int nghost,
    const InputNlist& lmp_list);

void DipoleChargeModifier::print_summary(const std::string& pre) const {
  deepmd::print_summary(pre);
}

double DipoleChargeModifier::cutoff() const { return dcm->cutoff(); }

int DipoleChargeModifier::numb_types() const { return dcm->numb_types(); }

const std::vector<int>& DipoleChargeModifier::sel_types() const {
  return dcm->sel_types();
}
