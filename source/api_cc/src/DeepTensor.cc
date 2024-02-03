// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepTensor.h"

#include <memory>

#ifdef BUILD_TENSORFLOW
#include "DeepTensorTF.h"
#endif
#include "common.h"

using namespace deepmd;

DeepTensor::DeepTensor() : inited(false) {}

DeepTensor::DeepTensor(const std::string &model,
                       const int &gpu_rank,
                       const std::string &name_scope_)
    : inited(false) {
  init(model, gpu_rank, name_scope_);
}

DeepTensor::~DeepTensor() {}

void DeepTensor::init(const std::string &model,
                      const int &gpu_rank,
                      const std::string &name_scope_) {
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
    dt = std::make_shared<deepmd::DeepTensorTF>(model, gpu_rank, name_scope_);
#else
    throw deepmd::deepmd_exception("TensorFlow backend is not built.");
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

void DeepTensor::print_summary(const std::string &pre) const {
  deepmd::print_summary(pre);
}

template <typename VALUETYPE>
void DeepTensor::compute(std::vector<VALUETYPE> &dtensor_,
                         const std::vector<VALUETYPE> &dcoord_,
                         const std::vector<int> &datype_,
                         const std::vector<VALUETYPE> &dbox) {
  std::vector<VALUETYPE> force_, virial_, datom_tensor_, datom_virial_;
  dt->computew(dtensor_, force_, virial_, datom_tensor_, datom_virial_, dcoord_,
               datype_, dbox, false);
}

template void DeepTensor::compute<double>(std::vector<double> &dtensor_,
                                          const std::vector<double> &dcoord_,
                                          const std::vector<int> &datype_,
                                          const std::vector<double> &dbox);

template void DeepTensor::compute<float>(std::vector<float> &dtensor_,
                                         const std::vector<float> &dcoord_,
                                         const std::vector<int> &datype_,
                                         const std::vector<float> &dbox);

template <typename VALUETYPE>
void DeepTensor::compute(std::vector<VALUETYPE> &dtensor_,
                         const std::vector<VALUETYPE> &dcoord_,
                         const std::vector<int> &datype_,
                         const std::vector<VALUETYPE> &dbox,
                         const int nghost,
                         const InputNlist &lmp_list) {
  std::vector<VALUETYPE> force_, virial_, datom_tensor_, datom_virial_;
  dt->computew(dtensor_, force_, virial_, datom_tensor_, datom_virial_, dcoord_,
               datype_, dbox, nghost, lmp_list, false);
}

template void DeepTensor::compute<double>(std::vector<double> &dtensor_,
                                          const std::vector<double> &dcoord_,
                                          const std::vector<int> &datype_,
                                          const std::vector<double> &dbox,
                                          const int nghost,
                                          const InputNlist &lmp_list);

template void DeepTensor::compute<float>(std::vector<float> &dtensor_,
                                         const std::vector<float> &dcoord_,
                                         const std::vector<int> &datype_,
                                         const std::vector<float> &dbox,
                                         const int nghost,
                                         const InputNlist &lmp_list);

template <typename VALUETYPE>
void DeepTensor::compute(std::vector<VALUETYPE> &dglobal_tensor_,
                         std::vector<VALUETYPE> &dforce_,
                         std::vector<VALUETYPE> &dvirial_,
                         const std::vector<VALUETYPE> &dcoord_,
                         const std::vector<int> &datype_,
                         const std::vector<VALUETYPE> &dbox) {
  std::vector<VALUETYPE> datom_tensor_, datom_virial_;
  dt->computew(dglobal_tensor_, dforce_, dvirial_, datom_tensor_, datom_virial_,
               dcoord_, datype_, dbox, true);
}

template void DeepTensor::compute<double>(std::vector<double> &dglobal_tensor_,
                                          std::vector<double> &dforce_,
                                          std::vector<double> &dvirial_,
                                          const std::vector<double> &dcoord_,
                                          const std::vector<int> &datype_,
                                          const std::vector<double> &dbox);

template void DeepTensor::compute<float>(std::vector<float> &dglobal_tensor_,
                                         std::vector<float> &dforce_,
                                         std::vector<float> &dvirial_,
                                         const std::vector<float> &dcoord_,
                                         const std::vector<int> &datype_,
                                         const std::vector<float> &dbox);

template <typename VALUETYPE>
void DeepTensor::compute(std::vector<VALUETYPE> &dglobal_tensor_,
                         std::vector<VALUETYPE> &dforce_,
                         std::vector<VALUETYPE> &dvirial_,
                         const std::vector<VALUETYPE> &dcoord_,
                         const std::vector<int> &datype_,
                         const std::vector<VALUETYPE> &dbox,
                         const int nghost,
                         const InputNlist &lmp_list) {
  std::vector<VALUETYPE> datom_tensor_, datom_virial_;
  dt->computew(dglobal_tensor_, dforce_, dvirial_, datom_tensor_, datom_virial_,
               dcoord_, datype_, dbox, nghost, lmp_list, true);
}

template void DeepTensor::compute<double>(std::vector<double> &dglobal_tensor_,
                                          std::vector<double> &dforce_,
                                          std::vector<double> &dvirial_,
                                          const std::vector<double> &dcoord_,
                                          const std::vector<int> &datype_,
                                          const std::vector<double> &dbox,
                                          const int nghost,
                                          const InputNlist &lmp_list);

template void DeepTensor::compute<float>(std::vector<float> &dglobal_tensor_,
                                         std::vector<float> &dforce_,
                                         std::vector<float> &dvirial_,
                                         const std::vector<float> &dcoord_,
                                         const std::vector<int> &datype_,
                                         const std::vector<float> &dbox,
                                         const int nghost,
                                         const InputNlist &lmp_list);

template <typename VALUETYPE>
void DeepTensor::compute(std::vector<VALUETYPE> &dglobal_tensor_,
                         std::vector<VALUETYPE> &dforce_,
                         std::vector<VALUETYPE> &dvirial_,
                         std::vector<VALUETYPE> &datom_tensor_,
                         std::vector<VALUETYPE> &datom_virial_,
                         const std::vector<VALUETYPE> &dcoord_,
                         const std::vector<int> &datype_,
                         const std::vector<VALUETYPE> &dbox) {
  dt->computew(dglobal_tensor_, dforce_, dvirial_, datom_tensor_, datom_virial_,
               dcoord_, datype_, dbox, true);
}

template void DeepTensor::compute<double>(std::vector<double> &dglobal_tensor_,
                                          std::vector<double> &dforce_,
                                          std::vector<double> &dvirial_,
                                          std::vector<double> &datom_tensor_,
                                          std::vector<double> &datom_virial_,
                                          const std::vector<double> &dcoord_,
                                          const std::vector<int> &datype_,
                                          const std::vector<double> &dbox);

template void DeepTensor::compute<float>(std::vector<float> &dglobal_tensor_,
                                         std::vector<float> &dforce_,
                                         std::vector<float> &dvirial_,
                                         std::vector<float> &datom_tensor_,
                                         std::vector<float> &datom_virial_,
                                         const std::vector<float> &dcoord_,
                                         const std::vector<int> &datype_,
                                         const std::vector<float> &dbox);

template <typename VALUETYPE>
void DeepTensor::compute(std::vector<VALUETYPE> &dglobal_tensor_,
                         std::vector<VALUETYPE> &dforce_,
                         std::vector<VALUETYPE> &dvirial_,
                         std::vector<VALUETYPE> &datom_tensor_,
                         std::vector<VALUETYPE> &datom_virial_,
                         const std::vector<VALUETYPE> &dcoord_,
                         const std::vector<int> &datype_,
                         const std::vector<VALUETYPE> &dbox,
                         const int nghost,
                         const InputNlist &lmp_list) {
  dt->computew(dglobal_tensor_, dforce_, dvirial_, datom_tensor_, datom_virial_,
               dcoord_, datype_, dbox, nghost, lmp_list, true);
}

template void DeepTensor::compute<double>(std::vector<double> &dglobal_tensor_,
                                          std::vector<double> &dforce_,
                                          std::vector<double> &dvirial_,
                                          std::vector<double> &datom_tensor_,
                                          std::vector<double> &datom_virial_,
                                          const std::vector<double> &dcoord_,
                                          const std::vector<int> &datype_,
                                          const std::vector<double> &dbox,
                                          const int nghost,
                                          const InputNlist &lmp_list);

template void DeepTensor::compute<float>(std::vector<float> &dglobal_tensor_,
                                         std::vector<float> &dforce_,
                                         std::vector<float> &dvirial_,
                                         std::vector<float> &datom_tensor_,
                                         std::vector<float> &datom_virial_,
                                         const std::vector<float> &dcoord_,
                                         const std::vector<int> &datype_,
                                         const std::vector<float> &dbox,
                                         const int nghost,
                                         const InputNlist &lmp_list);

void DeepTensor::get_type_map(std::string &type_map) {
  dt->get_type_map(type_map);
}

double DeepTensor::cutoff() const { return dt->cutoff(); }

int DeepTensor::output_dim() const { return dt->output_dim(); }

const std::vector<int> &DeepTensor::sel_types() const {
  return dt->sel_types();
}

int DeepTensor::numb_types() const { return dt->numb_types(); }
