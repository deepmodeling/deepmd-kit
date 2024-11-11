// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DeepBaseModel.h"

#include <memory>
#include <stdexcept>

#include "AtomMap.h"
#include "common.h"
#include "device.h"

using namespace deepmd;

DeepBaseModel::DeepBaseModel() : inited(false) {}

DeepBaseModel::~DeepBaseModel() {}

void DeepBaseModel::print_summary(const std::string& pre) const {
  deepmd::print_summary(pre);
}

double DeepBaseModel::cutoff() const { return dpbase->cutoff(); }

int DeepBaseModel::numb_types() const { return dpbase->numb_types(); }

int DeepBaseModel::numb_types_spin() const { return dpbase->numb_types_spin(); }

int DeepBaseModel::dim_fparam() const { return dpbase->dim_fparam(); }

int DeepBaseModel::dim_aparam() const { return dpbase->dim_aparam(); }

void DeepBaseModel::get_type_map(std::string& type_map) {
  dpbase->get_type_map(type_map);
}

bool DeepBaseModel::is_aparam_nall() const { return dpbase->is_aparam_nall(); }

DeepBaseModelDevi::DeepBaseModelDevi() : inited(false), numb_models(0) {}

// DeepBaseModelDevi::DeepBaseModelDevi(
//     const std::vector<std::string>& models,
//     const int& gpu_rank,
//     const std::vector<std::string>& file_contents)
//     : inited(false), numb_models(0) {
//   init(models, gpu_rank, file_contents);
// }

DeepBaseModelDevi::~DeepBaseModelDevi() {}

// void DeepBaseModelDevi::init(const std::vector<std::string>& models,
//                             const int& gpu_rank,
//                             const std::vector<std::string>& file_contents) {
//   if (inited) {
//     std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
//                  "nothing at the second call of initializer"
//               << std::endl;
//     return;
//   }
//   numb_models = models.size();
//   if (numb_models == 0) {
//     throw deepmd::deepmd_exception("no model is specified");
//   }
//   dps.resize(numb_models);
//   for (unsigned int ii = 0; ii < numb_models; ++ii) {
//     dps[ii].init(models[ii], gpu_rank,
//                  file_contents.size() > ii ? file_contents[ii] : "");
//   }
//   inited = true;
// }

template <typename VALUETYPE>
void DeepBaseModelDevi::compute_avg(VALUETYPE& dener,
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

template void DeepBaseModelDevi::compute_avg<double>(
    double& dener, const std::vector<double>& all_energy);

template void DeepBaseModelDevi::compute_avg<float>(
    float& dener, const std::vector<float>& all_energy);

template <typename VALUETYPE>
void DeepBaseModelDevi::compute_avg(
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

template void DeepBaseModelDevi::compute_avg<double>(
    std::vector<double>& avg, const std::vector<std::vector<double>>& xx);

template void DeepBaseModelDevi::compute_avg<float>(
    std::vector<float>& avg, const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepBaseModelDevi::compute_std(
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

template void DeepBaseModelDevi::compute_std<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx,
    const int& stride);

template void DeepBaseModelDevi::compute_std<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx,
    const int& stride);

template <typename VALUETYPE>
void DeepBaseModelDevi::compute_std_e(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  compute_std(std, avg, xx, 1);
}

template void DeepBaseModelDevi::compute_std_e<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx);

template void DeepBaseModelDevi::compute_std_e<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepBaseModelDevi::compute_std_f(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  compute_std(std, avg, xx, 3);
}

template void DeepBaseModelDevi::compute_std_f<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx);

template void DeepBaseModelDevi::compute_std_f<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepBaseModelDevi::compute_relative_std(std::vector<VALUETYPE>& std,
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

template void DeepBaseModelDevi::compute_relative_std<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const double eps,
    const int& stride);

template void DeepBaseModelDevi::compute_relative_std<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const float eps,
    const int& stride);

template <typename VALUETYPE>
void DeepBaseModelDevi::compute_relative_std_f(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const VALUETYPE eps) {
  compute_relative_std(std, avg, eps, 3);
}

template void DeepBaseModelDevi::compute_relative_std_f<double>(
    std::vector<double>& std, const std::vector<double>& avg, const double eps);

template void DeepBaseModelDevi::compute_relative_std_f<float>(
    std::vector<float>& std, const std::vector<float>& avg, const float eps);
