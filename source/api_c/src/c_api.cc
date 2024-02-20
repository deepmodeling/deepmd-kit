// SPDX-License-Identifier: LGPL-3.0-or-later
#include "c_api.h"

#include <numeric>
#include <string>
#include <vector>

#include "DataModifier.h"
#include "DeepPot.h"
#include "DeepTensor.h"
#include "c_api_internal.h"
#include "common.h"

extern "C" {

DP_Nlist::DP_Nlist() {}
DP_Nlist::DP_Nlist(deepmd::InputNlist& nl) : nl(nl) {}

DP_Nlist* DP_NewNlist(int inum_,
                      int* ilist_,
                      int* numneigh_,
                      int** firstneigh_) {
  DP_NEW_OK(DP_Nlist,
            deepmd::InputNlist nl(inum_, ilist_, numneigh_, firstneigh_);
            DP_Nlist* new_nl = new DP_Nlist(nl); return new_nl;)
}

void DP_DeleteNlist(DP_Nlist* nl) { delete nl; }

DP_DeepPot::DP_DeepPot() {}
DP_DeepPot::DP_DeepPot(deepmd::DeepPot& dp) : dp(dp) {
  dfparam = dp.dim_fparam();
  daparam = dp.dim_aparam();
  aparam_nall = dp.is_aparam_nall();
}

DP_DeepPot* DP_NewDeepPot(const char* c_model) {
  std::string model(c_model);
  DP_NEW_OK(DP_DeepPot, deepmd::DeepPot dp(model);
            DP_DeepPot* new_dp = new DP_DeepPot(dp); return new_dp;)
}

DP_DeepPot* DP_NewDeepPotWithParam(const char* c_model,
                                   const int gpu_rank,
                                   const char* c_file_content) {
  std::string model(c_model);
  std::string file_content(c_file_content);
  DP_NEW_OK(DP_DeepPot,
            if (file_content.size() > 0) throw deepmd::deepmd_exception(
                "file_content is broken in DP_NewDeepPotWithParam. Use "
                "DP_NewDeepPotWithParam2 instead.");
            deepmd::DeepPot dp(model, gpu_rank, file_content);
            DP_DeepPot* new_dp = new DP_DeepPot(dp); return new_dp;)
}

DP_DeepPot* DP_NewDeepPotWithParam2(const char* c_model,
                                    const int gpu_rank,
                                    const char* c_file_content,
                                    const int size_file_content) {
  std::string model(c_model);
  std::string file_content(c_file_content, c_file_content + size_file_content);
  DP_NEW_OK(DP_DeepPot, deepmd::DeepPot dp(model, gpu_rank, file_content);
            DP_DeepPot* new_dp = new DP_DeepPot(dp); return new_dp;)
}

void DP_DeleteDeepPot(DP_DeepPot* dp) { delete dp; }

DP_DeepPotModelDevi::DP_DeepPotModelDevi() {}
DP_DeepPotModelDevi::DP_DeepPotModelDevi(deepmd::DeepPotModelDevi& dp)
    : dp(dp) {
  dfparam = dp.dim_fparam();
  daparam = dp.dim_aparam();
  aparam_nall = dp.is_aparam_nall();
}

DP_DeepPotModelDevi* DP_NewDeepPotModelDevi(const char** c_models,
                                            int n_models) {
  std::vector<std::string> model(c_models, c_models + n_models);
  DP_NEW_OK(DP_DeepPotModelDevi, deepmd::DeepPotModelDevi dp(model);
            DP_DeepPotModelDevi* new_dp = new DP_DeepPotModelDevi(dp);
            return new_dp;)
}

DP_DeepPotModelDevi* DP_NewDeepPotModelDeviWithParam(
    const char** c_models,
    const int n_models,
    const int gpu_rank,
    const char** c_file_contents,
    const int n_file_contents,
    const int* size_file_contents) {
  std::vector<std::string> model(c_models, c_models + n_models);
  std::vector<std::string> file_content;
  file_content.reserve(n_file_contents);
  for (int ii = 0; ii < n_file_contents; ++ii) {
    file_content.push_back(std::string(
        c_file_contents[ii], c_file_contents[ii] + size_file_contents[ii]));
  }
  DP_NEW_OK(DP_DeepPotModelDevi,
            deepmd::DeepPotModelDevi dp(model, gpu_rank, file_content);
            DP_DeepPotModelDevi* new_dp = new DP_DeepPotModelDevi(dp);
            return new_dp;)
}

void DP_DeleteDeepPotModelDevi(DP_DeepPotModelDevi* dp) { delete dp; }

DP_DeepTensor::DP_DeepTensor() {}
DP_DeepTensor::DP_DeepTensor(deepmd::DeepTensor& dt) : dt(dt) {}

DP_DeepTensor* DP_NewDeepTensor(const char* c_model) {
  std::string model(c_model);
  DP_NEW_OK(DP_DeepTensor, deepmd::DeepTensor dt(model);
            DP_DeepTensor* new_dt = new DP_DeepTensor(dt); return new_dt;)
}

DP_DeepTensor* DP_NewDeepTensorWithParam(const char* c_model,
                                         const int gpu_rank,
                                         const char* c_name_scope) {
  std::string model(c_model);
  std::string name_scope(c_name_scope);
  DP_NEW_OK(DP_DeepTensor, deepmd::DeepTensor dt(model, gpu_rank, name_scope);
            DP_DeepTensor* new_dt = new DP_DeepTensor(dt); return new_dt;)
}

void DP_DeleteDeepTensor(DP_DeepTensor* dt) { delete dt; }

DP_DipoleChargeModifier::DP_DipoleChargeModifier() {}
DP_DipoleChargeModifier::DP_DipoleChargeModifier(
    deepmd::DipoleChargeModifier& dcm)
    : dcm(dcm) {}

DP_DipoleChargeModifier* DP_NewDipoleChargeModifier(const char* c_model) {
  std::string model(c_model);
  DP_NEW_OK(DP_DipoleChargeModifier, deepmd::DipoleChargeModifier dcm(model);
            DP_DipoleChargeModifier* new_dcm = new DP_DipoleChargeModifier(dcm);
            return new_dcm;)
}

DP_DipoleChargeModifier* DP_NewDipoleChargeModifierWithParam(
    const char* c_model, const int gpu_rank, const char* c_name_scope) {
  std::string model(c_model);
  std::string name_scope(c_name_scope);
  DP_NEW_OK(DP_DipoleChargeModifier,
            deepmd::DipoleChargeModifier dcm(model, gpu_rank, name_scope);
            DP_DipoleChargeModifier* new_dcm = new DP_DipoleChargeModifier(dcm);
            return new_dcm;)
}

void DP_DeleteDipoleChargeModifier(DP_DipoleChargeModifier* dcm) { delete dcm; }

}  // extern "C"

template <typename VALUETYPE>
inline void DP_DeepPotCompute_variant(DP_DeepPot* dp,
                                      const int nframes,
                                      const int natoms,
                                      const VALUETYPE* coord,
                                      const int* atype,
                                      const VALUETYPE* cell,
                                      const VALUETYPE* fparam,
                                      const VALUETYPE* aparam,
                                      double* energy,
                                      VALUETYPE* force,
                                      VALUETYPE* virial,
                                      VALUETYPE* atomic_energy,
                                      VALUETYPE* atomic_virial) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + nframes * natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + nframes * 9);
  }
  std::vector<VALUETYPE> fparam_;
  if (fparam) {
    fparam_.assign(fparam, fparam + nframes * dp->dfparam);
  }
  std::vector<VALUETYPE> aparam_;
  if (aparam) {
    aparam_.assign(aparam, aparam + nframes * natoms * dp->daparam);
  }
  std::vector<double> e;
  std::vector<VALUETYPE> f, v, ae, av;

  DP_REQUIRES_OK(dp, dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_,
                                    fparam_, aparam_));
  // copy from C++ vectors to C arrays, if not NULL pointer
  if (energy) {
    std::copy(e.begin(), e.end(), energy);
  }
  if (force) {
    std::copy(f.begin(), f.end(), force);
  }
  if (virial) {
    std::copy(v.begin(), v.end(), virial);
  }
  if (atomic_energy) {
    std::copy(ae.begin(), ae.end(), atomic_energy);
  }
  if (atomic_virial) {
    std::copy(av.begin(), av.end(), atomic_virial);
  }
}

template void DP_DeepPotCompute_variant<double>(DP_DeepPot* dp,
                                                const int nframes,
                                                const int natoms,
                                                const double* coord,
                                                const int* atype,
                                                const double* cell,
                                                const double* fparam,
                                                const double* aparam,
                                                double* energy,
                                                double* force,
                                                double* virial,
                                                double* atomic_energy,
                                                double* atomic_virial);

template void DP_DeepPotCompute_variant<float>(DP_DeepPot* dp,
                                               const int nframes,
                                               const int natoms,
                                               const float* coord,
                                               const int* atype,
                                               const float* cell,
                                               const float* fparam,
                                               const float* aparam,
                                               double* energy,
                                               float* force,
                                               float* virial,
                                               float* atomic_energy,
                                               float* atomic_virial);

template <typename VALUETYPE>
inline void DP_DeepPotComputeNList_variant(DP_DeepPot* dp,
                                           const int nframes,
                                           const int natoms,
                                           const VALUETYPE* coord,
                                           const int* atype,
                                           const VALUETYPE* cell,
                                           const int nghost,
                                           const DP_Nlist* nlist,
                                           const int ago,
                                           const VALUETYPE* fparam,
                                           const VALUETYPE* aparam,
                                           double* energy,
                                           VALUETYPE* force,
                                           VALUETYPE* virial,
                                           VALUETYPE* atomic_energy,
                                           VALUETYPE* atomic_virial) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + nframes * natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + nframes * 9);
  }
  std::vector<VALUETYPE> fparam_;
  if (fparam) {
    fparam_.assign(fparam, fparam + nframes * dp->dfparam);
  }
  std::vector<VALUETYPE> aparam_;
  if (aparam) {
    aparam_.assign(aparam,
                   aparam + nframes *
                                (dp->aparam_nall ? natoms : (natoms - nghost)) *
                                dp->daparam);
  }
  std::vector<double> e;
  std::vector<VALUETYPE> f, v, ae, av;

  DP_REQUIRES_OK(dp, dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_,
                                    nghost, nlist->nl, ago, fparam_, aparam_));
  // copy from C++ vectors to C arrays, if not NULL pointer
  if (energy) {
    std::copy(e.begin(), e.end(), energy);
  }
  if (force) {
    std::copy(f.begin(), f.end(), force);
  }
  if (virial) {
    std::copy(v.begin(), v.end(), virial);
  }
  if (atomic_energy) {
    std::copy(ae.begin(), ae.end(), atomic_energy);
  }
  if (atomic_virial) {
    std::copy(av.begin(), av.end(), atomic_virial);
  }
}

template void DP_DeepPotComputeNList_variant<double>(DP_DeepPot* dp,
                                                     const int nframes,
                                                     const int natoms,
                                                     const double* coord,
                                                     const int* atype,
                                                     const double* cell,
                                                     const int nghost,
                                                     const DP_Nlist* nlist,
                                                     const int ago,
                                                     const double* fparam,
                                                     const double* aparam,
                                                     double* energy,
                                                     double* force,
                                                     double* virial,
                                                     double* atomic_energy,
                                                     double* atomic_virial);

template void DP_DeepPotComputeNList_variant<float>(DP_DeepPot* dp,
                                                    const int nframes,
                                                    const int natoms,
                                                    const float* coord,
                                                    const int* atype,
                                                    const float* cell,
                                                    const int nghost,
                                                    const DP_Nlist* nlist,
                                                    const int ago,
                                                    const float* fparam,
                                                    const float* aparam,
                                                    double* energy,
                                                    float* force,
                                                    float* virial,
                                                    float* atomic_energy,
                                                    float* atomic_virial);

template <typename VALUETYPE>
inline void DP_DeepPotComputeMixedType_variant(DP_DeepPot* dp,
                                               const int nframes,
                                               const int natoms,
                                               const VALUETYPE* coord,
                                               const int* atype,
                                               const VALUETYPE* cell,
                                               const VALUETYPE* fparam,
                                               const VALUETYPE* aparam,
                                               double* energy,
                                               VALUETYPE* force,
                                               VALUETYPE* virial,
                                               VALUETYPE* atomic_energy,
                                               VALUETYPE* atomic_virial) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + nframes * natoms * 3);
  std::vector<int> atype_(atype, atype + nframes * natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + nframes * 9);
  }
  std::vector<VALUETYPE> fparam_;
  if (fparam) {
    fparam_.assign(fparam, fparam + nframes * dp->dfparam);
  }
  std::vector<VALUETYPE> aparam_;
  if (aparam) {
    aparam_.assign(aparam, aparam + nframes * natoms * dp->daparam);
  }
  std::vector<double> e;
  std::vector<VALUETYPE> f, v, ae, av;

  DP_REQUIRES_OK(
      dp, dp->dp.compute_mixed_type(e, f, v, ae, av, nframes, coord_, atype_,
                                    cell_, fparam_, aparam_));
  // copy from C++ vectors to C arrays, if not NULL pointer
  if (energy) {
    std::copy(e.begin(), e.end(), energy);
  }
  if (force) {
    std::copy(f.begin(), f.end(), force);
  }
  if (virial) {
    std::copy(v.begin(), v.end(), virial);
  }
  if (atomic_energy) {
    std::copy(ae.begin(), ae.end(), atomic_energy);
  }
  if (atomic_virial) {
    std::copy(av.begin(), av.end(), atomic_virial);
  }
}

template void DP_DeepPotComputeMixedType_variant<double>(DP_DeepPot* dp,
                                                         const int nframes,
                                                         const int natoms,
                                                         const double* coord,
                                                         const int* atype,
                                                         const double* cell,
                                                         const double* fparam,
                                                         const double* aparam,
                                                         double* energy,
                                                         double* force,
                                                         double* virial,
                                                         double* atomic_energy,
                                                         double* atomic_virial);

template void DP_DeepPotComputeMixedType_variant<float>(DP_DeepPot* dp,
                                                        const int nframes,
                                                        const int natoms,
                                                        const float* coord,
                                                        const int* atype,
                                                        const float* cell,
                                                        const float* fparam,
                                                        const float* aparam,
                                                        double* energy,
                                                        float* force,
                                                        float* virial,
                                                        float* atomic_energy,
                                                        float* atomic_virial);

template <typename VALUETYPE>
inline void flatten_vector(std::vector<VALUETYPE>& onedv,
                           const std::vector<std::vector<VALUETYPE>>& twodv) {
  onedv.clear();
  for (size_t ii = 0; ii < twodv.size(); ++ii) {
    onedv.insert(onedv.end(), twodv[ii].begin(), twodv[ii].end());
  }
}

template <typename VALUETYPE>
void DP_DeepPotModelDeviComputeNList_variant(DP_DeepPotModelDevi* dp,
                                             const int nframes,
                                             const int natoms,
                                             const VALUETYPE* coord,
                                             const int* atype,
                                             const VALUETYPE* cell,
                                             const int nghost,
                                             const DP_Nlist* nlist,
                                             const int ago,
                                             const VALUETYPE* fparam,
                                             const VALUETYPE* aparam,
                                             double* energy,
                                             VALUETYPE* force,
                                             VALUETYPE* virial,
                                             VALUETYPE* atomic_energy,
                                             VALUETYPE* atomic_virial) {
  if (nframes > 1) {
    throw std::runtime_error("nframes > 1 not supported yet");
  }
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + 9);
  }
  std::vector<VALUETYPE> fparam_;
  if (fparam) {
    fparam_.assign(fparam, fparam + dp->dfparam);
  }
  std::vector<VALUETYPE> aparam_;
  if (aparam) {
    aparam_.assign(
        aparam,
        aparam + (dp->aparam_nall ? natoms : (natoms - nghost)) * dp->daparam);
  }
  // different from DeepPot
  std::vector<double> e;
  std::vector<std::vector<VALUETYPE>> f, v, ae, av;

  DP_REQUIRES_OK(dp, dp->dp.compute(e, f, v, ae, av, coord_, atype_, cell_,
                                    nghost, nlist->nl, ago, fparam_, aparam_));
  // 2D vector to 2D array, flatten first
  if (energy) {
    std::copy(e.begin(), e.end(), energy);
  }
  if (force) {
    std::vector<VALUETYPE> f_flat;
    flatten_vector(f_flat, f);
    std::copy(f_flat.begin(), f_flat.end(), force);
  }
  if (virial) {
    std::vector<VALUETYPE> v_flat;
    flatten_vector(v_flat, v);
    std::copy(v_flat.begin(), v_flat.end(), virial);
  }
  if (atomic_energy) {
    std::vector<VALUETYPE> ae_flat;
    flatten_vector(ae_flat, ae);
    std::copy(ae_flat.begin(), ae_flat.end(), atomic_energy);
  }
  if (atomic_virial) {
    std::vector<VALUETYPE> av_flat;
    flatten_vector(av_flat, av);
    std::copy(av_flat.begin(), av_flat.end(), atomic_virial);
  }
}

template void DP_DeepPotModelDeviComputeNList_variant<double>(
    DP_DeepPotModelDevi* dp,
    const int nframes,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    const int nghost,
    const DP_Nlist* nlist,
    const int ago,
    const double* fparam,
    const double* aparam,
    double* energy,
    double* force,
    double* virial,
    double* atomic_energy,
    double* atomic_virial);

template void DP_DeepPotModelDeviComputeNList_variant<float>(
    DP_DeepPotModelDevi* dp,
    const int nframes,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    const int nghost,
    const DP_Nlist* nlist,
    const int ago,
    const float* fparam,
    const float* aparam,
    double* energy,
    float* force,
    float* virial,
    float* atomic_energy,
    float* atomic_virial);

template <typename VALUETYPE>
inline void DP_DeepTensorComputeTensor_variant(DP_DeepTensor* dt,
                                               const int natoms,
                                               const VALUETYPE* coord,
                                               const int* atype,
                                               const VALUETYPE* cell,
                                               VALUETYPE** tensor,
                                               int* size) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + 9);
  }
  std::vector<VALUETYPE> t;

  DP_REQUIRES_OK(dt, dt->dt.compute(t, coord_, atype_, cell_));
  // do not know the size of tensor in advance...
  *tensor = new VALUETYPE[t.size()];
  std::copy(t.begin(), t.end(), *tensor);
  *size = t.size();
}

template void DP_DeepTensorComputeTensor_variant<double>(DP_DeepTensor* dt,
                                                         const int natoms,
                                                         const double* coord,
                                                         const int* atype,
                                                         const double* cell,
                                                         double** tensor,
                                                         int* size);

template void DP_DeepTensorComputeTensor_variant<float>(DP_DeepTensor* dt,
                                                        const int natoms,
                                                        const float* coord,
                                                        const int* atype,
                                                        const float* cell,
                                                        float** tensor,
                                                        int* size);

template <typename VALUETYPE>
inline void DP_DeepTensorComputeTensorNList_variant(DP_DeepTensor* dt,
                                                    const int natoms,
                                                    const VALUETYPE* coord,
                                                    const int* atype,
                                                    const VALUETYPE* cell,
                                                    const int nghost,
                                                    const DP_Nlist* nlist,
                                                    VALUETYPE** tensor,
                                                    int* size) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + 9);
  }
  std::vector<VALUETYPE> t;

  DP_REQUIRES_OK(dt,
                 dt->dt.compute(t, coord_, atype_, cell_, nghost, nlist->nl));
  // do not know the size of tensor in advance...
  *tensor = new VALUETYPE[t.size()];
  std::copy(t.begin(), t.end(), *tensor);
  *size = t.size();
}

template void DP_DeepTensorComputeTensorNList_variant<double>(
    DP_DeepTensor* dt,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    const int nghost,
    const DP_Nlist* nlist,
    double** tensor,
    int* size);

template void DP_DeepTensorComputeTensorNList_variant<float>(
    DP_DeepTensor* dt,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    const int nghost,
    const DP_Nlist* nlist,
    float** tensor,
    int* size);

template <typename VALUETYPE>
inline void DP_DeepTensorCompute_variant(DP_DeepTensor* dt,
                                         const int natoms,
                                         const VALUETYPE* coord,
                                         const int* atype,
                                         const VALUETYPE* cell,
                                         VALUETYPE* global_tensor,
                                         VALUETYPE* force,
                                         VALUETYPE* virial,
                                         VALUETYPE** atomic_tensor,
                                         VALUETYPE* atomic_virial,
                                         int* size_at) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + 9);
  }
  std::vector<VALUETYPE> t, f, v, at, av;

  DP_REQUIRES_OK(dt, dt->dt.compute(t, f, v, at, av, coord_, atype_, cell_));
  // copy from C++ vectors to C arrays, if not NULL pointer
  if (global_tensor) {
    std::copy(t.begin(), t.end(), global_tensor);
  }
  if (force) {
    std::copy(f.begin(), f.end(), force);
  }
  if (virial) {
    std::copy(v.begin(), v.end(), virial);
  }
  if (atomic_virial) {
    std::copy(av.begin(), av.end(), atomic_virial);
  }
  // do not know the size of atomic tensor in advance...
  if (atomic_tensor) {
    *atomic_tensor = new VALUETYPE[at.size()];
    std::copy(at.begin(), at.end(), *atomic_tensor);
  }
  if (size_at) {
    *size_at = at.size();
  }
}

template void DP_DeepTensorCompute_variant<double>(DP_DeepTensor* dt,
                                                   const int natoms,
                                                   const double* coord,
                                                   const int* atype,
                                                   const double* cell,
                                                   double* global_tensor,
                                                   double* force,
                                                   double* virial,
                                                   double** atomic_tensor,
                                                   double* atomic_virial,
                                                   int* size_at);

template void DP_DeepTensorCompute_variant<float>(DP_DeepTensor* dt,
                                                  const int natoms,
                                                  const float* coord,
                                                  const int* atype,
                                                  const float* cell,
                                                  float* global_tensor,
                                                  float* force,
                                                  float* virial,
                                                  float** atomic_tensor,
                                                  float* atomic_virial,
                                                  int* size_at);

template <typename VALUETYPE>
inline void DP_DeepTensorComputeNList_variant(DP_DeepTensor* dt,
                                              const int natoms,
                                              const VALUETYPE* coord,
                                              const int* atype,
                                              const VALUETYPE* cell,
                                              const int nghost,
                                              const DP_Nlist* nlist,
                                              VALUETYPE* global_tensor,
                                              VALUETYPE* force,
                                              VALUETYPE* virial,
                                              VALUETYPE** atomic_tensor,
                                              VALUETYPE* atomic_virial,
                                              int* size_at) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + 9);
  }
  std::vector<VALUETYPE> t, f, v, at, av;

  DP_REQUIRES_OK(dt, dt->dt.compute(t, f, v, at, av, coord_, atype_, cell_,
                                    nghost, nlist->nl));
  // copy from C++ vectors to C arrays, if not NULL pointer
  if (global_tensor) {
    std::copy(t.begin(), t.end(), global_tensor);
  }
  if (force) {
    std::copy(f.begin(), f.end(), force);
  }
  if (virial) {
    std::copy(v.begin(), v.end(), virial);
  }
  if (atomic_virial) {
    std::copy(av.begin(), av.end(), atomic_virial);
  }
  // do not know the size of atomic tensor in advance...
  if (atomic_tensor) {
    *atomic_tensor = new VALUETYPE[at.size()];
    std::copy(at.begin(), at.end(), *atomic_tensor);
  }
  if (size_at) {
    *size_at = at.size();
  }
}

template void DP_DeepTensorComputeNList_variant<double>(DP_DeepTensor* dt,
                                                        const int natoms,
                                                        const double* coord,
                                                        const int* atype,
                                                        const double* cell,
                                                        const int nghost,
                                                        const DP_Nlist* nlist,
                                                        double* global_tensor,
                                                        double* force,
                                                        double* virial,
                                                        double** atomic_tensor,
                                                        double* atomic_virial,
                                                        int* size_at);

template void DP_DeepTensorComputeNList_variant<float>(DP_DeepTensor* dt,
                                                       const int natoms,
                                                       const float* coord,
                                                       const int* atype,
                                                       const float* cell,
                                                       const int nghost,
                                                       const DP_Nlist* nlist,
                                                       float* global_tensor,
                                                       float* force,
                                                       float* virial,
                                                       float** atomic_tensor,
                                                       float* atomic_virial,
                                                       int* size_at);

template <typename VALUETYPE>
inline void DP_DipoleChargeModifierComputeNList_variant(
    DP_DipoleChargeModifier* dcm,
    const int natoms,
    const VALUETYPE* coord,
    const int* atype,
    const VALUETYPE* cell,
    const int* pairs,
    const int npairs,
    const VALUETYPE* delef,
    const int nghost,
    const DP_Nlist* nlist,
    VALUETYPE* dfcorr_,
    VALUETYPE* dvcorr_) {
  // init C++ vectors from C arrays
  std::vector<VALUETYPE> coord_(coord, coord + natoms * 3);
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<VALUETYPE> cell_;
  if (cell) {
    // pbc
    cell_.assign(cell, cell + 9);
  }
  // pairs
  std::vector<std::pair<int, int>> pairs_;
  for (int i = 0; i < npairs; i++) {
    pairs_.push_back(std::make_pair(pairs[i * 2], pairs[i * 2 + 1]));
  }
  std::vector<VALUETYPE> delef_(delef, delef + (natoms - nghost) * 3);
  std::vector<VALUETYPE> df, dv;

  DP_REQUIRES_OK(dcm, dcm->dcm.compute(df, dv, coord_, atype_, cell_, pairs_,
                                       delef_, nghost, nlist->nl));
  // copy from C++ vectors to C arrays, if not NULL pointer
  if (dfcorr_) {
    std::copy(df.begin(), df.end(), dfcorr_);
  }
  if (dvcorr_) {
    std::copy(dv.begin(), dv.end(), dvcorr_);
  }
}

template void DP_DipoleChargeModifierComputeNList_variant<double>(
    DP_DipoleChargeModifier* dcm,
    const int natoms,
    const double* coord,
    const int* atype,
    const double* cell,
    const int* pairs,
    const int npairs,
    const double* delef,
    const int nghost,
    const DP_Nlist* nlist,
    double* dfcorr_,
    double* dvcorr_);

template void DP_DipoleChargeModifierComputeNList_variant<float>(
    DP_DipoleChargeModifier* dcm,
    const int natoms,
    const float* coord,
    const int* atype,
    const float* cell,
    const int* pairs,
    const int npairs,
    const float* delef,
    const int nghost,
    const DP_Nlist* nlist,
    float* dfcorr_,
    float* dvcorr_);

/**
 * @brief Convert std::string to const char
 * @param[in] str std::string to be converted
 * @return const char*
 */
const char* string_to_char(std::string& str) {
  // copy from string to char*
  const std::string::size_type size = str.size();
  // +1 for '\0'
  char* buffer = new char[size + 1];
  std::copy(str.begin(), str.end(), buffer);
  buffer[size] = '\0';
  return buffer;
}

extern "C" {

const char* DP_NlistCheckOK(DP_Nlist* nlist) {
  return string_to_char(nlist->exception);
}

void DP_DeepPotCompute(DP_DeepPot* dp,
                       const int natoms,
                       const double* coord,
                       const int* atype,
                       const double* cell,
                       double* energy,
                       double* force,
                       double* virial,
                       double* atomic_energy,
                       double* atomic_virial) {
  DP_DeepPotCompute_variant<double>(dp, 1, natoms, coord, atype, cell, NULL,
                                    NULL, energy, force, virial, atomic_energy,
                                    atomic_virial);
}

void DP_DeepPotComputef(DP_DeepPot* dp,
                        const int natoms,
                        const float* coord,
                        const int* atype,
                        const float* cell,
                        double* energy,
                        float* force,
                        float* virial,
                        float* atomic_energy,
                        float* atomic_virial) {
  DP_DeepPotCompute_variant<float>(dp, 1, natoms, coord, atype, cell, NULL,
                                   NULL, energy, force, virial, atomic_energy,
                                   atomic_virial);
}

void DP_DeepPotComputeNList(DP_DeepPot* dp,
                            const int natoms,
                            const double* coord,
                            const int* atype,
                            const double* cell,
                            const int nghost,
                            const DP_Nlist* nlist,
                            const int ago,
                            double* energy,
                            double* force,
                            double* virial,
                            double* atomic_energy,
                            double* atomic_virial) {
  DP_DeepPotComputeNList_variant<double>(
      dp, 1, natoms, coord, atype, cell, nghost, nlist, ago, NULL, NULL, energy,
      force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputeNListf(DP_DeepPot* dp,
                             const int natoms,
                             const float* coord,
                             const int* atype,
                             const float* cell,
                             const int nghost,
                             const DP_Nlist* nlist,
                             const int ago,
                             double* energy,
                             float* force,
                             float* virial,
                             float* atomic_energy,
                             float* atomic_virial) {
  DP_DeepPotComputeNList_variant<float>(
      dp, 1, natoms, coord, atype, cell, nghost, nlist, ago, NULL, NULL, energy,
      force, virial, atomic_energy, atomic_virial);
}

// multiple frames
void DP_DeepPotCompute2(DP_DeepPot* dp,
                        const int nframes,
                        const int natoms,
                        const double* coord,
                        const int* atype,
                        const double* cell,
                        const double* fparam,
                        const double* aparam,
                        double* energy,
                        double* force,
                        double* virial,
                        double* atomic_energy,
                        double* atomic_virial) {
  DP_DeepPotCompute_variant<double>(dp, nframes, natoms, coord, atype, cell,
                                    fparam, aparam, energy, force, virial,
                                    atomic_energy, atomic_virial);
}

void DP_DeepPotComputef2(DP_DeepPot* dp,
                         const int nframes,
                         const int natoms,
                         const float* coord,
                         const int* atype,
                         const float* cell,
                         const float* fparam,
                         const float* aparam,
                         double* energy,
                         float* force,
                         float* virial,
                         float* atomic_energy,
                         float* atomic_virial) {
  DP_DeepPotCompute_variant<float>(dp, nframes, natoms, coord, atype, cell,
                                   fparam, aparam, energy, force, virial,
                                   atomic_energy, atomic_virial);
}

void DP_DeepPotComputeNList2(DP_DeepPot* dp,
                             const int nframes,
                             const int natoms,
                             const double* coord,
                             const int* atype,
                             const double* cell,
                             const int nghost,
                             const DP_Nlist* nlist,
                             const int ago,
                             const double* fparam,
                             const double* aparam,
                             double* energy,
                             double* force,
                             double* virial,
                             double* atomic_energy,
                             double* atomic_virial) {
  DP_DeepPotComputeNList_variant<double>(
      dp, nframes, natoms, coord, atype, cell, nghost, nlist, ago, fparam,
      aparam, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputeNListf2(DP_DeepPot* dp,
                              const int nframes,
                              const int natoms,
                              const float* coord,
                              const int* atype,
                              const float* cell,
                              const int nghost,
                              const DP_Nlist* nlist,
                              const int ago,
                              const float* fparam,
                              const float* aparam,
                              double* energy,
                              float* force,
                              float* virial,
                              float* atomic_energy,
                              float* atomic_virial) {
  DP_DeepPotComputeNList_variant<float>(
      dp, nframes, natoms, coord, atype, cell, nghost, nlist, ago, fparam,
      aparam, energy, force, virial, atomic_energy, atomic_virial);
}
// end multiple frames

void DP_DeepPotComputeMixedType(DP_DeepPot* dp,
                                const int nframes,
                                const int natoms,
                                const double* coord,
                                const int* atype,
                                const double* cell,
                                const double* fparam,
                                const double* aparam,
                                double* energy,
                                double* force,
                                double* virial,
                                double* atomic_energy,
                                double* atomic_virial) {
  DP_DeepPotComputeMixedType_variant<double>(
      dp, nframes, natoms, coord, atype, cell, fparam, aparam, energy, force,
      virial, atomic_energy, atomic_virial);
}

void DP_DeepPotComputeMixedTypef(DP_DeepPot* dp,
                                 const int nframes,
                                 const int natoms,
                                 const float* coord,
                                 const int* atype,
                                 const float* cell,
                                 const float* fparam,
                                 const float* aparam,
                                 double* energy,
                                 float* force,
                                 float* virial,
                                 float* atomic_energy,
                                 float* atomic_virial) {
  DP_DeepPotComputeMixedType_variant<float>(
      dp, nframes, natoms, coord, atype, cell, fparam, aparam, energy, force,
      virial, atomic_energy, atomic_virial);
}

const char* DP_DeepPotGetTypeMap(DP_DeepPot* dp) {
  std::string type_map;
  dp->dp.get_type_map(type_map);
  return string_to_char(type_map);
}

double DP_DeepPotGetCutoff(DP_DeepPot* dp) { return dp->dp.cutoff(); }

int DP_DeepPotGetNumbTypes(DP_DeepPot* dp) { return dp->dp.numb_types(); }

int DP_DeepPotGetNumbTypesSpin(DP_DeepPot* dp) {
  return dp->dp.numb_types_spin();
}

int DP_DeepPotGetDimFParam(DP_DeepPot* dp) { return dp->dfparam; }

int DP_DeepPotGetDimAParam(DP_DeepPot* dp) { return dp->daparam; }

bool DP_DeepPotIsAParamNAll(DP_DeepPot* dp) { return dp->aparam_nall; }

const char* DP_DeepPotCheckOK(DP_DeepPot* dp) {
  return string_to_char(dp->exception);
}

void DP_DeepPotModelDeviComputeNList(DP_DeepPotModelDevi* dp,
                                     const int natoms,
                                     const double* coord,
                                     const int* atype,
                                     const double* cell,
                                     const int nghost,
                                     const DP_Nlist* nlist,
                                     const int ago,
                                     double* energy,
                                     double* force,
                                     double* virial,
                                     double* atomic_energy,
                                     double* atomic_virial) {
  DP_DeepPotModelDeviComputeNList_variant<double>(
      dp, 1, natoms, coord, atype, cell, nghost, nlist, ago, NULL, NULL, energy,
      force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotModelDeviComputeNListf(DP_DeepPotModelDevi* dp,
                                      const int natoms,
                                      const float* coord,
                                      const int* atype,
                                      const float* cell,
                                      const int nghost,
                                      const DP_Nlist* nlist,
                                      const int ago,
                                      double* energy,
                                      float* force,
                                      float* virial,
                                      float* atomic_energy,
                                      float* atomic_virial) {
  DP_DeepPotModelDeviComputeNList_variant<float>(
      dp, 1, natoms, coord, atype, cell, nghost, nlist, ago, NULL, NULL, energy,
      force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotModelDeviComputeNList2(DP_DeepPotModelDevi* dp,
                                      const int nframes,
                                      const int natoms,
                                      const double* coord,
                                      const int* atype,
                                      const double* cell,
                                      const int nghost,
                                      const DP_Nlist* nlist,
                                      const int ago,
                                      const double* fparam,
                                      const double* aparam,
                                      double* energy,
                                      double* force,
                                      double* virial,
                                      double* atomic_energy,
                                      double* atomic_virial) {
  DP_DeepPotModelDeviComputeNList_variant<double>(
      dp, nframes, natoms, coord, atype, cell, nghost, nlist, ago, fparam,
      aparam, energy, force, virial, atomic_energy, atomic_virial);
}

void DP_DeepPotModelDeviComputeNListf2(DP_DeepPotModelDevi* dp,
                                       const int nframes,
                                       const int natoms,
                                       const float* coord,
                                       const int* atype,
                                       const float* cell,
                                       const int nghost,
                                       const DP_Nlist* nlist,
                                       const int ago,
                                       const float* fparam,
                                       const float* aparam,
                                       double* energy,
                                       float* force,
                                       float* virial,
                                       float* atomic_energy,
                                       float* atomic_virial) {
  DP_DeepPotModelDeviComputeNList_variant<float>(
      dp, nframes, natoms, coord, atype, cell, nghost, nlist, ago, fparam,
      aparam, energy, force, virial, atomic_energy, atomic_virial);
}

double DP_DeepPotModelDeviGetCutoff(DP_DeepPotModelDevi* dp) {
  return dp->dp.cutoff();
}

int DP_DeepPotModelDeviGetNumbTypes(DP_DeepPotModelDevi* dp) {
  return dp->dp.numb_types();
}

int DP_DeepPotModelDeviGetNumbTypesSpin(DP_DeepPotModelDevi* dp) {
  return dp->dp.numb_types_spin();
}

int DP_DeepPotModelDeviGetDimFParam(DP_DeepPotModelDevi* dp) {
  return dp->dfparam;
}

int DP_DeepPotModelDeviGetDimAParam(DP_DeepPotModelDevi* dp) {
  return dp->daparam;
}

bool DP_DeepPotModelDeviIsAParamNAll(DP_DeepPotModelDevi* dp) {
  return dp->aparam_nall;
}

const char* DP_DeepPotModelDeviCheckOK(DP_DeepPotModelDevi* dp) {
  return string_to_char(dp->exception);
}

void DP_DeepTensorComputeTensor(DP_DeepTensor* dt,
                                const int natoms,
                                const double* coord,
                                const int* atype,
                                const double* cell,
                                double** tensor,
                                int* size) {
  DP_DeepTensorComputeTensor_variant<double>(dt, natoms, coord, atype, cell,
                                             tensor, size);
}

void DP_DeepTensorComputeTensorf(DP_DeepTensor* dt,
                                 const int natoms,
                                 const float* coord,
                                 const int* atype,
                                 const float* cell,
                                 float** tensor,
                                 int* size) {
  DP_DeepTensorComputeTensor_variant<float>(dt, natoms, coord, atype, cell,
                                            tensor, size);
}

void DP_DeepTensorComputeTensorNList(DP_DeepTensor* dt,
                                     const int natoms,
                                     const double* coord,
                                     const int* atype,
                                     const double* cell,
                                     const int nghost,
                                     const DP_Nlist* nlist,
                                     double** tensor,
                                     int* size) {
  DP_DeepTensorComputeTensorNList_variant<double>(
      dt, natoms, coord, atype, cell, nghost, nlist, tensor, size);
}

void DP_DeepTensorComputeTensorNListf(DP_DeepTensor* dt,
                                      const int natoms,
                                      const float* coord,
                                      const int* atype,
                                      const float* cell,
                                      const int nghost,
                                      const DP_Nlist* nlist,
                                      float** tensor,
                                      int* size) {
  DP_DeepTensorComputeTensorNList_variant<float>(dt, natoms, coord, atype, cell,
                                                 nghost, nlist, tensor, size);
}

void DP_DeepTensorCompute(DP_DeepTensor* dt,
                          const int natoms,
                          const double* coord,
                          const int* atype,
                          const double* cell,
                          double* global_tensor,
                          double* force,
                          double* virial,
                          double** atomic_tensor,
                          double* atomic_virial,
                          int* size_at) {
  DP_DeepTensorCompute_variant<double>(dt, natoms, coord, atype, cell,
                                       global_tensor, force, virial,
                                       atomic_tensor, atomic_virial, size_at);
}

void DP_DeepTensorComputef(DP_DeepTensor* dt,
                           const int natoms,
                           const float* coord,
                           const int* atype,
                           const float* cell,
                           float* global_tensor,
                           float* force,
                           float* virial,
                           float** atomic_tensor,
                           float* atomic_virial,
                           int* size_at) {
  DP_DeepTensorCompute_variant<float>(dt, natoms, coord, atype, cell,
                                      global_tensor, force, virial,
                                      atomic_tensor, atomic_virial, size_at);
}

void DP_DeepTensorComputeNList(DP_DeepTensor* dt,
                               const int natoms,
                               const double* coord,
                               const int* atype,
                               const double* cell,
                               const int nghost,
                               const DP_Nlist* nlist,
                               double* global_tensor,
                               double* force,
                               double* virial,
                               double** atomic_tensor,
                               double* atomic_virial,
                               int* size_at) {
  DP_DeepTensorComputeNList_variant<double>(
      dt, natoms, coord, atype, cell, nghost, nlist, global_tensor, force,
      virial, atomic_tensor, atomic_virial, size_at);
}

void DP_DeepTensorComputeNListf(DP_DeepTensor* dt,
                                const int natoms,
                                const float* coord,
                                const int* atype,
                                const float* cell,
                                const int nghost,
                                const DP_Nlist* nlist,
                                float* global_tensor,
                                float* force,
                                float* virial,
                                float** atomic_tensor,
                                float* atomic_virial,
                                int* size_at) {
  DP_DeepTensorComputeNList_variant<float>(
      dt, natoms, coord, atype, cell, nghost, nlist, global_tensor, force,
      virial, atomic_tensor, atomic_virial, size_at);
}

double DP_DeepTensorGetCutoff(DP_DeepTensor* dt) { return dt->dt.cutoff(); }

int DP_DeepTensorGetNumbTypes(DP_DeepTensor* dt) { return dt->dt.numb_types(); }

int DP_DeepTensorGetOutputDim(DP_DeepTensor* dt) { return dt->dt.output_dim(); }

int* DP_DeepTensorGetSelTypes(DP_DeepTensor* dt) {
  return (int*)&(dt->dt.sel_types())[0];
}

int DP_DeepTensorGetNumbSelTypes(DP_DeepTensor* dt) {
  return dt->dt.sel_types().size();
}

const char* DP_DeepTensorGetTypeMap(DP_DeepTensor* dt) {
  std::string type_map;
  dt->dt.get_type_map(type_map);
  return string_to_char(type_map);
}

const char* DP_DeepTensorCheckOK(DP_DeepTensor* dt) {
  return string_to_char(dt->exception);
}

void DP_DipoleChargeModifierComputeNList(DP_DipoleChargeModifier* dcm,
                                         const int natom,
                                         const double* coord,
                                         const int* atype,
                                         const double* cell,
                                         const int* pairs,
                                         const int npairs,
                                         const double* delef_,
                                         const int nghost,
                                         const DP_Nlist* nlist,
                                         double* dfcorr_,
                                         double* dvcorr_) {
  DP_DipoleChargeModifierComputeNList_variant<double>(
      dcm, natom, coord, atype, cell, pairs, npairs, delef_, nghost, nlist,
      dfcorr_, dvcorr_);
}

void DP_DipoleChargeModifierComputeNListf(DP_DipoleChargeModifier* dcm,
                                          const int natom,
                                          const float* coord,
                                          const int* atype,
                                          const float* cell,
                                          const int* pairs,
                                          const int npairs,
                                          const float* delef_,
                                          const int nghost,
                                          const DP_Nlist* nlist,
                                          float* dfcorr_,
                                          float* dvcorr_) {
  DP_DipoleChargeModifierComputeNList_variant<float>(
      dcm, natom, coord, atype, cell, pairs, npairs, delef_, nghost, nlist,
      dfcorr_, dvcorr_);
}

double DP_DipoleChargeModifierGetCutoff(DP_DipoleChargeModifier* dcm) {
  return dcm->dcm.cutoff();
}

int DP_DipoleChargeModifierGetNumbTypes(DP_DipoleChargeModifier* dcm) {
  return dcm->dcm.numb_types();
}

int* DP_DipoleChargeModifierGetSelTypes(DP_DipoleChargeModifier* dcm) {
  return (int*)&(dcm->dcm.sel_types())[0];
}

int DP_DipoleChargeModifierGetNumbSelTypes(DP_DipoleChargeModifier* dcm) {
  return dcm->dcm.sel_types().size();
}

const char* DP_DipoleChargeModifierCheckOK(DP_DipoleChargeModifier* dcm) {
  return string_to_char(dcm->exception);
}

void DP_ConvertPbtxtToPb(const char* c_pbtxt, const char* c_pb) {
  std::string pbtxt(c_pbtxt);
  std::string pb(c_pb);
  deepmd::convert_pbtxt_to_pb(pbtxt, pb);
}

void DP_PrintSummary(const char* c_pre) {
  std::string pre(c_pre);
  deepmd::print_summary(pre);
}

const char* DP_ReadFileToChar(const char* c_model) {
  std::string model(c_model);
  std::string file_content;
  deepmd::read_file_to_string(model, file_content);
  return string_to_char(file_content);
}

const char* DP_ReadFileToChar2(const char* c_model, int* size) {
  std::string model(c_model);
  std::string file_content;
  try {
    deepmd::read_file_to_string(model, file_content);
  } catch (deepmd::deepmd_exception& ex) {
    // use negtive size to indicate error
    std::string error_message = std::string(ex.what());
    *size = -error_message.size();
    return string_to_char(error_message);
  }
  *size = file_content.size();
  return string_to_char(file_content);
}

void DP_SelectByType(const int natoms,
                     const int* atype,
                     const int nghost,
                     const int nsel_type,
                     const int* sel_type,
                     int* fwd_map,
                     int* nreal,
                     int* bkw_map,
                     int* nghost_real) {
  std::vector<int> atype_(atype, atype + natoms);
  std::vector<int> sel_type_(sel_type, sel_type + nsel_type);
  std::vector<int> fwd_map_, bkw_map_;
  int nghost_real_;
  deepmd::select_by_type(fwd_map_, bkw_map_, nghost_real_,
                         std::vector<double>(), atype_, nghost, sel_type_);
  if (fwd_map) {
    std::copy(fwd_map_.begin(), fwd_map_.end(), fwd_map);
  }
  if (bkw_map) {
    std::copy(bkw_map_.begin(), bkw_map_.end(), bkw_map);
  }
  if (nreal) {
    *nreal = bkw_map_.size();
  }
  if (nghost_real) {
    *nghost_real = nghost_real_;
  }
}

void DP_SelectMapInt(const int* in,
                     const int* fwd_map,
                     const int stride,
                     const int nall1,
                     const int nall2,
                     int* out) {
  std::vector<int> in_(in, in + stride * nall1);
  std::vector<int> fwd_map_(fwd_map, fwd_map + nall1);
  std::vector<int> out_(static_cast<size_t>(stride) * nall2);
  deepmd::select_map(out_, in_, fwd_map_, stride);
  if (out) {
    std::copy(out_.begin(), out_.end(), out);
  }
}

void DP_DeleteChar(const char* c_str) { delete[] c_str; }

}  // extern "C"
