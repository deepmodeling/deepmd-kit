// SPDX-License-Identifier: LGPL-3.0-or-later
/*
Header-only DeePMD-kit C++ 11 library

This header-only library provides a C++ 11 interface to the DeePMD-kit C API.
*/

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "c_api.h"

namespace deepmd {
namespace hpp {
/**
 * @brief General DeePMD-kit exception. Throw if anything doesn't work.
 **/
struct deepmd_exception : public std::runtime_error {
 public:
  deepmd_exception() : runtime_error("DeePMD-kit C API Error!"){};
  deepmd_exception(const std::string &msg)
      : runtime_error(std::string("DeePMD-kit C API Error: ") + msg){};
};
}  // namespace hpp
}  // namespace deepmd

/**
 * @brief Check if any exceptions throw in the C++ API. Throw if possible.
 */
#define DP_CHECK_OK(check_func, dp)                   \
  const char *err_msg = check_func(dp);               \
  if (std::strlen(err_msg)) {                         \
    std::string err_msg_str = std::string(err_msg);   \
    DP_DeleteChar(err_msg);                           \
    throw deepmd::hpp::deepmd_exception(err_msg_str); \
  }                                                   \
  DP_DeleteChar(err_msg);

template <typename FPTYPE>
inline void _DP_DeepPotCompute(DP_DeepPot *dp,
                               const int nframes,
                               const int natom,
                               const FPTYPE *coord,
                               const int *atype,
                               const FPTYPE *cell,
                               const FPTYPE *fparam,
                               const FPTYPE *aparam,
                               double *energy,
                               FPTYPE *force,
                               FPTYPE *virial,
                               FPTYPE *atomic_energy,
                               FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotCompute<double>(DP_DeepPot *dp,
                                       const int nframes,
                                       const int natom,
                                       const double *coord,
                                       const int *atype,
                                       const double *cell,
                                       const double *fparam,
                                       const double *aparam,
                                       double *energy,
                                       double *force,
                                       double *virial,
                                       double *atomic_energy,
                                       double *atomic_virial) {
  DP_DeepPotCompute2(dp, nframes, natom, coord, atype, cell, fparam, aparam,
                     energy, force, virial, atomic_energy, atomic_virial);
}

template <>
inline void _DP_DeepPotCompute<float>(DP_DeepPot *dp,
                                      const int nframes,
                                      const int natom,
                                      const float *coord,
                                      const int *atype,
                                      const float *cell,
                                      const float *fparam,
                                      const float *aparam,
                                      double *energy,
                                      float *force,
                                      float *virial,
                                      float *atomic_energy,
                                      float *atomic_virial) {
  DP_DeepPotComputef2(dp, nframes, natom, coord, atype, cell, fparam, aparam,
                      energy, force, virial, atomic_energy, atomic_virial);
}

template <typename FPTYPE>
inline void _DP_DeepPotComputeNList(DP_DeepPot *dp,
                                    const int nframes,
                                    const int natom,
                                    const FPTYPE *coord,
                                    const int *atype,
                                    const FPTYPE *cell,
                                    const int nghost,
                                    const DP_Nlist *nlist,
                                    const int ago,
                                    const FPTYPE *fparam,
                                    const FPTYPE *aparam,
                                    double *energy,
                                    FPTYPE *force,
                                    FPTYPE *virial,
                                    FPTYPE *atomic_energy,
                                    FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotComputeNList<double>(DP_DeepPot *dp,
                                            const int nframes,
                                            const int natom,
                                            const double *coord,
                                            const int *atype,
                                            const double *cell,
                                            const int nghost,
                                            const DP_Nlist *nlist,
                                            const int ago,
                                            const double *fparam,
                                            const double *aparam,
                                            double *energy,
                                            double *force,
                                            double *virial,
                                            double *atomic_energy,
                                            double *atomic_virial) {
  DP_DeepPotComputeNList2(dp, nframes, natom, coord, atype, cell, nghost, nlist,
                          ago, fparam, aparam, energy, force, virial,
                          atomic_energy, atomic_virial);
}

template <>
inline void _DP_DeepPotComputeNList<float>(DP_DeepPot *dp,
                                           const int nframes,
                                           const int natom,
                                           const float *coord,
                                           const int *atype,
                                           const float *cell,
                                           const int nghost,
                                           const DP_Nlist *nlist,
                                           const int ago,
                                           const float *fparam,
                                           const float *aparam,
                                           double *energy,
                                           float *force,
                                           float *virial,
                                           float *atomic_energy,
                                           float *atomic_virial) {
  DP_DeepPotComputeNListf2(dp, nframes, natom, coord, atype, cell, nghost,
                           nlist, ago, fparam, aparam, energy, force, virial,
                           atomic_energy, atomic_virial);
}

template <typename FPTYPE>
inline void _DP_DeepPotComputeMixedType(DP_DeepPot *dp,
                                        const int nframes,
                                        const int natom,
                                        const FPTYPE *coord,
                                        const int *atype,
                                        const FPTYPE *cell,
                                        const FPTYPE *fparam,
                                        const FPTYPE *aparam,
                                        double *energy,
                                        FPTYPE *force,
                                        FPTYPE *virial,
                                        FPTYPE *atomic_energy,
                                        FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotComputeMixedType<double>(DP_DeepPot *dp,
                                                const int nframes,
                                                const int natom,
                                                const double *coord,
                                                const int *atype,
                                                const double *cell,
                                                const double *fparam,
                                                const double *aparam,
                                                double *energy,
                                                double *force,
                                                double *virial,
                                                double *atomic_energy,
                                                double *atomic_virial) {
  DP_DeepPotComputeMixedType(dp, nframes, natom, coord, atype, cell, fparam,
                             aparam, energy, force, virial, atomic_energy,
                             atomic_virial);
}

template <>
inline void _DP_DeepPotComputeMixedType<float>(DP_DeepPot *dp,
                                               const int nframes,
                                               const int natom,
                                               const float *coord,
                                               const int *atype,
                                               const float *cell,
                                               const float *fparam,
                                               const float *aparam,
                                               double *energy,
                                               float *force,
                                               float *virial,
                                               float *atomic_energy,
                                               float *atomic_virial) {
  DP_DeepPotComputeMixedTypef(dp, nframes, natom, coord, atype, cell, fparam,
                              aparam, energy, force, virial, atomic_energy,
                              atomic_virial);
}

template <typename FPTYPE>
inline void _DP_DeepPotModelDeviComputeNList(DP_DeepPotModelDevi *dp,
                                             const int natom,
                                             const FPTYPE *coord,
                                             const int *atype,
                                             const FPTYPE *cell,
                                             const int nghost,
                                             const DP_Nlist *nlist,
                                             const int ago,
                                             const FPTYPE *fparam,
                                             const FPTYPE *aparam,
                                             double *energy,
                                             FPTYPE *force,
                                             FPTYPE *virial,
                                             FPTYPE *atomic_energy,
                                             FPTYPE *atomic_virial);

template <>
inline void _DP_DeepPotModelDeviComputeNList<double>(DP_DeepPotModelDevi *dp,
                                                     const int natom,
                                                     const double *coord,
                                                     const int *atype,
                                                     const double *cell,
                                                     const int nghost,
                                                     const DP_Nlist *nlist,
                                                     const int ago,
                                                     const double *fparam,
                                                     const double *aparam,
                                                     double *energy,
                                                     double *force,
                                                     double *virial,
                                                     double *atomic_energy,
                                                     double *atomic_virial) {
  DP_DeepPotModelDeviComputeNList2(dp, 1, natom, coord, atype, cell, nghost,
                                   nlist, ago, fparam, aparam, energy, force,
                                   virial, atomic_energy, atomic_virial);
}

template <>
inline void _DP_DeepPotModelDeviComputeNList<float>(DP_DeepPotModelDevi *dp,
                                                    const int natom,
                                                    const float *coord,
                                                    const int *atype,
                                                    const float *cell,
                                                    const int nghost,
                                                    const DP_Nlist *nlist,
                                                    const int ago,
                                                    const float *fparam,
                                                    const float *aparam,
                                                    double *energy,
                                                    float *force,
                                                    float *virial,
                                                    float *atomic_energy,
                                                    float *atomic_virial) {
  DP_DeepPotModelDeviComputeNListf2(dp, 1, natom, coord, atype, cell, nghost,
                                    nlist, ago, fparam, aparam, energy, force,
                                    virial, atomic_energy, atomic_virial);
}

template <typename FPTYPE>
inline void _DP_DeepTensorComputeTensor(DP_DeepTensor *dt,
                                        const int natom,
                                        const FPTYPE *coord,
                                        const int *atype,
                                        const FPTYPE *cell,
                                        FPTYPE **tensor,
                                        int *size);

template <>
inline void _DP_DeepTensorComputeTensor<double>(DP_DeepTensor *dt,
                                                const int natom,
                                                const double *coord,
                                                const int *atype,
                                                const double *cell,
                                                double **tensor,
                                                int *size) {
  DP_DeepTensorComputeTensor(dt, natom, coord, atype, cell, tensor, size);
}

template <>
inline void _DP_DeepTensorComputeTensor<float>(DP_DeepTensor *dt,
                                               const int natom,
                                               const float *coord,
                                               const int *atype,
                                               const float *cell,
                                               float **tensor,
                                               int *size) {
  DP_DeepTensorComputeTensorf(dt, natom, coord, atype, cell, tensor, size);
}

template <typename FPTYPE>
inline void _DP_DeepTensorComputeTensorNList(DP_DeepTensor *dt,
                                             const int natom,
                                             const FPTYPE *coord,
                                             const int *atype,
                                             const FPTYPE *cell,
                                             const int nghost,
                                             const DP_Nlist *nlist,
                                             FPTYPE **tensor,
                                             int *size);

template <>
inline void _DP_DeepTensorComputeTensorNList<double>(DP_DeepTensor *dt,
                                                     const int natom,
                                                     const double *coord,
                                                     const int *atype,
                                                     const double *cell,
                                                     const int nghost,
                                                     const DP_Nlist *nlist,
                                                     double **tensor,
                                                     int *size) {
  DP_DeepTensorComputeTensorNList(dt, natom, coord, atype, cell, nghost, nlist,
                                  tensor, size);
}

template <>
inline void _DP_DeepTensorComputeTensorNList<float>(DP_DeepTensor *dt,
                                                    const int natom,
                                                    const float *coord,
                                                    const int *atype,
                                                    const float *cell,
                                                    const int nghost,
                                                    const DP_Nlist *nlist,
                                                    float **tensor,
                                                    int *size) {
  DP_DeepTensorComputeTensorNListf(dt, natom, coord, atype, cell, nghost, nlist,
                                   tensor, size);
}

template <typename FPTYPE>
inline void _DP_DeepTensorCompute(DP_DeepTensor *dt,
                                  const int natom,
                                  const FPTYPE *coord,
                                  const int *atype,
                                  const FPTYPE *cell,
                                  FPTYPE *global_tensor,
                                  FPTYPE *force,
                                  FPTYPE *virial,
                                  FPTYPE **atomic_energy,
                                  FPTYPE *atomic_virial,
                                  int *size_at);

template <>
inline void _DP_DeepTensorCompute<double>(DP_DeepTensor *dt,
                                          const int natom,
                                          const double *coord,
                                          const int *atype,
                                          const double *cell,
                                          double *global_tensor,
                                          double *force,
                                          double *virial,
                                          double **atomic_tensor,
                                          double *atomic_virial,
                                          int *size_at) {
  DP_DeepTensorCompute(dt, natom, coord, atype, cell, global_tensor, force,
                       virial, atomic_tensor, atomic_virial, size_at);
}

template <>
inline void _DP_DeepTensorCompute<float>(DP_DeepTensor *dt,
                                         const int natom,
                                         const float *coord,
                                         const int *atype,
                                         const float *cell,
                                         float *global_tensor,
                                         float *force,
                                         float *virial,
                                         float **atomic_tensor,
                                         float *atomic_virial,
                                         int *size_at) {
  DP_DeepTensorComputef(dt, natom, coord, atype, cell, global_tensor, force,
                        virial, atomic_tensor, atomic_virial, size_at);
}

template <typename FPTYPE>
inline void _DP_DeepTensorComputeNList(DP_DeepTensor *dt,
                                       const int natom,
                                       const FPTYPE *coord,
                                       const int *atype,
                                       const FPTYPE *cell,
                                       const int nghost,
                                       const DP_Nlist *nlist,
                                       FPTYPE *global_tensor,
                                       FPTYPE *force,
                                       FPTYPE *virial,
                                       FPTYPE **atomic_energy,
                                       FPTYPE *atomic_virial,
                                       int *size_at);

template <>
inline void _DP_DeepTensorComputeNList<double>(DP_DeepTensor *dt,
                                               const int natom,
                                               const double *coord,
                                               const int *atype,
                                               const double *cell,
                                               const int nghost,
                                               const DP_Nlist *nlist,
                                               double *global_tensor,
                                               double *force,
                                               double *virial,
                                               double **atomic_tensor,
                                               double *atomic_virial,
                                               int *size_at) {
  DP_DeepTensorComputeNList(dt, natom, coord, atype, cell, nghost, nlist,
                            global_tensor, force, virial, atomic_tensor,
                            atomic_virial, size_at);
}

template <>
inline void _DP_DeepTensorComputeNList<float>(DP_DeepTensor *dt,
                                              const int natom,
                                              const float *coord,
                                              const int *atype,
                                              const float *cell,
                                              const int nghost,
                                              const DP_Nlist *nlist,
                                              float *global_tensor,
                                              float *force,
                                              float *virial,
                                              float **atomic_tensor,
                                              float *atomic_virial,
                                              int *size_at) {
  DP_DeepTensorComputeNListf(dt, natom, coord, atype, cell, nghost, nlist,
                             global_tensor, force, virial, atomic_tensor,
                             atomic_virial, size_at);
}

template <typename FPTYPE>
inline void _DP_DipoleChargeModifierComputeNList(DP_DipoleChargeModifier *dcm,
                                                 const int natom,
                                                 const FPTYPE *coord,
                                                 const int *atype,
                                                 const FPTYPE *cell,
                                                 const int *pairs,
                                                 const int npairs,
                                                 const FPTYPE *delef_,
                                                 const int nghost,
                                                 const DP_Nlist *nlist,
                                                 FPTYPE *dfcorr_,
                                                 FPTYPE *dvcorr_);

template <>
inline void _DP_DipoleChargeModifierComputeNList<double>(
    DP_DipoleChargeModifier *dcm,
    const int natom,
    const double *coord,
    const int *atype,
    const double *cell,
    const int *pairs,
    const int npairs,
    const double *delef_,
    const int nghost,
    const DP_Nlist *nlist,
    double *dfcorr_,
    double *dvcorr_) {
  DP_DipoleChargeModifierComputeNList(dcm, natom, coord, atype, cell, pairs,
                                      npairs, delef_, nghost, nlist, dfcorr_,
                                      dvcorr_);
}

template <>
inline void _DP_DipoleChargeModifierComputeNList<float>(
    DP_DipoleChargeModifier *dcm,
    const int natom,
    const float *coord,
    const int *atype,
    const float *cell,
    const int *pairs,
    const int npairs,
    const float *delef_,
    const int nghost,
    const DP_Nlist *nlist,
    float *dfcorr_,
    float *dvcorr_) {
  DP_DipoleChargeModifierComputeNListf(dcm, natom, coord, atype, cell, pairs,
                                       npairs, delef_, nghost, nlist, dfcorr_,
                                       dvcorr_);
}

inline double *_DP_Get_Energy_Pointer(std::vector<double> &vec,
                                      const int nframes) {
  vec.resize(nframes);
  return &vec[0];
}

inline double *_DP_Get_Energy_Pointer(double &vec, const int nframes) {
  assert(nframes == 1);
  return &vec;
}

namespace deepmd {
namespace hpp {
/**
 * @brief Neighbor list.
 **/
struct InputNlist {
  InputNlist()
      : inum(0),
        ilist(nullptr),
        numneigh(nullptr),
        firstneigh(nullptr),
        nl(DP_NewNlist(0, nullptr, nullptr, nullptr)) {
    DP_CHECK_OK(DP_NlistCheckOK, nl);
  };
  InputNlist(int inum_, int *ilist_, int *numneigh_, int **firstneigh_)
      : inum(inum_),
        ilist(ilist_),
        numneigh(numneigh_),
        firstneigh(firstneigh_),
        nl(DP_NewNlist(inum_, ilist_, numneigh_, firstneigh_)) {
    DP_CHECK_OK(DP_NlistCheckOK, nl);
  };
  ~InputNlist() { DP_DeleteNlist(nl); };
  /// @brief C API neighbor list.
  DP_Nlist *nl;
  /// @brief Number of core region atoms
  int inum;
  /// @brief Array stores the core region atom's index
  int *ilist;
  /// @brief Array stores the core region atom's neighbor atom number
  int *numneigh;
  /// @brief Array stores the core region atom's neighbor index
  int **firstneigh;
};

/**
 * @brief Convert pbtxt to pb.
 * @param[in] fn_pb_txt Filename of the pb txt file.
 * @param[in] fn_pb Filename of the pb file.
 **/
void inline convert_pbtxt_to_pb(std::string fn_pb_txt, std::string fn_pb) {
  DP_ConvertPbtxtToPb(fn_pb_txt.c_str(), fn_pb.c_str());
};
/**
 * @brief Convert int vector to InputNlist.
 * @param[out] to_nlist InputNlist.
 * @param[in] from_nlist 2D int vector. The first axis represents the centeral
 * atoms and the second axis represents the neighbor atoms.
 */
void inline convert_nlist(InputNlist &to_nlist,
                          std::vector<std::vector<int>> &from_nlist) {
  to_nlist.inum = from_nlist.size();
  for (int ii = 0; ii < to_nlist.inum; ++ii) {
    to_nlist.ilist[ii] = ii;
    to_nlist.numneigh[ii] = from_nlist[ii].size();
    to_nlist.firstneigh[ii] = &from_nlist[ii][0];
  }
  // delete the original nl
  DP_DeleteNlist(to_nlist.nl);
  to_nlist.nl = DP_NewNlist(to_nlist.inum, to_nlist.ilist, to_nlist.numneigh,
                            to_nlist.firstneigh);
}
/**
 * @brief Deep Potential.
 **/
class DeepPot {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPot() : dp(nullptr){};
  ~DeepPot() { DP_DeleteDeepPot(dp); };
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank.
   * @param[in] file_content The content of the frozen model file.
   **/
  DeepPot(const std::string &model,
          const int &gpu_rank = 0,
          const std::string &file_content = "")
      : dp(nullptr) {
    try {
      init(model, gpu_rank, file_content);
    } catch (...) {
      // Clean up and rethrow, as the destructor will not be called
      if (dp) {
        DP_DeleteDeepPot(dp);
      }
      throw;
    }
  };
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank.
   * @param[in] file_content The content of the frozen model file.
   **/
  void init(const std::string &model,
            const int &gpu_rank = 0,
            const std::string &file_content = "") {
    if (dp) {
      std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                   "nothing at the second call of initializer"
                << std::endl;
      return;
    }
    dp = DP_NewDeepPotWithParam2(model.c_str(), gpu_rank, file_content.c_str(),
                                 file_content.size());
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
    dfparam = DP_DeepPotGetDimFParam(dp);
    daparam = DP_DeepPotGetDimAParam(dp);
    aparam_nall = DP_DeepPotIsAParamNAll(dp);
  };

  /**
   * @brief Evaluate the energy, force and virial by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @warning Natoms should not be zero when computing multiple frames.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(
      ENERGYVTYPE &ener,
      std::vector<VALUETYPE> &force,
      std::vector<VALUETYPE> &virial,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size();
    unsigned int nframes = natoms > 0 ? coord.size() / natoms / 3 : 1;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(static_cast<size_t>(nframes) * natoms * 3);
    virial.resize(static_cast<size_t>(nframes) * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, natoms, fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes, natoms * daparam, aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotCompute<VALUETYPE>(dp, nframes, natoms, coord_, atype_, box_,
                                  fparam__, aparam__, ener_, force_, virial_,
                                  nullptr, nullptr);
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
  };
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @warning Natoms should not be zero when computing multiple frames.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(
      ENERGYVTYPE &ener,
      std::vector<VALUETYPE> &force,
      std::vector<VALUETYPE> &virial,
      std::vector<VALUETYPE> &atom_energy,
      std::vector<VALUETYPE> &atom_virial,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size();
    unsigned int nframes = natoms > 0 ? coord.size() / natoms / 3 : 1;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(static_cast<size_t>(nframes) * natoms * 3);
    virial.resize(static_cast<size_t>(nframes) * 9);
    atom_energy.resize(static_cast<size_t>(nframes) * natoms);
    atom_virial.resize(static_cast<size_t>(nframes) * natoms * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_ener_ = &atom_energy[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, natoms, fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes, natoms * daparam, aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotCompute<VALUETYPE>(dp, nframes, natoms, coord_, atype_, box_,
                                  fparam__, aparam__, ener_, force_, virial_,
                                  atomic_ener_, atomic_virial_);
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
  };

  /**
   * @brief Evaluate the energy, force and virial by using this DP with the
   *neighbor list.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] nghost The number of ghost atoms.
   * @param[in] nlist The neighbor list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @warning Natoms should not be zero when computing multiple frames.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(
      ENERGYVTYPE &ener,
      std::vector<VALUETYPE> &force,
      std::vector<VALUETYPE> &virial,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const int nghost,
      const InputNlist &lmp_list,
      const int &ago,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size();
    unsigned int nframes = natoms > 0 ? coord.size() / natoms / 3 : 1;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(static_cast<size_t>(nframes) * natoms * 3);
    virial.resize(static_cast<size_t>(nframes) * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, (aparam_nall ? natoms : (natoms - nghost)),
                           fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes,
                       (aparam_nall ? natoms : (natoms - nghost)) * daparam,
                       aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotComputeNList<VALUETYPE>(
        dp, nframes, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago,
        fparam__, aparam__, ener_, force_, virial_, nullptr, nullptr);
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
  };
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP with the neighbor list.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] nghost The number of ghost atoms.
   * @param[in] nlist The neighbor list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @warning Natoms should not be zero when computing multiple frames.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(
      ENERGYVTYPE &ener,
      std::vector<VALUETYPE> &force,
      std::vector<VALUETYPE> &virial,
      std::vector<VALUETYPE> &atom_energy,
      std::vector<VALUETYPE> &atom_virial,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const int nghost,
      const InputNlist &lmp_list,
      const int &ago,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size();
    unsigned int nframes = natoms > 0 ? coord.size() / natoms / 3 : 1;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(static_cast<size_t>(nframes) * natoms * 3);
    virial.resize(static_cast<size_t>(nframes) * 9);
    atom_energy.resize(static_cast<size_t>(nframes) * natoms);
    atom_virial.resize(static_cast<size_t>(nframes) * natoms * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_ener_ = &atom_energy[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, (aparam_nall ? natoms : (natoms - nghost)),
                           fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes,
                       (aparam_nall ? natoms : (natoms - nghost)) * daparam,
                       aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotComputeNList<VALUETYPE>(dp, nframes, natoms, coord_, atype_,
                                       box_, nghost, lmp_list.nl, ago, fparam__,
                                       aparam__, ener_, force_, virial_,
                                       atomic_ener_, atomic_virial_);
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
  };
  /**
   * @brief Evaluate the energy, force and virial by using this DP with the
   *mixed type.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_mixed_type(
      ENERGYVTYPE &ener,
      std::vector<VALUETYPE> &force,
      std::vector<VALUETYPE> &virial,
      const int &nframes,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size() / nframes;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(static_cast<size_t>(nframes) * natoms * 3);
    virial.resize(static_cast<size_t>(nframes) * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, natoms, fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes, natoms * daparam, aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotComputeMixedType<VALUETYPE>(dp, nframes, natoms, coord_, atype_,
                                           box_, fparam__, aparam__, ener_,
                                           force_, virial_, nullptr, nullptr);
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
  };
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP with the mixed type.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_mixed_type(
      ENERGYVTYPE &ener,
      std::vector<VALUETYPE> &force,
      std::vector<VALUETYPE> &virial,
      std::vector<VALUETYPE> &atom_energy,
      std::vector<VALUETYPE> &atom_virial,
      const int &nframes,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size() / nframes;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(static_cast<size_t>(nframes) * natoms * 3);
    virial.resize(static_cast<size_t>(nframes) * 9);
    atom_energy.resize(static_cast<size_t>(nframes) * natoms);
    atom_virial.resize(static_cast<size_t>(nframes) * natoms * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_ener_ = &atom_energy[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, natoms, fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes, natoms * daparam, aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotComputeMixedType<VALUETYPE>(
        dp, nframes, natoms, coord_, atype_, box_, fparam__, aparam__, ener_,
        force_, virial_, atomic_ener_, atomic_virial_);
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
  };
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(dp);
    return DP_DeepPotGetCutoff(dp);
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(dp);
    return DP_DeepPotGetNumbTypes(dp);
  };
  /**
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  int numb_types_spin() const {
    assert(dp);
    return DP_DeepPotGetNumbTypesSpin(dp);
  };
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string &type_map) {
    const char *type_map_c = DP_DeepPotGetTypeMap(dp);
    type_map.assign(type_map_c);
    DP_DeleteChar(type_map_c);
  };
  /**
   * @brief Print the summary of DeePMD-kit, including the version and the build
   * information.
   * @param[in] pre The prefix to each line.
   */
  void print_summary(const std::string &pre) const {
    DP_PrintSummary(pre.c_str());
  }
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  int dim_fparam() const {
    assert(dp);
    return dfparam;
  }
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  int dim_aparam() const {
    assert(dp);
    return daparam;
  }

 private:
  DP_DeepPot *dp;
  int dfparam;
  int daparam;
  bool aparam_nall;
  template <typename VALUETYPE>
  void validate_fparam_aparam(const int &nframes,
                              const int &nloc,
                              const std::vector<VALUETYPE> &fparam,
                              const std::vector<VALUETYPE> &aparam) const {
    if (fparam.size() != dfparam &&
        fparam.size() != static_cast<size_t>(nframes) * dfparam) {
      throw deepmd::hpp::deepmd_exception(
          "the dim of frame parameter provided is not consistent with what the "
          "model uses");
    }

    if (aparam.size() != static_cast<size_t>(daparam) * nloc &&
        aparam.size() != static_cast<size_t>(nframes) * daparam * nloc) {
      throw deepmd::hpp::deepmd_exception(
          "the dim of atom parameter provided is not consistent with what the "
          "model uses");
    }
  }
  template <typename VALUETYPE>
  void tile_fparam_aparam(std::vector<VALUETYPE> &out_param,
                          const int &nframes,
                          const int &dparam,
                          const std::vector<VALUETYPE> &param) const {
    if (param.size() == dparam) {
      out_param.resize(static_cast<size_t>(nframes) * dparam);
      for (int ii = 0; ii < nframes; ++ii) {
        std::copy(param.begin(), param.end(),
                  out_param.begin() + static_cast<std::ptrdiff_t>(ii) * dparam);
      }
    } else if (param.size() == static_cast<size_t>(nframes) * dparam) {
      out_param = param;
    }
  }
};

/**
 * @brief Deep Potential model deviation.
 **/
class DeepPotModelDevi {
 public:
  /**
   * @brief DP model deviation constructor without initialization.
   **/
  DeepPotModelDevi() : dp(nullptr){};
  ~DeepPotModelDevi() { DP_DeleteDeepPotModelDevi(dp); };
  /**
   * @brief DP model deviation constructor with initialization.
   * @param[in] models The names of the frozen model file.
   **/
  DeepPotModelDevi(const std::vector<std::string> &models) : dp(nullptr) {
    try {
      init(models);
    } catch (...) {
      // Clean up and rethrow, as the destructor will not be called
      if (dp) {
        DP_DeleteDeepPotModelDevi(dp);
      }
      throw;
    }
  };
  /**
   * @brief Initialize the DP model deviation.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank.
   * @param[in] file_content The content of the frozen model file.
   **/
  void init(const std::vector<std::string> &models,
            const int &gpu_rank = 0,
            const std::vector<std::string> &file_content =
                std::vector<std::string>()) {
    if (dp) {
      std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                   "nothing at the second call of initializer"
                << std::endl;
      return;
    }
    std::vector<const char *> cstrings;
    cstrings.reserve(models.size());
    for (std::string const &str : models) {
      cstrings.push_back(str.data());
    }

    std::vector<const char *> c_file_contents;
    std::vector<int> size_file_contents;
    c_file_contents.reserve(file_content.size());
    size_file_contents.reserve(file_content.size());
    for (std::string const &str : file_content) {
      c_file_contents.push_back(str.data());
      size_file_contents.push_back(str.size());
    }

    dp = DP_NewDeepPotModelDeviWithParam(
        cstrings.data(), cstrings.size(), gpu_rank, c_file_contents.data(),
        c_file_contents.size(), size_file_contents.data());
    DP_CHECK_OK(DP_DeepPotModelDeviCheckOK, dp);
    numb_models = models.size();
    dfparam = DP_DeepPotModelDeviGetDimFParam(dp);
    daparam = DP_DeepPotModelDeviGetDimAParam(dp);
    aparam_nall = DP_DeepPotModelDeviIsAParamNAll(dp);
  };

  /**
   * @brief Evaluate the energy, force and virial by using this DP model
   *deviation.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   **/
  template <typename VALUETYPE>
  void compute(
      std::vector<double> &ener,
      std::vector<std::vector<VALUETYPE>> &force,
      std::vector<std::vector<VALUETYPE>> &virial,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const int nghost,
      const InputNlist &lmp_list,
      const int &ago,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size();
    unsigned int nframes = 1;
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    // memory will be continous for std::vector but not std::vector<std::vector>
    std::vector<double> energy_flat(numb_models);
    std::vector<VALUETYPE> force_flat(static_cast<size_t>(numb_models) *
                                      natoms * 3);
    std::vector<VALUETYPE> virial_flat(numb_models * 9);
    double *ener_ = &energy_flat[0];
    VALUETYPE *force_ = &force_flat[0];
    VALUETYPE *virial_ = &virial_flat[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, (aparam_nall ? natoms : (natoms - nghost)),
                           fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes,
                       (aparam_nall ? natoms : (natoms - nghost)) * daparam,
                       aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotModelDeviComputeNList<VALUETYPE>(
        dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, fparam__,
        aparam__, ener_, force_, virial_, nullptr, nullptr);
    DP_CHECK_OK(DP_DeepPotModelDeviCheckOK, dp);

    // reshape
    ener.resize(numb_models);
    force.resize(numb_models);
    virial.resize(numb_models);
    for (int i = 0; i < numb_models; i++) {
      ener[i] = energy_flat[i];
      force[i].resize(static_cast<size_t>(natoms) * 3);
      virial[i].resize(9);
      for (int j = 0; j < natoms * 3; j++) {
        force[i][j] = force_flat[i * natoms * 3 + j];
      }
      for (int j = 0; j < 9; j++) {
        virial[i][j] = virial_flat[i * 9 + j];
      }
    }
  };
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP model deviation.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   **/
  template <typename VALUETYPE>
  void compute(
      std::vector<double> &ener,
      std::vector<std::vector<VALUETYPE>> &force,
      std::vector<std::vector<VALUETYPE>> &virial,
      std::vector<std::vector<VALUETYPE>> &atom_energy,
      std::vector<std::vector<VALUETYPE>> &atom_virial,
      const std::vector<VALUETYPE> &coord,
      const std::vector<int> &atype,
      const std::vector<VALUETYPE> &box,
      const int nghost,
      const InputNlist &lmp_list,
      const int &ago,
      const std::vector<VALUETYPE> &fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE> &aparam = std::vector<VALUETYPE>()) {
    unsigned int natoms = atype.size();
    unsigned int nframes = 1;
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    std::vector<double> energy_flat(numb_models);
    std::vector<VALUETYPE> force_flat(static_cast<size_t>(numb_models) *
                                      natoms * 3);
    std::vector<VALUETYPE> virial_flat(numb_models * 9);
    std::vector<VALUETYPE> atom_energy_flat(static_cast<size_t>(numb_models) *
                                            natoms);
    std::vector<VALUETYPE> atom_virial_flat(static_cast<size_t>(numb_models) *
                                            natoms * 9);
    double *ener_ = &energy_flat[0];
    VALUETYPE *force_ = &force_flat[0];
    VALUETYPE *virial_ = &virial_flat[0];
    VALUETYPE *atomic_ener_ = &atom_energy_flat[0];
    VALUETYPE *atomic_virial_ = &atom_virial_flat[0];
    std::vector<VALUETYPE> fparam_, aparam_;
    validate_fparam_aparam(nframes, (aparam_nall ? natoms : (natoms - nghost)),
                           fparam, aparam);
    tile_fparam_aparam(fparam_, nframes, dfparam, fparam);
    tile_fparam_aparam(aparam_, nframes,
                       (aparam_nall ? natoms : (natoms - nghost)) * daparam,
                       aparam);
    const VALUETYPE *fparam__ = !fparam_.empty() ? &fparam_[0] : nullptr;
    const VALUETYPE *aparam__ = !aparam_.empty() ? &aparam_[0] : nullptr;

    _DP_DeepPotModelDeviComputeNList<VALUETYPE>(
        dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, fparam__,
        aparam__, ener_, force_, virial_, atomic_ener_, atomic_virial_);
    DP_CHECK_OK(DP_DeepPotModelDeviCheckOK, dp);

    // reshape
    ener.resize(numb_models);
    force.resize(numb_models);
    virial.resize(numb_models);
    atom_energy.resize(numb_models);
    atom_virial.resize(numb_models);
    for (int i = 0; i < numb_models; i++) {
      ener[i] = energy_flat[i];
      force[i].resize(static_cast<size_t>(natoms) * 3);
      virial[i].resize(9);
      atom_energy[i].resize(natoms);
      atom_virial[i].resize(static_cast<size_t>(natoms) * 9);
      for (int j = 0; j < natoms * 3; j++) {
        force[i][j] = force_flat[i * natoms * 3 + j];
      }
      for (int j = 0; j < 9; j++) {
        virial[i][j] = virial_flat[i * 9 + j];
      }
      for (int j = 0; j < natoms; j++) {
        atom_energy[i][j] = atom_energy_flat[i * natoms + j];
      }
      for (int j = 0; j < natoms * 9; j++) {
        atom_virial[i][j] = atom_virial_flat[i * natoms * 9 + j];
      }
    }
  };
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(dp);
    return DP_DeepPotModelDeviGetCutoff(dp);
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(dp);
    return DP_DeepPotModelDeviGetNumbTypes(dp);
  };
  /**
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  int numb_types_spin() const {
    assert(dp);
    return DP_DeepPotModelDeviGetNumbTypesSpin(dp);
  };
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  int dim_fparam() const {
    assert(dp);
    return dfparam;
  }
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  int dim_aparam() const {
    assert(dp);
    return daparam;
  }
  /**
   * @brief Compute the average of vectors.
   * @param[out] avg The average of vectors.
   * @param[in] xx The vectors of all models.
   **/
  template <typename VALUETYPE>
  void compute_avg(std::vector<VALUETYPE> &avg,
                   const std::vector<std::vector<VALUETYPE>> &xx) {
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
  };
  /**
   * @brief Compute the standard deviation of vectors.
   * @param[out] std The standard deviation of vectors.
   * @param[in] avg The average of vectors.
   * @param[in] xx The vectors of all models.
   * @param[in] stride The stride to compute the deviation.
   **/
  template <typename VALUETYPE>
  void compute_std(std::vector<VALUETYPE> &std,
                   const std::vector<VALUETYPE> &avg,
                   const std::vector<std::vector<VALUETYPE>> &xx,
                   const int &stride) {
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
        const VALUETYPE *tmp_f = &(xx[ii][static_cast<size_t>(jj) * stride]);
        const VALUETYPE *tmp_avg = &(avg[static_cast<size_t>(jj) * stride]);
        for (unsigned dd = 0; dd < stride; ++dd) {
          VALUETYPE vdiff = tmp_f[dd] - tmp_avg[dd];
          std[jj] += vdiff * vdiff;
        }
      }
    }

    for (unsigned jj = 0; jj < nloc; ++jj) {
      std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
    }
  };
  /**
   * @brief Compute the relative standard deviation of vectors.
   * @param[out] std The standard deviation of vectors.
   * @param[in] avg The average of vectors.
   * @param[in] eps The level parameter for computing the deviation.
   * @param[in] stride The stride to compute the deviation.
   **/
  template <typename VALUETYPE>
  void compute_relative_std(std::vector<VALUETYPE> &std,
                            const std::vector<VALUETYPE> &avg,
                            const VALUETYPE eps,
                            const int &stride) {
    unsigned ndof = avg.size();
    unsigned nloc = std.size();
    assert(nloc * stride == ndof);

    for (unsigned ii = 0; ii < nloc; ++ii) {
      const VALUETYPE *tmp_avg = &(avg[static_cast<size_t>(ii) * stride]);
      VALUETYPE f_norm = 0.0;
      for (unsigned dd = 0; dd < stride; ++dd) {
        f_norm += tmp_avg[dd] * tmp_avg[dd];
      }
      f_norm = sqrt(f_norm);
      std[ii] /= f_norm + eps;
    }
  };
  /**
   * @brief Compute the standard deviation of forces.
   * @param[out] std The standard deviation of forces.
   * @param[in] avg The average of forces.
   * @param[in] xx The vectors of all forces.
   **/
  template <typename VALUETYPE>
  void compute_std_f(std::vector<VALUETYPE> &std,
                     const std::vector<VALUETYPE> &avg,
                     const std::vector<std::vector<VALUETYPE>> &xx) {
    compute_std(std, avg, xx, 3);
  };
  /**
   * @brief Compute the relative standard deviation of forces.
   * @param[out] std The relative standard deviation of forces.
   * @param[in] avg The relative average of forces.
   * @param[in] eps The level parameter for computing the deviation.
   **/
  template <typename VALUETYPE>
  void compute_relative_std_f(std::vector<VALUETYPE> &std,
                              const std::vector<VALUETYPE> &avg,
                              const VALUETYPE eps) {
    compute_relative_std(std, avg, eps, 3);
  };

 private:
  DP_DeepPotModelDevi *dp;
  int numb_models;
  int dfparam;
  int daparam;
  bool aparam_nall;
  template <typename VALUETYPE>
  void validate_fparam_aparam(const int &nframes,
                              const int &nloc,
                              const std::vector<VALUETYPE> &fparam,
                              const std::vector<VALUETYPE> &aparam) const {
    if (fparam.size() != dfparam &&
        fparam.size() != static_cast<size_t>(nframes) * dfparam) {
      throw deepmd::hpp::deepmd_exception(
          "the dim of frame parameter provided is not consistent with what the "
          "model uses");
    }

    if (aparam.size() != static_cast<size_t>(daparam) * nloc &&
        aparam.size() != static_cast<size_t>(nframes) * daparam * nloc) {
      throw deepmd::hpp::deepmd_exception(
          "the dim of atom parameter provided is not consistent with what the "
          "model uses");
    }
  }
  template <typename VALUETYPE>
  void tile_fparam_aparam(std::vector<VALUETYPE> &out_param,
                          const int &nframes,
                          const int &dparam,
                          const std::vector<VALUETYPE> &param) const {
    if (param.size() == dparam) {
      out_param.resize(static_cast<size_t>(nframes) * dparam);
      for (int ii = 0; ii < nframes; ++ii) {
        std::copy(param.begin(), param.end(),
                  out_param.begin() + static_cast<std::ptrdiff_t>(ii) * dparam);
      }
    } else if (param.size() == static_cast<size_t>(nframes) * dparam) {
      out_param = param;
    }
  }
};

/**
 * @brief Deep Tensor.
 **/
class DeepTensor {
 public:
  /**
   * @brief Deep Tensor constructor without initialization.
   **/
  DeepTensor() : dt(nullptr){};
  ~DeepTensor() { DP_DeleteDeepTensor(dt); };
  /**
   * @brief DeepTensor constructor with initialization.
   * @param[in] model The name of the frozen model file.
   **/
  DeepTensor(const std::string &model,
             const int &gpu_rank = 0,
             const std::string &name_scope = "")
      : dt(nullptr) {
    try {
      init(model, gpu_rank, name_scope);
    } catch (...) {
      // Clean up and rethrow, as the destructor will not be called
      if (dt) {
        DP_DeleteDeepTensor(dt);
      }
      throw;
    }
  };
  /**
   * @brief Initialize the DeepTensor.
   * @param[in] model The name of the frozen model file.
   **/
  void init(const std::string &model,
            const int &gpu_rank = 0,
            const std::string &name_scope = "") {
    if (dt) {
      std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                   "nothing at the second call of initializer"
                << std::endl;
      return;
    }
    dt = DP_NewDeepTensorWithParam(model.c_str(), gpu_rank, name_scope.c_str());
    DP_CHECK_OK(DP_DeepTensorCheckOK, dt);
    odim = output_dim();
    nsel_types = DP_DeepTensorGetNumbSelTypes(dt);
  };

  /**
   * @brief Evaluate the tensor, force and virial by using this Deep Tensor.
   * @param[out] tensor The atomic tensor.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE> &tensor,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    VALUETYPE *tensor_;
    VALUETYPE **p_tensor = &tensor_;
    int size;
    int *p_size = &size;

    _DP_DeepTensorComputeTensor<VALUETYPE>(dt, natoms, coord_, atype_, box_,
                                           p_tensor, p_size);
    DP_CHECK_OK(DP_DeepTensorCheckOK, dt);

    tensor.resize(size);
    std::copy(tensor_, tensor_ + size, tensor.begin());
    delete[] tensor_;
  };

  /**
   * @brief Evaluate the tensor, force and virial by using this Deep Tensor with
   *the neighbor list.
   * @param[out] tensor The tensor.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] nghost The number of ghost atoms.
   * @param[in] nlist The neighbor list.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE> &tensor,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box,
               const int nghost,
               const InputNlist &lmp_list) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    VALUETYPE *tensor_;
    VALUETYPE **p_tensor = &tensor_;
    int size;
    int *p_size = &size;

    _DP_DeepTensorComputeTensorNList<VALUETYPE>(dt, natoms, coord_, atype_,
                                                box_, nghost, lmp_list.nl,
                                                p_tensor, p_size);
    DP_CHECK_OK(DP_DeepTensorCheckOK, dt);

    tensor.resize(size);
    std::copy(tensor_, tensor_ + size, tensor.begin());
    delete[] tensor_;
  };

  /**
   * @brief Evaluate the global tensor, force and virial by using this Deep
   *Tensor.
   * @param[out] global_tensor The global tensor.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE> &global_tensor,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    global_tensor.resize(odim);
    force.resize(static_cast<size_t>(odim) * natoms * 3);
    virial.resize(static_cast<size_t>(odim) * 9);
    VALUETYPE *global_tensor_ = &global_tensor[0];
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];

    _DP_DeepTensorCompute<VALUETYPE>(dt, natoms, coord_, atype_, box_,
                                     global_tensor_, force_, virial_, nullptr,
                                     nullptr, nullptr);
    DP_CHECK_OK(DP_DeepTensorCheckOK, dt);
  };
  /**
   * @brief Evaluate the global tensor, force, virial, atomic tensor, and atomic
   *virial by using this Deep Tensor.
   * @param[out] global_tensor The global tensor.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_tensor The atomic tensor.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE> &global_tensor,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               std::vector<VALUETYPE> &atom_tensor,
               std::vector<VALUETYPE> &atom_virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    global_tensor.resize(odim);
    force.resize(static_cast<size_t>(odim) * natoms * 3);
    virial.resize(static_cast<size_t>(odim) * 9);
    atom_virial.resize(static_cast<size_t>(odim) * natoms * 9);
    VALUETYPE *global_tensor_ = &global_tensor[0];
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];

    VALUETYPE *atomic_tensor_;
    VALUETYPE **p_atomic_tensor = &atomic_tensor_;
    int size_at;
    int *p_size_at = &size_at;

    _DP_DeepTensorCompute<VALUETYPE>(
        dt, natoms, coord_, atype_, box_, global_tensor_, force_, virial_,
        p_atomic_tensor, atomic_virial_, p_size_at);
    DP_CHECK_OK(DP_DeepTensorCheckOK, dt);

    atom_tensor.resize(size_at);
    std::copy(atomic_tensor_, atomic_tensor_ + size_at, atom_tensor.begin());
    delete[] atomic_tensor_;
  };

  /**
   * @brief Evaluate the global tensor, force and virial by using this Deep
   *Tensor with the neighbor list.
   * @param[out] global_tensor The global tensor.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] nghost The number of ghost atoms.
   * @param[in] nlist The neighbor list.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE> &global_tensor,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box,
               const int nghost,
               const InputNlist &lmp_list) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    global_tensor.resize(odim);
    force.resize(static_cast<size_t>(odim) * natoms * 3);
    virial.resize(static_cast<size_t>(odim) * 9);
    VALUETYPE *global_tensor_ = &global_tensor[0];
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];

    _DP_DeepTensorComputeNList<VALUETYPE>(
        dt, natoms, coord_, atype_, box_, nghost, lmp_list.nl, global_tensor_,
        force_, virial_, nullptr, nullptr, nullptr);
    DP_CHECK_OK(DP_DeepTensorCheckOK, dt);
  };
  /**
   * @brief Evaluate the global tensor, force, virial, atomic tensor, and atomic
   *virial by using this Deep Tensor with the neighbor list.
   * @param[out] global_tensor The global tensor.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_tensor The atomic tensor.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9 (PBC) or empty (no PBC).
   * @param[in] nghost The number of ghost atoms.
   * @param[in] nlist The neighbor list.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE> &global_tensor,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               std::vector<VALUETYPE> &atom_tensor,
               std::vector<VALUETYPE> &atom_virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box,
               const int nghost,
               const InputNlist &lmp_list) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    global_tensor.resize(odim);
    force.resize(static_cast<size_t>(odim) * natoms * 3);
    virial.resize(static_cast<size_t>(odim) * 9);
    atom_virial.resize(static_cast<size_t>(odim) * natoms * 9);
    VALUETYPE *global_tensor_ = &global_tensor[0];
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];

    VALUETYPE *atomic_tensor_;
    VALUETYPE **p_atomic_tensor = &atomic_tensor_;
    int size_at;
    int *p_size_at = &size_at;

    _DP_DeepTensorComputeNList<VALUETYPE>(
        dt, natoms, coord_, atype_, box_, nghost, lmp_list.nl, global_tensor_,
        force_, virial_, p_atomic_tensor, atomic_virial_, p_size_at);
    DP_CHECK_OK(DP_DeepTensorCheckOK, dt);

    atom_tensor.resize(size_at);
    std::copy(atomic_tensor_, atomic_tensor_ + size_at, atom_tensor.begin());
    delete[] atomic_tensor_;
  };
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(dt);
    return DP_DeepTensorGetCutoff(dt);
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(dt);
    return DP_DeepTensorGetNumbTypes(dt);
  };
  /**
   * @brief Get the output dimension.
   * @return The output dimension.
   **/
  int output_dim() const {
    assert(dt);
    return DP_DeepTensorGetOutputDim(dt);
  }

  std::vector<int> sel_types() const {
    int *sel_types_arr = DP_DeepTensorGetSelTypes(dt);
    std::vector<int> sel_types_vec =
        std::vector<int>(sel_types_arr, sel_types_arr + nsel_types);
    return sel_types_vec;
  }
  /**
   * @brief Print the summary of DeePMD-kit, including the version and the build
   * information.
   * @param[in] pre The prefix to each line.
   */
  void print_summary(const std::string &pre) const {
    DP_PrintSummary(pre.c_str());
  }
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string &type_map) {
    const char *type_map_c = DP_DeepTensorGetTypeMap(dt);
    type_map.assign(type_map_c);
    DP_DeleteChar(type_map_c);
  };

 private:
  DP_DeepTensor *dt;
  int odim;
  int nsel_types;
};

class DipoleChargeModifier {
 public:
  /**
   * @brief DipoleChargeModifier constructor without initialization.
   **/
  DipoleChargeModifier() : dcm(nullptr){};
  ~DipoleChargeModifier() { DP_DeleteDipoleChargeModifier(dcm); };
  /**
   * @brief DipoleChargeModifier constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The rank of the GPU to be used.
   * @param[in] name_scope The name scope of the model.
   **/
  DipoleChargeModifier(const std::string &model,
                       const int &gpu_rank = 0,
                       const std::string &name_scope = "")
      : dcm(nullptr) {
    try {
      init(model, gpu_rank, name_scope);
    } catch (...) {
      // Clean up and rethrow, as the destructor will not be called
      if (dcm) {
        DP_DeleteDipoleChargeModifier(dcm);
      }
      throw;
    }
  };
  /**
   * @brief Initialize the DipoleChargeModifier.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The rank of the GPU to be used.
   * @param[in] name_scope The name scope of the model.
   **/
  void init(const std::string &model,
            const int &gpu_rank = 0,
            const std::string &name_scope = "") {
    if (dcm) {
      std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                   "nothing at the second call of initializer"
                << std::endl;
      return;
    }
    dcm = DP_NewDipoleChargeModifierWithParam(model.c_str(), gpu_rank,
                                              name_scope.c_str());
    DP_CHECK_OK(DP_DipoleChargeModifierCheckOK, dcm);
    nsel_types = DP_DipoleChargeModifierGetNumbSelTypes(dcm);
  };
  /**
   * @brief Evaluate the force and virial correction by using this dipole charge
   *modifier.
   * @param[out] dfcorr_ The force correction on each atom.
   * @param[out] dvcorr_ The virial correction.
   * @param[in] dcoord_ The coordinates of atoms. The array should be of size
   *nall x 3.
   * @param[in] datype_ The atom types. The list should contain nall ints.
   * @param[in] dbox The cell of the region. The array should be of size 9.
   * @param[in] pairs The pairs of atoms. The list should contain npairs pairs
   *of ints.
   * @param[in] delef_ The electric field on each atom. The array should be of
   *size nghost x 3.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The neighbor list.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE> &dfcorr_,
               std::vector<VALUETYPE> &dvcorr_,
               const std::vector<VALUETYPE> &dcoord_,
               const std::vector<int> &datype_,
               const std::vector<VALUETYPE> &dbox,
               const std::vector<std::pair<int, int>> &pairs,
               const std::vector<VALUETYPE> &delef_,
               const int nghost,
               const InputNlist &lmp_list) {
    unsigned int natoms = datype_.size();
    assert(natoms * 3 == dcoord_.size());
    if (!dbox.empty()) {
      assert(dbox.size() == 9);
    }
    const VALUETYPE *dcoord = &dcoord_[0];
    const VALUETYPE *dbox_ = !dbox.empty() ? &dbox[0] : nullptr;
    const int *datype = &datype_[0];
    const int npairs = pairs.size();
    const int *dpairs = reinterpret_cast<const int *>(&pairs[0]);
    const VALUETYPE *delef = &delef_[0];

    dfcorr_.resize(static_cast<size_t>(natoms) * 3);
    dvcorr_.resize(9);
    VALUETYPE *dfcorr = &dfcorr_[0];
    VALUETYPE *dvcorr = &dvcorr_[0];

    _DP_DipoleChargeModifierComputeNList<VALUETYPE>(
        dcm, natoms, dcoord, datype, dbox_, dpairs, npairs, delef, nghost,
        lmp_list.nl, dfcorr, dvcorr);
    DP_CHECK_OK(DP_DipoleChargeModifierCheckOK, dcm);
  };
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(dcm);
    return DP_DipoleChargeModifierGetCutoff(dcm);
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(dcm);
    return DP_DipoleChargeModifierGetNumbTypes(dcm);
  };

  std::vector<int> sel_types() const {
    int *sel_types_arr = DP_DipoleChargeModifierGetSelTypes(dcm);
    std::vector<int> sel_types_vec =
        std::vector<int>(sel_types_arr, sel_types_arr + nsel_types);
    return sel_types_vec;
  }

  /**
   * @brief Print the summary of DeePMD-kit, including the version and the build
   * information.
   * @param[in] pre The prefix to each line.
   */
  void print_summary(const std::string &pre) const {
    DP_PrintSummary(pre.c_str());
  }

 private:
  DP_DipoleChargeModifier *dcm;
  int nsel_types;
};

/**
 * @brief Read model file to a string.
 * @param[in] model Path to the model.
 * @param[out] file_content Content of the model file.
 **/
void inline read_file_to_string(std::string model, std::string &file_content) {
  int size;
  const char *c_file_content = DP_ReadFileToChar2(model.c_str(), &size);
  if (size < 0) {
    // negtive size indicates error
    std::string error_message = std::string(c_file_content, -size);
    DP_DeleteChar(c_file_content);
    throw deepmd::hpp::deepmd_exception(error_message);
  }
  file_content = std::string(c_file_content, size);
  DP_DeleteChar(c_file_content);
};

/**
 * @brief Get forward and backward map of selected atoms by
 * atom types.
 * @param[out] fwd_map The forward map with size natoms.
 * @param[out] bkw_map The backward map with size nreal.
 * @param[out] nghost_real The number of selected ghost atoms.
 * @param[in] dcoord_ The coordinates of all atoms. Reserved for compatibility.
 * @param[in] datype_ The atom types of all atoms.
 * @param[in] nghost The number of ghost atoms.
 * @param[in] sel_type_ The selected atom types.
 */
template <typename VALUETYPE>
void select_by_type(std::vector<int> &fwd_map,
                    std::vector<int> &bkw_map,
                    int &nghost_real,
                    const std::vector<VALUETYPE> &dcoord_,
                    const std::vector<int> &datype_,
                    const int &nghost,
                    const std::vector<int> &sel_type_) {
  const int natoms = datype_.size();
  const int nsel_type = sel_type_.size();
  fwd_map.resize(natoms);
  // do not know nghost_real at this time
  bkw_map.resize(natoms);
  int nreal;
  DP_SelectByType(natoms, &datype_[0], nghost, nsel_type, &sel_type_[0],
                  &fwd_map[0], &nreal, &bkw_map[0], &nghost_real);
  bkw_map.resize(nreal);
};

/**
 * @brief Apply the given map to a vector. Assume nframes is 1.
 * @tparam VT The value type of the vector. Only support int.
 * @param[out] out The output vector.
 * @param[in] in The input vector.
 * @param[in] fwd_map The map.
 * @param[in] stride The stride of the input vector.
 */
template <typename VT>
void select_map(std::vector<VT> &out,
                const std::vector<VT> &in,
                const std::vector<int> &fwd_map,
                const int &stride) {
  static_assert(std::is_same<int, VT>(), "only support int");
  const int nall1 = in.size() / stride;
  int nall2 = 0;
  for (int ii = 0; ii < nall1; ++ii) {
    if (fwd_map[ii] >= 0) {
      nall2++;
    }
  }
  out.resize(static_cast<size_t>(nall2) * stride);
  DP_SelectMapInt(&in[0], &fwd_map[0], stride, nall1, nall2, &out[0]);
};

}  // namespace hpp
}  // namespace deepmd
