/*
Header-only DeePMD-kit C++ 11 library

This header-only library provides a C++ 11 interface to the DeePMD-kit C API.
*/

#pragma once

#include <algorithm>
#include <cassert>
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
#define DP_CHECK_OK(check_func, dp)     \
  const char *err_msg = check_func(dp); \
  if (std::strlen(err_msg))             \
    throw deepmd::hpp::deepmd_exception(std::string(err_msg));

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
                                                     double *energy,
                                                     double *force,
                                                     double *virial,
                                                     double *atomic_energy,
                                                     double *atomic_virial) {
  DP_DeepPotModelDeviComputeNList(dp, natom, coord, atype, cell, nghost, nlist,
                                  ago, energy, force, virial, atomic_energy,
                                  atomic_virial);
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
                                                    double *energy,
                                                    float *force,
                                                    float *virial,
                                                    float *atomic_energy,
                                                    float *atomic_virial) {
  DP_DeepPotModelDeviComputeNListf(dp, natom, coord, atype, cell, nghost, nlist,
                                   ago, energy, force, virial, atomic_energy,
                                   atomic_virial);
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
  ~DeepPot(){};
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
    init(model, gpu_rank, file_content);
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
    dp = DP_NewDeepPotWithParam(model.c_str(), gpu_rank, file_content.c_str());
    DP_CHECK_OK(DP_DeepPotCheckOK, dp);
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
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE &ener,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box) {
    unsigned int natoms = atype.size();
    unsigned int nframes = coord.size() / natoms / 3;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(nframes * natoms * 3);
    virial.resize(nframes * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];

    _DP_DeepPotCompute<VALUETYPE>(dp, nframes, natoms, coord_, atype_, box_,
                                  nullptr, nullptr, ener_, force_, virial_,
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
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE &ener,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               std::vector<VALUETYPE> &atom_energy,
               std::vector<VALUETYPE> &atom_virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box) {
    unsigned int natoms = atype.size();
    unsigned int nframes = coord.size() / natoms / 3;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(nframes * natoms * 3);
    virial.resize(nframes * 9);
    atom_energy.resize(nframes * natoms);
    atom_virial.resize(nframes * natoms * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_ener_ = &atom_energy[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];

    _DP_DeepPotCompute<VALUETYPE>(dp, nframes, natoms, coord_, atype_, box_,
                                  nullptr, nullptr, ener_, force_, virial_,
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
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE &ener,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box,
               const int nghost,
               const InputNlist &lmp_list,
               const int &ago) {
    unsigned int natoms = atype.size();
    unsigned int nframes = coord.size() / natoms / 3;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(nframes * natoms * 3);
    virial.resize(nframes * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];

    _DP_DeepPotComputeNList<VALUETYPE>(
        dp, nframes, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago,
        nullptr, nullptr, ener_, force_, virial_, nullptr, nullptr);
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
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE &ener,
               std::vector<VALUETYPE> &force,
               std::vector<VALUETYPE> &virial,
               std::vector<VALUETYPE> &atom_energy,
               std::vector<VALUETYPE> &atom_virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box,
               const int nghost,
               const InputNlist &lmp_list,
               const int &ago) {
    unsigned int natoms = atype.size();
    unsigned int nframes = coord.size() / natoms / 3;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(nframes * natoms * 3);
    virial.resize(nframes * 9);
    atom_energy.resize(nframes * natoms);
    atom_virial.resize(nframes * natoms * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_ener_ = &atom_energy[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];

    _DP_DeepPotComputeNList<VALUETYPE>(
        dp, nframes, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago,
        nullptr, nullptr, ener_, force_, virial_, atomic_ener_, atomic_virial_);
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
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_mixed_type(ENERGYVTYPE &ener,
                          std::vector<VALUETYPE> &force,
                          std::vector<VALUETYPE> &virial,
                          const int &nframes,
                          const std::vector<VALUETYPE> &coord,
                          const std::vector<int> &atype,
                          const std::vector<VALUETYPE> &box) {
    unsigned int natoms = atype.size() / nframes;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];
    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(nframes * natoms * 3);
    virial.resize(nframes * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];

    _DP_DeepPotComputeMixedType<VALUETYPE>(dp, nframes, natoms, coord_, atype_,
                                           box_, nullptr, nullptr, ener_,
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
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_mixed_type(ENERGYVTYPE &ener,
                          std::vector<VALUETYPE> &force,
                          std::vector<VALUETYPE> &virial,
                          std::vector<VALUETYPE> &atom_energy,
                          std::vector<VALUETYPE> &atom_virial,
                          const int &nframes,
                          const std::vector<VALUETYPE> &coord,
                          const std::vector<int> &atype,
                          const std::vector<VALUETYPE> &box) {
    unsigned int natoms = atype.size() / nframes;
    assert(nframes * natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == nframes * 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    double *ener_ = _DP_Get_Energy_Pointer(ener, nframes);
    force.resize(nframes * natoms * 3);
    virial.resize(nframes * 9);
    atom_energy.resize(nframes * natoms);
    atom_virial.resize(nframes * natoms * 9);
    VALUETYPE *force_ = &force[0];
    VALUETYPE *virial_ = &virial[0];
    VALUETYPE *atomic_ener_ = &atom_energy[0];
    VALUETYPE *atomic_virial_ = &atom_virial[0];

    _DP_DeepPotComputeMixedType<VALUETYPE>(
        dp, nframes, natoms, coord_, atype_, box_, nullptr, nullptr, ener_,
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
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string &type_map) {
    const char *type_map_c = DP_DeepPotGetTypeMap(dp);
    type_map.assign(type_map_c);
    delete[] type_map_c;
  };
  /**
   * @brief Print the summary of DeePMD-kit, including the version and the build
   * information.
   * @param[in] pre The prefix to each line.
   */
  void print_summary(const std::string &pre) const {
    DP_PrintSummary(pre.c_str());
  }

 private:
  DP_DeepPot *dp;
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
  ~DeepPotModelDevi(){};
  /**
   * @brief DP model deviation constructor with initialization.
   * @param[in] models The names of the frozen model file.
   **/
  DeepPotModelDevi(const std::vector<std::string> &models) : dp(nullptr) {
    init(models);
  };
  /**
   * @brief Initialize the DP model deviation.
   * @param[in] model The name of the frozen model file.
   **/
  void init(const std::vector<std::string> &models) {
    if (dp) {
      std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                   "nothing at the second call of initializer"
                << std::endl;
      return;
    }
    std::vector<const char *> cstrings;
    cstrings.reserve(models.size());
    for (std::string const &str : models) cstrings.push_back(str.data());

    dp = DP_NewDeepPotModelDevi(cstrings.data(), cstrings.size());
    DP_CHECK_OK(DP_DeepPotModelDeviCheckOK, dp);
    numb_models = models.size();
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
  void compute(std::vector<double> &ener,
               std::vector<std::vector<VALUETYPE>> &force,
               std::vector<std::vector<VALUETYPE>> &virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box,
               const int nghost,
               const InputNlist &lmp_list,
               const int &ago) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    // memory will be continous for std::vector but not std::vector<std::vector>
    std::vector<double> energy_flat(numb_models);
    std::vector<VALUETYPE> force_flat(numb_models * natoms * 3);
    std::vector<VALUETYPE> virial_flat(numb_models * 9);
    double *ener_ = &energy_flat[0];
    VALUETYPE *force_ = &force_flat[0];
    VALUETYPE *virial_ = &virial_flat[0];

    _DP_DeepPotModelDeviComputeNList<VALUETYPE>(
        dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, ener_,
        force_, virial_, nullptr, nullptr);
    DP_CHECK_OK(DP_DeepPotModelDeviCheckOK, dp);

    // reshape
    ener.resize(numb_models);
    force.resize(numb_models);
    virial.resize(numb_models);
    for (int i = 0; i < numb_models; i++) {
      ener[i] = energy_flat[i];
      force[i].resize(natoms * 3);
      virial[i].resize(9);
      for (int j = 0; j < natoms * 3; j++)
        force[i][j] = force_flat[i * natoms * 3 + j];
      for (int j = 0; j < 9; j++) virial[i][j] = virial_flat[i * 9 + j];
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
  void compute(std::vector<double> &ener,
               std::vector<std::vector<VALUETYPE>> &force,
               std::vector<std::vector<VALUETYPE>> &virial,
               std::vector<std::vector<VALUETYPE>> &atom_energy,
               std::vector<std::vector<VALUETYPE>> &atom_virial,
               const std::vector<VALUETYPE> &coord,
               const std::vector<int> &atype,
               const std::vector<VALUETYPE> &box,
               const int nghost,
               const InputNlist &lmp_list,
               const int &ago) {
    unsigned int natoms = atype.size();
    assert(natoms * 3 == coord.size());
    if (!box.empty()) {
      assert(box.size() == 9);
    }
    const VALUETYPE *coord_ = &coord[0];
    const VALUETYPE *box_ = !box.empty() ? &box[0] : nullptr;
    const int *atype_ = &atype[0];

    std::vector<double> energy_flat(numb_models);
    std::vector<VALUETYPE> force_flat(numb_models * natoms * 3);
    std::vector<VALUETYPE> virial_flat(numb_models * 9);
    std::vector<VALUETYPE> atom_energy_flat(numb_models * natoms);
    std::vector<VALUETYPE> atom_virial_flat(numb_models * natoms * 9);
    double *ener_ = &energy_flat[0];
    VALUETYPE *force_ = &force_flat[0];
    VALUETYPE *virial_ = &virial_flat[0];
    VALUETYPE *atomic_ener_ = &atom_energy_flat[0];
    VALUETYPE *atomic_virial_ = &atom_virial_flat[0];

    _DP_DeepPotModelDeviComputeNList<VALUETYPE>(
        dp, natoms, coord_, atype_, box_, nghost, lmp_list.nl, ago, ener_,
        force_, virial_, atomic_ener_, atomic_virial_);
    DP_CHECK_OK(DP_DeepPotModelDeviCheckOK, dp);

    // reshape
    ener.resize(numb_models);
    force.resize(numb_models);
    virial.resize(numb_models);
    atom_energy.resize(numb_models);
    atom_virial.resize(numb_models);
    for (int i = 0; i < numb_models; i++) {
      ener[i] = energy_flat[i];
      force[i].resize(natoms * 3);
      virial[i].resize(9);
      atom_energy[i].resize(natoms);
      atom_virial[i].resize(natoms * 9);
      for (int j = 0; j < natoms * 3; j++)
        force[i][j] = force_flat[i * natoms * 3 + j];
      for (int j = 0; j < 9; j++) virial[i][j] = virial_flat[i * 9 + j];
      for (int j = 0; j < natoms; j++)
        atom_energy[i][j] = atom_energy_flat[i * natoms + j];
      for (int j = 0; j < natoms * 9; j++)
        atom_virial[i][j] = atom_virial_flat[i * natoms * 9 + j];
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

 private:
  DP_DeepPotModelDevi *dp;
  int numb_models;
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
  ~DeepTensor(){};
  /**
   * @brief DeepTensor constructor with initialization.
   * @param[in] model The name of the frozen model file.
   **/
  DeepTensor(const std::string &model,
             const int &gpu_rank = 0,
             const std::string &name_scope = "")
      : dt(nullptr) {
    init(model, gpu_rank, name_scope);
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
    force.resize(odim * natoms * 3);
    virial.resize(odim * 9);
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
    force.resize(odim * natoms * 3);
    virial.resize(odim * 9);
    atom_virial.resize(odim * natoms * 9);
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
    force.resize(odim * natoms * 3);
    virial.resize(odim * 9);
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
    force.resize(odim * natoms * 3);
    virial.resize(odim * 9);
    atom_virial.resize(odim * natoms * 9);
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
  ~DipoleChargeModifier(){};
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
    init(model, gpu_rank, name_scope);
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
   *natoms x 3.
   * @param[in] datype_ The atom types. The list should contain natoms ints.
   * @param[in] dbox The cell of the region. The array should be of size 9.
   * @param[in] pairs The pairs of atoms. The list should contain npairs pairs
   *of ints.
   * @param[in] delef_ The electric field on each atom. The array should be of
   *size natoms x 3.
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

    dfcorr_.resize(natoms * 3);
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
}  // namespace hpp
}  // namespace deepmd
