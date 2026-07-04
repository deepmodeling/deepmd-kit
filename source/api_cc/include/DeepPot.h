// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <memory>

#include "DeepBaseModel.h"
#include "common.h"
#include "neighbor_list.h"

namespace deepmd {
/**
 * @brief Deep Potential.
 **/
class DeepPotBackend : public DeepBaseModelBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPotBackend() {};
  virtual ~DeepPotBackend() {};
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepPotBackend(const std::string& model,
                 const int& gpu_rank = 0,
                 const std::string& file_content = "");
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  virtual void init(const std::string& model,
                    const int& gpu_rank = 0,
                    const std::string& file_content = "") = 0;

  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @note The double precision interface is used by i-PI, ABACUS, and
   *CP2k.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Request atomic energy and virial if atomic is true.
   * @{
   **/
  virtual void computew(std::vector<double>& ener,
                        std::vector<double>& force,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const std::vector<double>& fparam,
                        const std::vector<double>& aparam,
                        const bool atomic) = 0;
  virtual void computew(std::vector<double>& ener,
                        std::vector<float>& force,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const std::vector<float>& fparam,
                        const std::vector<float>& aparam,
                        const bool atomic) = 0;
  /** @} */
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using this DP.
   * @note The double precision interface is used by LAMMPS and AMBER.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Request atomic energy and virial if atomic is true.
   * @{
   **/
  virtual void computew(std::vector<double>& ener,
                        std::vector<double>& force,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const int& ago,
                        const std::vector<double>& fparam,
                        const std::vector<double>& aparam,
                        const bool atomic) = 0;
  virtual void computew(std::vector<double>& ener,
                        std::vector<float>& force,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const int& ago,
                        const std::vector<float>& fparam,
                        const std::vector<float>& aparam,
                        const bool atomic) = 0;
  /** @} */

  /**
   * @brief Evaluate the energy, force, and virial with the mixed type
   *by using this DP.
   * @note At this time, no external program uses this interface.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The array should be of size nframes x
   *natoms.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] atomic Request atomic energy and virial if atomic is true.
   * @{
   **/
  virtual void computew_mixed_type(std::vector<double>& ener,
                                   std::vector<double>& force,
                                   std::vector<double>& virial,
                                   std::vector<double>& atom_energy,
                                   std::vector<double>& atom_virial,
                                   const int& nframes,
                                   const std::vector<double>& coord,
                                   const std::vector<int>& atype,
                                   const std::vector<double>& box,
                                   const std::vector<double>& fparam,
                                   const std::vector<double>& aparam,
                                   const bool atomic) = 0;
  virtual void computew_mixed_type(std::vector<double>& ener,
                                   std::vector<float>& force,
                                   std::vector<float>& virial,
                                   std::vector<float>& atom_energy,
                                   std::vector<float>& atom_virial,
                                   const int& nframes,
                                   const std::vector<float>& coord,
                                   const std::vector<int>& atype,
                                   const std::vector<float>& box,
                                   const std::vector<float>& fparam,
                                   const std::vector<float>& aparam,
                                   const bool atomic) = 0;
  /** @} */

  /**
   * @brief Get dimension of charge/spin condition inputs.
   * Returns 0 for backends that do not support charge/spin conditioning.
   **/
  virtual int dim_chg_spin() const { return 0; }

  // charge_spin-aware computew overloads.  Default implementations call the
  // existing pure-virtual overloads (ignoring charge_spin) so that backends
  // that do not support charge/spin do not need any changes.  DeepPotPTExpt
  // overrides these to thread charge_spin through to the model.
  virtual void computew(std::vector<double>& ener,
                        std::vector<double>& force,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const std::vector<double>& fparam,
                        const std::vector<double>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
             fparam, aparam, atomic);
  }
  virtual void computew(std::vector<double>& ener,
                        std::vector<float>& force,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const std::vector<float>& fparam,
                        const std::vector<float>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
             fparam, aparam, atomic);
  }
  virtual void computew(std::vector<double>& ener,
                        std::vector<double>& force,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const int& ago,
                        const std::vector<double>& fparam,
                        const std::vector<double>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
             nghost, inlist, ago, fparam, aparam, atomic);
  }
  virtual void computew(std::vector<double>& ener,
                        std::vector<float>& force,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const int& ago,
                        const std::vector<float>& fparam,
                        const std::vector<float>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
             nghost, inlist, ago, fparam, aparam, atomic);
  }
  virtual void computew_mixed_type(std::vector<double>& ener,
                                   std::vector<double>& force,
                                   std::vector<double>& virial,
                                   std::vector<double>& atom_energy,
                                   std::vector<double>& atom_virial,
                                   const int& nframes,
                                   const std::vector<double>& coord,
                                   const std::vector<int>& atype,
                                   const std::vector<double>& box,
                                   const std::vector<double>& fparam,
                                   const std::vector<double>& aparam,
                                   const std::vector<double>& charge_spin,
                                   const bool atomic) {
    computew_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                        coord, atype, box, fparam, aparam, atomic);
  }
  virtual void computew_mixed_type(std::vector<double>& ener,
                                   std::vector<float>& force,
                                   std::vector<float>& virial,
                                   std::vector<float>& atom_energy,
                                   std::vector<float>& atom_virial,
                                   const int& nframes,
                                   const std::vector<float>& coord,
                                   const std::vector<int>& atype,
                                   const std::vector<float>& box,
                                   const std::vector<float>& fparam,
                                   const std::vector<float>& aparam,
                                   const std::vector<double>& charge_spin,
                                   const bool atomic) {
    computew_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                        coord, atype, box, fparam, aparam, atomic);
  }
  /**
   * @brief GPU-resident edge inference backend hook.
   *
   * Given device-resident edge tensors, write the per-atom energy, force, and
   * virial back to the device output pointers. The PyTorch Exportable backend
   * overrides this; every other backend inherits the throwing default. The
   * signature is torch-free so the dispatcher stays backend-agnostic and
   * ``libdeepmd_cc`` need not link PyTorch. See DeepPot::compute_edges_gpu for
   * the device pointer, graph, and communication contracts.
   */
  virtual void compute_edges_gpu(double* d_atom_energy,
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
                                 const InputNlist* comm_nlist);
  virtual void compute_edges_gpu(double* d_atom_energy,
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
                                 const InputNlist* comm_nlist);
  virtual bool supports_device_edge_inference() const;
  virtual bool uses_fp32_edge_vectors() const;
};

/**
 * @brief Deep Potential to automatically switch backends.
 **/
class DeepPot : public DeepBaseModel {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPot();
  virtual ~DeepPot();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepPot(const std::string& model,
          const int& gpu_rank = 0,
          const std::string& file_content = "");
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

  /**
   * @brief Evaluate the energy, force and virial by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   * @{
   **/
  template <typename VALUETYPE>
  void compute(ENERGYTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */
  /**
   * @brief Evaluate the energy, force and virial by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] inlist The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   * @{
   **/
  template <typename VALUETYPE>
  void compute(ENERGYTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& inlist,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& inlist,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */
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
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   * @{
   **/
  template <typename VALUETYPE>
  void compute(ENERGYTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */

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
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   * @{
   **/
  template <typename VALUETYPE>
  void compute(ENERGYTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */
  /**
   * @brief Evaluate the energy, force, and virial with the mixed type
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The array should be of size nframes x
   *natoms.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   * @{
   **/
  template <typename VALUETYPE>
  void compute_mixed_type(
      ENERGYTYPE& ener,
      std::vector<VALUETYPE>& force,
      std::vector<VALUETYPE>& virial,
      const int& nframes,
      const std::vector<VALUETYPE>& coord,
      const std::vector<int>& atype,
      const std::vector<VALUETYPE>& box,
      const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
      const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute_mixed_type(
      std::vector<ENERGYTYPE>& ener,
      std::vector<VALUETYPE>& force,
      std::vector<VALUETYPE>& virial,
      const int& nframes,
      const std::vector<VALUETYPE>& coord,
      const std::vector<int>& atype,
      const std::vector<VALUETYPE>& box,
      const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
      const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */
  /**
   * @brief Evaluate the energy, force, and virial with the mixed type
   *by using this DP.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] nframes The number of frames.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The array should be of size nframes x
   *natoms.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   * @{
   **/
  template <typename VALUETYPE>
  void compute_mixed_type(
      ENERGYTYPE& ener,
      std::vector<VALUETYPE>& force,
      std::vector<VALUETYPE>& virial,
      std::vector<VALUETYPE>& atom_energy,
      std::vector<VALUETYPE>& atom_virial,
      const int& nframes,
      const std::vector<VALUETYPE>& coord,
      const std::vector<int>& atype,
      const std::vector<VALUETYPE>& box,
      const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
      const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute_mixed_type(
      std::vector<ENERGYTYPE>& ener,
      std::vector<VALUETYPE>& force,
      std::vector<VALUETYPE>& virial,
      std::vector<VALUETYPE>& atom_energy,
      std::vector<VALUETYPE>& atom_virial,
      const int& nframes,
      const std::vector<VALUETYPE>& coord,
      const std::vector<int>& atype,
      const std::vector<VALUETYPE>& box,
      const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
      const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
      const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */

  /**
   * @brief Fully device-resident inference for exported edge-input or
   * graph-input .pt2 models.
   *
   * Forwards to the PyTorch Exportable (.pt2) backend's GPU edge path; raising
   * if the active backend is not ``DeepPotPTExpt``.  All pointers reference GPU
   * memory on the model's device.  See
   * ``DeepPotPTExpt::compute_edges_gpu`` for the edge contract.  This signature
   * is intentionally torch-free so MD-engine call sites need no PyTorch
   * headers.
   *
   * @param[out] d_atom_energy Per-atom energy, GPU [nloc].
   * @param[out] d_force Per-atom force, GPU [nloc * 3] row-major.
   * @param[out] d_atom_virial Per-atom virial, GPU [nloc * 9] row-major.
   * @param[in] d_coord Local coordinates, GPU [nloc * 3] row-major.
   * @param[in] d_atype Local atom types, GPU [nloc].
   * @param[in] d_edge_index Local edge graph, GPU [2 * nedge].
   * @param[in] d_edge_vec Minimum-image bond vectors, GPU [nedge * 3].
   * @param[in] nloc Number of local atoms.
   * @param[in] nedge Number of physical edges.
   */
  void compute_edges_gpu(double* d_atom_energy,
                         double* d_force,
                         double* d_atom_virial,
                         const double* d_coord,
                         const int* d_atype,
                         const int* d_edge_index,
                         const double* d_edge_vec,
                         const int nloc,
                         const int nedge);

  /**
   * @brief GPU-resident edge inference with runtime frame / atomic parameters.
   *
   * As the parameter-free overload, but ``fparam`` (global, ``dfparam`` values)
   * and ``aparam`` (per-atom, ``nloc * daparam`` values) override the model's
   * stored defaults. Empty vectors fall back to the stored default fparam and
   * to no aparam, so the two overloads coincide.
   *
   * @param[in] fparam Runtime frame parameters, or empty for the model default.
   * @param[in] aparam Runtime per-atom parameters (row-major [nloc, daparam]),
   *   or empty for none.
   * @param[in] nall_nodes Total graph node count; 0 (or nloc) folds ghosts onto
   *   local owners (single domain), while nall_nodes > nloc keeps the extended
   *   (local + ghost) node set for a domain-decomposed run.
   * @param[in] comm_nlist Communication neighbor list (send/recv swaps) for the
   *   extended node set. Required for a message-passing model under domain
   *   decomposition, where ghost features are exchanged across ranks inside the
   *   forward pass; nullptr otherwise.
   */
  void compute_edges_gpu(double* d_atom_energy,
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
                         const int nall_nodes = 0,
                         const InputNlist* comm_nlist = nullptr);

  /**
   * @brief Device-edge inference with FP32 edge vectors.
   *
   * Call this overload only when ``uses_fp32_edge_vectors()`` is true.
   */
  void compute_edges_gpu(double* d_atom_energy,
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
                         const int nall_nodes = 0,
                         const InputNlist* comm_nlist = nullptr);

  /**
   * @brief Whether the loaded artifact supports device-edge inference.
   */
  bool supports_device_edge_inference() const;

  /**
   * @brief Whether the loaded artifact expects FP32 device edge vectors.
   */
  bool uses_fp32_edge_vectors() const;

  int dim_chg_spin() const;

 protected:
  std::shared_ptr<deepmd::DeepPotBackend> dp;
};

class DeepPotModelDevi : public DeepBaseModelDevi {
 public:
  /**
   * @brief DP model deviation constructor without initialization.
   **/
  DeepPotModelDevi();
  virtual ~DeepPotModelDevi();
  /**
   * @brief DP model deviation constructor with initialization.
   * @param[in] models The names of the frozen model files.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_contents The contents of the model files. If it is not
   *empty, DP will read from the strings instead of the files.
   **/
  DeepPotModelDevi(const std::vector<std::string>& models,
                   const int& gpu_rank = 0,
                   const std::vector<std::string>& file_contents =
                       std::vector<std::string>());
  /**
   * @brief Initialize the DP model deviation contrcutor.
   * @param[in] models The names of the frozen model files.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_contents The contents of the model files. If it is not
   *empty, DP will read from the strings instead of the files.
   **/
  void init(const std::vector<std::string>& models,
            const int& gpu_rank = 0,
            const std::vector<std::string>& file_contents =
                std::vector<std::string>());

  /**
   * @brief Get the dimension of the charge/spin input.
   * @return The dimension of the charge/spin input (0 if the models have no
   *charge/spin embedding). Taken from the first model; all models are assumed
   *to share the same value.
   **/
  int dim_chg_spin() const {
    return numb_models > 0 ? dps[0]->dim_chg_spin() : 0;
  };

  /**
   * @brief Evaluate the energy, force and virial by using these DP models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam. dim_aparam. Then all frames and atoms are provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& all_ener,
               std::vector<std::vector<VALUETYPE>>& all_force,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());

  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using these DP models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[out] all_atom_energy The atomic energies of all models.
   * @param[out] all_atom_virial The atomic virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam. dim_aparam. Then all frames and atoms are provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& all_ener,
               std::vector<std::vector<VALUETYPE>>& all_force,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               std::vector<std::vector<VALUETYPE>>& all_atom_energy,
               std::vector<std::vector<VALUETYPE>>& all_atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());

  /**
   * @brief Evaluate the energy, force and virial by using these DP models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam. dim_aparam. Then all frames and atoms are provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& all_ener,
               std::vector<std::vector<VALUETYPE>>& all_force,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using these DP models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[out] all_atom_energy The atomic energies of all models.
   * @param[out] all_atom_virial The atomic virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size nframes
   *x 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The input neighbour list.
   * @param[in] ago Update the internal neighbour list if ago is 0.
   * @param[in] fparam The frame parameter. The array can be of size :
   * nframes x dim_fparam.
   * dim_fparam. Then all frames are assumed to be provided with the same
   *fparam.
   * @param[in] aparam The atomic parameter The array can be of size :
   * nframes x natoms x dim_aparam.
   * natoms x dim_aparam. Then all frames are assumed to be provided with the
   *same aparam. dim_aparam. Then all frames and atoms are provided with the
   *same aparam.
   * @param[in] charge_spin The charge/spin parameter. The array can be of size
   *nframes x dim_chg_spin.
   * dim_chg_spin. Then all frames are assumed to be provided with the same
   *charge_spin. Leave it empty to use the model's stored default_chg_spin.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& all_ener,
               std::vector<std::vector<VALUETYPE>>& all_force,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               std::vector<std::vector<VALUETYPE>>& all_atom_energy,
               std::vector<std::vector<VALUETYPE>>& all_atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());

 protected:
  std::vector<std::shared_ptr<deepmd::DeepPot>> dps;
};
}  // namespace deepmd
