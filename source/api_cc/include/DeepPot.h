// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>

#include "DeepBaseModel.h"
#include "canonical_graph.h"
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

  /**
   * @brief Reject a charge/spin condition no embedding table can address.
   *
   * A descriptor embeds the condition by gathering one row of a charge table
   * and one row of a multiplicity table, whose geometry belongs to the
   * feature rather than to any one descriptor. Neither the gather nor the
   * compiled kernel bounds-checks the row, so a value outside a table reads
   * past it and a fractional value is truncated onto a neighbouring row,
   * both of which answer for a state the caller never asked for. The bounds
   * mirror ``deepmd/utils/charge_state.py``, which holds the same contract
   * for the Python boundaries.
   *
   * Both tests are relational rather than equalities, which keeps them exact
   * and lets a NaN, which compares false against everything, fall out of the
   * range test rather than through it.
   *
   * @param[in] charge_spin The condition, as ``{charge, multiplicity}``.
   **/
  static void require_addressable_charge_spin(
      const std::vector<double>& charge_spin) {
    static const char* const names[] = {"charge", "multiplicity"};
    static const double lows[] = {-100.0, 0.0};
    static const double highs[] = {100.0, 100.0};
    const std::size_t checked = std::min<std::size_t>(charge_spin.size(), 2);
    for (std::size_t ii = 0; ii < checked; ++ii) {
      const double value = charge_spin[ii];
      // Flooring leaves an integer where it is and moves everything else
      // down, so a strict drop is exactly a fractional part.
      if (std::floor(value) < value) {
        throw deepmd::deepmd_exception(
            std::string("the ") + names[ii] +
            " must be an integer, which indexes one row of its embedding "
            "table, but is " +
            std::to_string(value));
      }
      if (!(value >= lows[ii] && value < highs[ii])) {
        throw deepmd::deepmd_exception(
            std::string("the ") + names[ii] + " must lie in [" +
            std::to_string(lows[ii]) + ", " + std::to_string(highs[ii]) +
            "), which its embedding table covers, but is " +
            std::to_string(value));
      }
    }
  }

  /**
   * @brief Fix the charge/spin condition served for the rest of the run.
   *
   * An override is needed only where the condition has to be folded into state
   * that is built ahead of the evaluations using it, as in a compressed model
   * whose tables are specialized to one condition. A backend that reads the
   * condition as an ordinary per-call input has nothing to install, so the
   * default is a no-op rather than an error: the condition still reaches such
   * a backend on every evaluation, through the charge_spin argument of
   * computew(). The request is refused only by a model that carries no
   * charge/spin conditioning at all, which no route can honour.
   *
   * @param[in] charge_spin The condition, of length ``dim_chg_spin()``.
   **/
  virtual void set_charge_spin(const std::vector<double>& charge_spin) {
    if (dim_chg_spin() == 0) {
      throw deepmd::deepmd_exception(
          "this model does not support a charge/spin condition");
    }
  }

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
                                 const int nedge);
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
  virtual void compute_canonical_graph_gpu(
      double* d_atom_energy,
      double* d_force,
      double* d_virial,
      double* d_atom_virial,
      const std::int64_t* d_atype,
      const void* d_source,
      DP_CanonicalIndexWidth index_width,
      const float* d_edge_vec,
      const std::int64_t* d_destination_row_ptr,
      const std::int64_t* d_source_row_ptr,
      const std::int64_t* d_source_order,
      const double* d_coord,
      void* world,
      const std::int64_t nloc,
      const std::int64_t nall_nodes,
      const std::int64_t edge_storage,
      void* stream);
  virtual bool supports_device_edge_inference() const;
  virtual bool uses_fp32_edge_vectors() const;
  virtual bool uses_canonical_graph_inference() const;
  virtual bool canonical_graph_requires_source_csr() const;
  virtual bool has_long_range() const;
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
   * @param[in] d_edge_index Destination-major local edge graph, GPU
   *   [2 * nedge].
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
   * @param[in] fparam Host-resident runtime frame parameters, or empty for the
   *   model default.
   * @param[in] aparam Host-resident runtime per-atom parameters (row-major
   *   [nloc, daparam]), or empty for none.
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

  /**
   * @brief Whether the loaded artifact uses the compact canonical graph ABI.
   */
  bool uses_canonical_graph_inference() const;

  /**
   * @brief Whether every compact canonical evaluation needs the source CSR.
   *
   * An artifact whose force assembly groups edges by source reads the source
   * CSR on every call. Any other artifact reads it only for the per-atom
   * virial and accepts null source CSR pointers when that is not requested.
   */
  bool canonical_graph_requires_source_csr() const;

  /**
   * @brief Whether the model carries a long-range module.
   *
   * Such a module sums its interaction over every pair of atoms of the
   * system, so the model evaluates only a complete non-periodic system, and
   * compute_canonical_graph_gpu() passes it the positions of the nodes and,
   * on several ranks, the communicator.
   */
  bool has_long_range() const;

  /**
   * @brief Evaluate a compact canonical graph from device-resident buffers.
   *
   * Every pointer resides on the model device.
   *
   * @param[out] d_atom_energy Per-local-node energy, shape ``(nloc)``.
   * @param[out] d_force Per-node force, shape ``(nall_nodes, 3)``.
   * @param[out] d_virial Virial of the graph, nine components, or null when
   * it is not requested.
   * @param[out] d_atom_virial Per-node virial, shape ``(nall_nodes, 9)``, or
   * null when the per-atom virial is not requested.
   * @param[in] d_atype Per-node atom types, shape ``(nall_nodes)``.
   * @param[in] d_source Source node per edge storage slot.
   * @param[in] index_width Width of each source index, four or eight bytes.
   * @param[in] d_edge_vec FP32 edge vectors, shape ``(edge_storage, 3)``.
   * @param[in] d_destination_row_ptr Destination CSR offsets.
   * @param[in] d_source_row_ptr Source CSR offsets. May be null together with
   * ``d_source_order`` and ``d_atom_virial`` unless
   * canonical_graph_requires_source_csr().
   * @param[in] d_source_order Source-grouped int64 edge storage positions.
   * @param[in] d_coord Node positions in the length unit of the model, shape
   * ``(nall_nodes, 3)``, in the order of ``d_atype``, or null. Only a model
   * with a long-range module (has_long_range()) reads them, and it requires
   * them; every other model ignores them.
   * @param[in] world Pointer to the MPI communicator of the caller (an
   * ``MPI_Comm``), as InputNlist::world, when the system spans several
   * ranks, or null on a single rank. A model with a long-range module
   * evaluates a complete non-periodic system: on a single rank its nodes are
   * the whole system (``nall_nodes == nloc``); on several ranks it requires
   * the communicator, reads the positions of the nodes each rank owns, and
   * is evaluated on every rank at every step, a rank without nodes included
   * (``nall_nodes``, ``nloc`` and ``edge_storage`` zero), whose buffers
   * sized by the nodes or by the edge storage may then be null. Every other
   * model ignores it.
   * @param[in] nloc Number of owned local nodes.
   * @param[in] nall_nodes Total local-plus-halo node count.
   * @param[in] edge_storage Number of edge storage slots.
   * @param[in] stream Accelerator stream (``cudaStream_t``) of the model
   * device on which the inputs were produced and the outputs will be
   * consumed; a non-null stream requires a CUDA build. The evaluation is
   * enqueued on it without a synchronization of the stream. A graph without
   * halo nodes is captured into a CUDA graph on the second consecutive call
   * with the same buffers and extents and replayed while they stay the same,
   * except where a model with a long-range module receives a communicator.
   * The host waits for the device when a call captures or releases a graph,
   * which happens on that second call and on a call whose buffers or extents
   * change. set_charge_spin waits for the work in flight on the model device
   * before it replaces the constants and for the replacement before it
   * returns, and the caller completes the work on its stream before it
   * destroys the model. A null stream runs on the current stream of the
   * calling thread for the model device and waits for that stream before
   * returning. Models in different threads of one process evaluate
   * concurrently, and their captures, releases of graphs and charge-state
   * updates, which wait for the whole device, take turns; CUDA work outside
   * DeePMD-kit that waits for the whole device while a capture is under way
   * invalidates it, and the model then evaluates without replay.
   */
  void compute_canonical_graph_gpu(double* d_atom_energy,
                                   double* d_force,
                                   double* d_virial,
                                   double* d_atom_virial,
                                   const std::int64_t* d_atype,
                                   const void* d_source,
                                   DP_CanonicalIndexWidth index_width,
                                   const float* d_edge_vec,
                                   const std::int64_t* d_destination_row_ptr,
                                   const std::int64_t* d_source_row_ptr,
                                   const std::int64_t* d_source_order,
                                   const double* d_coord,
                                   void* world,
                                   const std::int64_t nloc,
                                   const std::int64_t nall_nodes,
                                   const std::int64_t edge_storage,
                                   void* stream = nullptr);

  int dim_chg_spin() const;

  /**
   * @brief Fix the charge/spin condition served for the rest of the run.
   * @param[in] charge_spin The condition, of length ``dim_chg_spin()``.
   **/
  void set_charge_spin(const std::vector<double>& charge_spin);

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
   * @brief Whether any of the models carries a long-range module; see
   * DeepPot::has_long_range().
   **/
  bool has_long_range() const {
    return std::any_of(dps.begin(), dps.end(), [](const auto& model) {
      return model->has_long_range();
    });
  };

  /**
   * @brief Fix the charge/spin condition served for the rest of the run.
   * Applied to every model, so that the deviation is taken between models
   * under the same condition.
   * @param[in] charge_spin The condition, of length ``dim_chg_spin()``.
   **/
  void set_charge_spin(const std::vector<double>& charge_spin) {
    for (unsigned ii = 0; ii < dps.size(); ++ii) {
      dps[ii]->set_charge_spin(charge_spin);
    }
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
