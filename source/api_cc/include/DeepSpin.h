// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <cstdint>
#include <memory>

#include "DeepBaseModel.h"
#include "common.h"
#include "neighbor_list.h"

namespace deepmd {
/**
 * @brief Deep Potential.
 **/
class DeepSpinBackend : public DeepBaseModelBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepSpinBackend() {};
  virtual ~DeepSpinBackend() {};
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepSpinBackend(const std::string& model,
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
   * @brief Evaluate the energy, force, magnetic force, virial, atomic energy,
   *and atomic virial by using this DP with spin input.
   * @note The double precision interface is used by i-PI, ABACUS, and
   *CP2k.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] force_mag The magnetic force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
                        std::vector<double>& force_mag,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<double>& spin,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const std::vector<double>& fparam,
                        const std::vector<double>& aparam,
                        const bool atomic) = 0;
  virtual void computew(std::vector<double>& ener,
                        std::vector<float>& force,
                        std::vector<float>& force_mag,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<float>& spin,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const std::vector<float>& fparam,
                        const std::vector<float>& aparam,
                        const bool atomic) = 0;
  /** @} */

  /**
   * @brief Evaluate the energy, force, magnetic force, virial, atomic energy,
   *and atomic virial by using this DP with spin input.
   * @note The double precision interface is used by LAMMPS and AMBER.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] force_mag The magnetic force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
                        std::vector<double>& force_mag,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<double>& spin,
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
                        std::vector<float>& force_mag,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<float>& spin,
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
   * @brief Get dimension of charge/spin condition inputs.
   * Returns 0 for backends that do not support charge/spin conditioning.
   **/
  virtual int dim_chg_spin() const { return 0; }

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
  // that do not support charge/spin do not need any changes.  DeepSpinPTExpt
  // overrides these to thread charge_spin through to the model.
  virtual void computew(std::vector<double>& ener,
                        std::vector<double>& force,
                        std::vector<double>& force_mag,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<double>& spin,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const std::vector<double>& fparam,
                        const std::vector<double>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
             spin, atype, box, fparam, aparam, atomic);
  }
  virtual void computew(std::vector<double>& ener,
                        std::vector<float>& force,
                        std::vector<float>& force_mag,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<float>& spin,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const std::vector<float>& fparam,
                        const std::vector<float>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
             spin, atype, box, fparam, aparam, atomic);
  }
  virtual void computew(std::vector<double>& ener,
                        std::vector<double>& force,
                        std::vector<double>& force_mag,
                        std::vector<double>& virial,
                        std::vector<double>& atom_energy,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<double>& spin,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const int& ago,
                        const std::vector<double>& fparam,
                        const std::vector<double>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
             spin, atype, box, nghost, inlist, ago, fparam, aparam, atomic);
  }
  virtual void computew(std::vector<double>& ener,
                        std::vector<float>& force,
                        std::vector<float>& force_mag,
                        std::vector<float>& virial,
                        std::vector<float>& atom_energy,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<float>& spin,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const int& ago,
                        const std::vector<float>& fparam,
                        const std::vector<float>& aparam,
                        const std::vector<double>& charge_spin,
                        const bool atomic) {
    computew(ener, force, force_mag, virial, atom_energy, atom_virial, coord,
             spin, atype, box, nghost, inlist, ago, fparam, aparam, atomic);
  }

  /**
   * @brief Get the per-type use_spin flags.
   * @return A vector of booleans indicating which atom types have spin enabled.
   *         Empty if the backend does not provide this information.
   **/
  virtual std::vector<bool> get_use_spin() const { return {}; };

  /**
   * @brief GPU-resident compact canonical graph inference backend hook.
   *
   * Given a device-resident dual-CSR graph and the per-node moment, write the
   * per-atom energy, force, magnetic force, and virial back to the device
   * output pointers. The PyTorch Exportable backend overrides this; every
   * other backend inherits the throwing default. The signature is torch-free
   * so the dispatcher stays backend-agnostic and ``libdeepmd_cc`` need not
   * link PyTorch. See DeepSpin::compute_canonical_graph_gpu for the device
   * pointer and graph contracts.
   */
  virtual void compute_canonical_graph_gpu(
      double* d_atom_energy,
      double* d_force,
      double* d_force_mag,
      double* d_atom_virial,
      const std::int64_t* d_atype,
      const std::uint32_t* d_source,
      const float* d_edge_vec,
      const std::int64_t* d_destination_row_ptr,
      const std::int64_t* d_source_row_ptr,
      const std::uint32_t* d_source_order,
      const float* d_spin,
      const int nloc,
      const int nall_nodes,
      const std::int64_t edge_storage);
  virtual bool uses_canonical_graph_inference() const;

  /**
   * @brief Whether the backend serves the loaded artifact under the native
   * spin scheme, in which the magnetic moment is a per-node descriptor input
   * and the model carries no virtual atoms. Backends of the virtual-atom
   * scheme inherit the negative default.
   */
  virtual bool uses_native_spin_scheme() const;
};

/**
 * @brief Deep Potential to automatically switch backends.
 **/
class DeepSpin : public DeepBaseModel {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepSpin();
  virtual ~DeepSpin();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepSpin(const std::string& model,
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
   * @brief Evaluate the energy, force, magnetic force and virial by using this
   *DP with spin input.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] force_mag The magnetic force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */

  /**
   * @brief Evaluate the energy, force, magnetic force and virial by using this
   *DP with spin input.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] force_mag The magnetic force on each atom.
   * @param[out] virial The virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
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
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
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
   * @brief Evaluate the energy, force, magnetic force, virial, atomic energy,
   *and atomic virial by using this DP with spin input.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] force_mag The magnetic force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  template <typename VALUETYPE>
  void compute(std::vector<ENERGYTYPE>& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());
  /** @} */

  /**
   * @brief Evaluate the energy, force, magnetic force, virial, atomic energy,
   *and atomic virial by using this DP with spin input.
   * @param[out] ener The system energy.
   * @param[out] force The force on each atom.
   * @param[out] force_mag The magnetic force on each atom.
   * @param[out] virial The virial.
   * @param[out] atom_energy The atomic energy.
   * @param[out] atom_virial The atomic virial.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
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
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
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
   * @brief Get dimension of the charge/spin condition inputs.
   * @return The dimension of charge_spin; 0 when the model does not support
   *charge/spin conditioning.
   **/
  int dim_chg_spin() const;

  /**
   * @brief Fix the charge/spin condition served for the rest of the run.
   * @param[in] charge_spin The condition, of length ``dim_chg_spin()``.
   **/
  void set_charge_spin(const std::vector<double>& charge_spin);

  /**
   * @brief Get the per-type use_spin flags.
   * @return A vector of booleans indicating which atom types have spin enabled.
   **/
  std::vector<bool> get_use_spin() const;

  /**
   * @brief Whether the loaded artifact uses the compact canonical graph ABI.
   */
  bool uses_canonical_graph_inference() const;

  /**
   * @brief Whether the loaded artifact is served under the native spin
   * scheme rather than the virtual-atom scheme.
   */
  bool uses_native_spin_scheme() const;

  /**
   * @brief Fully device-resident inference for a compact canonical native-spin
   *artifact.
   *
   * The native-spin twin of DeepPot::compute_canonical_graph_gpu: the same
   *dual-CSR compact schema plus the per-node moment, and the magnetic force
   *among the outputs. All pointers reference accelerator memory on the model
   *device and every output is written device-to-device. ``edge_storage`` is
   *the allocated edge capacity; the physical edge count is the last entry of
   *``d_destination_row_ptr`` and the tail beyond it is ignored.
   *
   * @param[out] d_atom_energy Per-atom energy, [nloc].
   * @param[out] d_force Per-node force, [nall_nodes * 3] row-major.
   * @param[out] d_force_mag Per-node magnetic force, [nall_nodes * 3]
   *row-major.
   * @param[out] d_atom_virial Per-node virial, [nall_nodes * 9] row-major.
   * @param[in] d_atype Per-node atom types, [nall_nodes].
   * @param[in] d_source Source-node index per edge, [edge_storage].
   * @param[in] d_edge_vec Destination-major edge vectors, [edge_storage * 3].
   * @param[in] d_destination_row_ptr Destination CSR offsets, [nall_nodes + 1].
   * @param[in] d_source_row_ptr Source CSR offsets, [nall_nodes + 1].
   * @param[in] d_source_order Source-grouped edge positions, [edge_storage].
   * @param[in] d_spin Per-node magnetic moment, [nall_nodes * 3] row-major;
   *ghost rows carry their owner's moment.
   * @param[in] nloc Number of local atoms.
   * @param[in] nall_nodes Graph node count (local + ghost).
   * @param[in] edge_storage Allocated edge capacity.
   */
  void compute_canonical_graph_gpu(double* d_atom_energy,
                                   double* d_force,
                                   double* d_force_mag,
                                   double* d_atom_virial,
                                   const std::int64_t* d_atype,
                                   const std::uint32_t* d_source,
                                   const float* d_edge_vec,
                                   const std::int64_t* d_destination_row_ptr,
                                   const std::int64_t* d_source_row_ptr,
                                   const std::uint32_t* d_source_order,
                                   const float* d_spin,
                                   const int nloc,
                                   const int nall_nodes,
                                   const std::int64_t edge_storage);

 protected:
  std::shared_ptr<deepmd::DeepSpinBackend> dp;
};

class DeepSpinModelDevi : public DeepBaseModelDevi {
 public:
  /**
   * @brief DP model deviation constructor without initialization.
   **/
  DeepSpinModelDevi();
  virtual ~DeepSpinModelDevi();
  /**
   * @brief DP model deviation constructor with initialization.
   * @param[in] models The names of the frozen model files.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_contents The contents of the model files. If it is not
   *empty, DP will read from the strings instead of the files.
   **/
  DeepSpinModelDevi(const std::vector<std::string>& models,
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
   * @brief Evaluate the energy, force and virial by using these DP spin models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_force_mag The magnetic forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<std::vector<VALUETYPE>>& all_force_mag,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());

  /**
   * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial
   *by using these DP spin models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_force_mag The magnetic forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[out] all_atom_energy The atomic energies of all models.
   * @param[out] all_atom_virial The atomic virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<std::vector<VALUETYPE>>& all_force_mag,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               std::vector<std::vector<VALUETYPE>>& all_atom_energy,
               std::vector<std::vector<VALUETYPE>>& all_atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());

  /**
   * @brief Evaluate the energy, force, magnetic force and virial by using these
   *DP spin models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_force_mag The magnetic forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<std::vector<VALUETYPE>>& all_force_mag,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());

  /**
   * @brief Evaluate the energy, force, magnetic force, virial, atomic energy,
   *and atomic virial by using these DP spin models.
   * @param[out] all_ener The system energies of all models.
   * @param[out] all_force The forces on each atom of all models.
   * @param[out] all_force_mag The magnetic forces on each atom of all models.
   * @param[out] all_virial The virials of all models.
   * @param[out] all_atom_energy The atomic energies of all models.
   * @param[out] all_atom_virial The atomic virials of all models.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *nframes x natoms x 3.
   * @param[in] spin The spins of atoms, [0, 0, 0] if no spin. The array should
   *be of size nframes x natoms x 3.
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
               std::vector<std::vector<VALUETYPE>>& all_force_mag,
               std::vector<std::vector<VALUETYPE>>& all_virial,
               std::vector<std::vector<VALUETYPE>>& all_atom_energy,
               std::vector<std::vector<VALUETYPE>>& all_atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& lmp_list,
               const int& ago,
               const std::vector<VALUETYPE>& fparam = std::vector<VALUETYPE>(),
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>(),
               const std::vector<double>& charge_spin = std::vector<double>());

  /**
   * @brief Get the per-type use_spin flags from the first model.
   * @return A vector of booleans indicating which atom types have spin enabled.
   **/
  std::vector<bool> get_use_spin() const;

 protected:
  std::vector<std::shared_ptr<deepmd::DeepSpin>> dps;
};
}  // namespace deepmd
