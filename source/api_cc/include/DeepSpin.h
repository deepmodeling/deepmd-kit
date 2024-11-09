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
   * @note The double precision interface is used by i-PI, GROMACS, ABACUS, and
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());
  /** @} */
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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());

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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());

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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());

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
               const std::vector<VALUETYPE>& aparam = std::vector<VALUETYPE>());

 protected:
  std::vector<std::shared_ptr<deepmd::DeepSpin>> dps;
};
}  // namespace deepmd
