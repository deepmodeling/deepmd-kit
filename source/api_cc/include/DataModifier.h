// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <memory>

#include "common.h"

namespace deepmd {
/**
 * @brief Dipole charge modifier. (Base class)
 **/
class DipoleChargeModifierBase {
 public:
  /**
   * @brief Dipole charge modifier without initialization.
   **/
  DipoleChargeModifierBase(){};
  /**
   * @brief Dipole charge modifier without initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope The name scope.
   **/
  DipoleChargeModifierBase(const std::string& model,
                           const int& gpu_rank = 0,
                           const std::string& name_scope = "");
  virtual ~DipoleChargeModifierBase(){};
  /**
   * @brief Initialize the dipole charge modifier.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope The name scope.
   **/
  virtual void init(const std::string& model,
                    const int& gpu_rank = 0,
                    const std::string& name_scope = "") = 0;
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
   @{
   **/
  virtual void computew(std::vector<double>& dfcorr_,
                        std::vector<double>& dvcorr_,
                        const std::vector<double>& dcoord_,
                        const std::vector<int>& datype_,
                        const std::vector<double>& dbox,
                        const std::vector<std::pair<int, int>>& pairs,
                        const std::vector<double>& delef_,
                        const int nghost,
                        const InputNlist& lmp_list) = 0;
  virtual void computew(std::vector<float>& dfcorr_,
                        std::vector<float>& dvcorr_,
                        const std::vector<float>& dcoord_,
                        const std::vector<int>& datype_,
                        const std::vector<float>& dbox,
                        const std::vector<std::pair<int, int>>& pairs,
                        const std::vector<float>& delef_,
                        const int nghost,
                        const InputNlist& lmp_list) = 0;
  /** @} */
  /**
   * @brief Get cutoff radius.
   * @return double cutoff radius.
   */
  virtual double cutoff() const = 0;
  /**
   * @brief Get the number of atom types.
   * @return int number of atom types.
   */
  virtual int numb_types() const = 0;
  /**
   * @brief Get the list of sel types.
   * @return The list of sel types.
   */
  virtual const std::vector<int>& sel_types() const = 0;
};

/**
 * @brief Dipole charge modifier.
 **/
class DipoleChargeModifier {
 public:
  /**
   * @brief Dipole charge modifier without initialization.
   **/
  DipoleChargeModifier();
  /**
   * @brief Dipole charge modifier without initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope The name scope.
   **/
  DipoleChargeModifier(const std::string& model,
                       const int& gpu_rank = 0,
                       const std::string& name_scope = "");
  ~DipoleChargeModifier();
  /**
   * @brief Initialize the dipole charge modifier.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope The name scope.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& name_scope = "");
  /**
   * @brief Print the DP summary to the screen.
   * @param[in] pre The prefix to each line.
   **/
  void print_summary(const std::string& pre) const;

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
   *size nloc x 3.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] lmp_list The neighbor list.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& dfcorr_,
               std::vector<VALUETYPE>& dvcorr_,
               const std::vector<VALUETYPE>& dcoord_,
               const std::vector<int>& datype_,
               const std::vector<VALUETYPE>& dbox,
               const std::vector<std::pair<int, int>>& pairs,
               const std::vector<VALUETYPE>& delef_,
               const int nghost,
               const InputNlist& lmp_list);
  /**
   * @brief Get cutoff radius.
   * @return double cutoff radius.
   */
  double cutoff() const;
  /**
   * @brief Get the number of atom types.
   * @return int number of atom types.
   */
  int numb_types() const;
  /**
   * @brief Get the list of sel types.
   * @return The list of sel types.
   */
  const std::vector<int>& sel_types() const;

 private:
  bool inited;
  std::shared_ptr<deepmd::DipoleChargeModifierBase> dcm;
};
}  // namespace deepmd
