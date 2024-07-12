// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include "DataModifier.h"
#include "common.h"
#include "commonTF.h"

namespace deepmd {
/**
 * @brief Dipole charge modifier.
 **/
class DipoleChargeModifierTF : public DipoleChargeModifierBase {
 public:
  /**
   * @brief Dipole charge modifier without initialization.
   **/
  DipoleChargeModifierTF();
  /**
   * @brief Dipole charge modifier without initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope The name scope.
   **/
  DipoleChargeModifierTF(const std::string& model,
                         const int& gpu_rank = 0,
                         const std::string& name_scope = "");
  ~DipoleChargeModifierTF();
  /**
   * @brief Initialize the dipole charge modifier.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope The name scope.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& name_scope = "");

 private:
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

 public:
  /**
   * @brief Get cutoff radius.
   * @return double cutoff radius.
   */
  double cutoff() const {
    assert(inited);
    return rcut;
  };
  /**
   * @brief Get the number of atom types.
   * @return int number of atom types.
   */
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  /**
   * @brief Get the list of sel types.
   * @return The list of sel types.
   */
  const std::vector<int>& sel_types() const {
    assert(inited);
    return sel_type;
  };
  void computew(std::vector<double>& dfcorr_,
                std::vector<double>& dvcorr_,
                const std::vector<double>& dcoord_,
                const std::vector<int>& datype_,
                const std::vector<double>& dbox,
                const std::vector<std::pair<int, int>>& pairs,
                const std::vector<double>& delef_,
                const int nghost,
                const InputNlist& lmp_list);
  void computew(std::vector<float>& dfcorr_,
                std::vector<float>& dvcorr_,
                const std::vector<float>& dcoord_,
                const std::vector<int>& datype_,
                const std::vector<float>& dbox,
                const std::vector<std::pair<int, int>>& pairs,
                const std::vector<float>& delef_,
                const int nghost,
                const InputNlist& lmp_list);

 private:
  tensorflow::Session* session;
  std::string name_scope, name_prefix;
  int num_intra_nthreads, num_inter_nthreads;
  tensorflow::GraphDef* graph_def;
  bool inited;
  double rcut;
  int dtype;
  double cell_size;
  int ntypes;
  std::string model_type;
  std::vector<int> sel_type;
  template <class VT>
  VT get_scalar(const std::string& name) const;
  template <class VT>
  void get_vector(std::vector<VT>& vec, const std::string& name) const;
  template <typename MODELTYPE, typename VALUETYPE>
  void run_model(std::vector<VALUETYPE>& dforce,
                 std::vector<VALUETYPE>& dvirial,
                 tensorflow::Session* session,
                 const std::vector<std::pair<std::string, tensorflow::Tensor>>&
                     input_tensors,
                 const AtomMap& atommap,
                 const int nghost);
};
}  // namespace deepmd
