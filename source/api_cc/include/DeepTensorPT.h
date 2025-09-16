// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include "DeepTensor.h"

namespace deepmd {
/**
 * @brief PyTorch implementation for Deep Tensor.
 **/
class DeepTensorPT : public DeepTensorBase {
 public:
  /**
   * @brief Deep Tensor constructor without initialization.
   **/
  DeepTensorPT();
  virtual ~DeepTensorPT();
  /**
   * @brief Deep Tensor constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope Name scopes of operations.
   **/
  DeepTensorPT(const std::string& model,
               const int& gpu_rank = 0,
               const std::string& name_scope = "");
  /**
   * @brief Initialize the Deep Tensor.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope Name scopes of operations.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& name_scope = "");

 private:
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evaluate.
   * @param[out] force The component-wise force of the global tensor, size odim
   *x natoms x 3.
   * @param[out] virial The component-wise virial of the global tensor, size
   *odim x 9.
   * @param[out] atom_tensor The atomic tensor value of the model, size natoms x
   *odim.
   * @param[out] atom_virial The component-wise atomic virial of the global
   *tensor, size odim x natoms x 9.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   * @param[in] request_deriv Whether to request the derivative of the global
   * tensor, including force and virial.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& global_tensor,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_tensor,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const bool request_deriv);
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evaluate.
   * @param[out] force The component-wise force of the global tensor, size odim
   *x natoms x 3.
   * @param[out] virial The component-wise virial of the global tensor, size
   *odim x 9.
   * @param[out] atom_tensor The atomic tensor value of the model, size natoms x
   *odim.
   * @param[out] atom_virial The component-wise atomic virial of the global
   *tensor, size odim x natoms x 9.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] inlist The input neighbour list.
   * @param[in] request_deriv Whether to request the derivative of the global
   * tensor, including force and virial.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& global_tensor,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_tensor,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& inlist,
               const bool request_deriv);

 public:
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(inited);
    return rcut;
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  /**
   * @brief Get the output dimension.
   * @return The output dimension.
   **/
  int output_dim() const {
    assert(inited);
    return odim;
  };
  /**
   * @brief Get the list of sel types.
   * @return The list of sel types.
   */
  const std::vector<int>& sel_types() const {
    assert(inited);
    return sel_type;
  };
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string& type_map);

  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evaluate.
   * @param[out] force The component-wise force of the global tensor, size odim
   *x natoms x 3.
   * @param[out] virial The component-wise virial of the global tensor, size
   *odim x 9.
   * @param[out] atom_tensor The atomic tensor value of the model, size natoms x
   *odim.
   * @param[out] atom_virial The component-wise atomic virial of the global
   *tensor, size odim x natoms x 9.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   * @param[in] request_deriv Whether to request the derivative of the global
   * tensor, including force and virial.
   * @{
   **/
  void computew(std::vector<double>& global_tensor,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_tensor,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const bool request_deriv);
  void computew(std::vector<float>& global_tensor,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_tensor,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const bool request_deriv);
  /** @} */
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evaluate.
   * @param[out] force The component-wise force of the global tensor, size odim
   *x natoms x 3.
   * @param[out] virial The component-wise virial of the global tensor, size
   *odim x 9.
   * @param[out] atom_tensor The atomic tensor value of the model, size natoms x
   *odim.
   * @param[out] atom_virial The component-wise atomic virial of the global
   *tensor, size odim x natoms x 9.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] inlist The input neighbour list.
   * @param[in] request_deriv Whether to request the derivative of the global
   * tensor, including force and virial.
   * @{
   **/
  void computew(std::vector<double>& global_tensor,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_tensor,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const int nghost,
                const InputNlist& inlist,
                const bool request_deriv);
  void computew(std::vector<float>& global_tensor,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_tensor,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const int nghost,
                const InputNlist& inlist,
                const bool request_deriv);
  /** @} */

 private:
  int num_intra_nthreads, num_inter_nthreads;
  bool inited;
  double rcut;
  int ntypes;
  mutable int odim;
  std::vector<int> sel_type;
  std::string name_scope;
  // PyTorch module and device management
  mutable torch::jit::script::Module module;
  int gpu_id;
  bool gpu_enabled;
  NeighborListData nlist_data;
  // Neighbor list tensors for efficient computation
  at::Tensor firstneigh_tensor;

  /**
   * @brief Translate PyTorch exceptions to the DeePMD-kit exception.
   * @param[in] f The function to run.
   * @example translate_error([&](){...});
   */
  void translate_error(std::function<void()> f);
};

}  // namespace deepmd
