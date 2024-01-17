// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <memory>

#include "common.h"
#include "neighbor_list.h"

namespace deepmd {
/**
 * @brief Deep Tensor.
 **/
class DeepTensorBase {
 public:
  /**
   * @brief Deep Tensor constructor without initialization.
   **/
  DeepTensorBase(){};
  virtual ~DeepTensorBase(){};
  /**
   * @brief Deep Tensor constructor with initialization..
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope Name scopes of operations.
   **/
  DeepTensorBase(const std::string& model,
                 const int& gpu_rank = 0,
                 const std::string& name_scope = "");
  /**
   * @brief Initialize the Deep Tensor.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope Name scopes of operations.
   **/
  virtual void init(const std::string& model,
                    const int& gpu_rank = 0,
                    const std::string& name_scope = "") = 0;
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evalute.
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
  virtual void computew(std::vector<double>& global_tensor,
                        std::vector<double>& force,
                        std::vector<double>& virial,
                        std::vector<double>& atom_tensor,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const bool request_deriv) = 0;
  virtual void computew(std::vector<float>& global_tensor,
                        std::vector<float>& force,
                        std::vector<float>& virial,
                        std::vector<float>& atom_tensor,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const bool request_deriv) = 0;
  /** @} */
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evalute.
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
  virtual void computew(std::vector<double>& global_tensor,
                        std::vector<double>& force,
                        std::vector<double>& virial,
                        std::vector<double>& atom_tensor,
                        std::vector<double>& atom_virial,
                        const std::vector<double>& coord,
                        const std::vector<int>& atype,
                        const std::vector<double>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const bool request_deriv) = 0;
  virtual void computew(std::vector<float>& global_tensor,
                        std::vector<float>& force,
                        std::vector<float>& virial,
                        std::vector<float>& atom_tensor,
                        std::vector<float>& atom_virial,
                        const std::vector<float>& coord,
                        const std::vector<int>& atype,
                        const std::vector<float>& box,
                        const int nghost,
                        const InputNlist& inlist,
                        const bool request_deriv) = 0;
  /** @} */
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  virtual double cutoff() const = 0;
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  virtual int numb_types() const = 0;
  /**
   * @brief Get the output dimension.
   * @return The output dimension.
   **/
  virtual int output_dim() const = 0;
  /**
   * @brief Get the list of sel types.
   * @return The list of sel types.
   */
  virtual const std::vector<int>& sel_types() const = 0;
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  virtual void get_type_map(std::string& type_map) = 0;
};

/**
 * @brief Deep Tensor.
 **/
class DeepTensor {
 public:
  /**
   * @brief Deep Tensor constructor without initialization.
   **/
  DeepTensor();
  ~DeepTensor();
  /**
   * @brief Deep Tensor constructor with initialization..
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] name_scope Name scopes of operations.
   **/
  DeepTensor(const std::string& model,
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
  /**
   * @brief Print the DP summary to the screen.
   * @param[in] pre The prefix to each line.
   **/
  void print_summary(const std::string& pre) const;

  /**
   * @brief Evaluate the value by using this model.
   * @param[out] value The value to evalute, usually would be the atomic tensor.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& value,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box);
  /**
   * @brief Evaluate the value by using this model.
   * @param[out] value The value to evalute, usually would be the atomic tensor.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] inlist The input neighbour list.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& value,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& inlist);
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evalute.
   * @param[out] force The component-wise force of the global tensor, size odim
   *x natoms x 3.
   * @param[out] virial The component-wise virial of the global tensor, size
   *odim x 9.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& global_tensor,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box);
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evalute.
   * @param[out] force The component-wise force of the global tensor, size odim
   *x natoms x 3.
   * @param[out] virial The component-wise virial of the global tensor, size
   *odim x 9.
   * @param[in] coord The coordinates of atoms. The array should be of size
   *natoms x 3.
   * @param[in] atype The atom types. The list should contain natoms ints.
   * @param[in] box The cell of the region. The array should be of size 9.
   * @param[in] nghost The number of ghost atoms.
   * @param[in] inlist The input neighbour list.
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& global_tensor,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const int nghost,
               const InputNlist& inlist);
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evalute.
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
   **/
  template <typename VALUETYPE>
  void compute(std::vector<VALUETYPE>& global_tensor,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_tensor,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box);
  /**
   * @brief Evaluate the global tensor and component-wise force and virial.
   * @param[out] global_tensor The global tensor to evalute.
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
               const InputNlist& inlist);
  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const;
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const;
  /**
   * @brief Get the output dimension.
   * @return The output dimension.
   **/
  int output_dim() const;
  /**
   * @brief Get the list of sel types.
   * @return The list of sel types.
   */
  const std::vector<int>& sel_types() const;
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string& type_map);

 private:
  bool inited;
  std::shared_ptr<deepmd::DeepTensorBase> dt;
};
}  // namespace deepmd
