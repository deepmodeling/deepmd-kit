// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <memory>

#include "common.h"
#include "neighbor_list.h"

namespace deepmd {
/**
 * @brief Deep Potential Base Model.
 **/
class DeepBaseModelBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepBaseModelBackend() {};
  virtual ~DeepBaseModelBackend() {};
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepBaseModelBackend(const std::string& model,
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
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  virtual int numb_types_spin() const = 0;
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  virtual int dim_fparam() const = 0;
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  virtual int dim_aparam() const = 0;
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  virtual void get_type_map(std::string& type_map) = 0;

  /**
   * @brief Get whether the atom dimension of aparam is nall instead of fparam.
   * @param[out] aparam_nall whether the atom dimension of aparam is nall
   *instead of fparam.
   **/
  virtual bool is_aparam_nall() const = 0;
};

/**
 * @brief Deep Potential Base Model to automatically switch backends.
 **/
class DeepBaseModel {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepBaseModel();
  virtual ~DeepBaseModel();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the frozen model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepBaseModel(const std::string& model,
                const int& gpu_rank = 0,
                const std::string& file_content = "");

  /**
   * @brief Print the DP summary to the screen.
   * @param[in] pre The prefix to each line.
   **/
  void print_summary(const std::string& pre) const;

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
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  int numb_types_spin() const;
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  int dim_fparam() const;
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  int dim_aparam() const;
  /**
   * @brief Get the type map (element name of the atom types) of this model.
   * @param[out] type_map The type map of this model.
   **/
  void get_type_map(std::string& type_map);

  /**
   * @brief Get whether the atom dimension of aparam is nall instead of fparam.
   * @param[out] aparam_nall whether the atom dimension of aparam is nall
   *instead of fparam.
   **/
  bool is_aparam_nall() const;

 protected:
  bool inited;
  std::shared_ptr<deepmd::DeepBaseModelBackend> dpbase;
};

class DeepBaseModelDevi {
 public:
  /**
   * @brief DP model deviation constructor without initialization.
   **/
  DeepBaseModelDevi();
  virtual ~DeepBaseModelDevi();

  /**
   * @brief Get the cutoff radius.
   * @return The cutoff radius.
   **/
  double cutoff() const {
    assert(inited);
    return dpbases[0]->cutoff();
  };
  /**
   * @brief Get the number of types.
   * @return The number of types.
   **/
  int numb_types() const {
    assert(inited);
    return dpbases[0]->numb_types();
  };
  /**
   * @brief Get the number of types with spin.
   * @return The number of types with spin.
   **/
  int numb_types_spin() const {
    assert(inited);
    return dpbases[0]->numb_types_spin();
  };
  /**
   * @brief Get the dimension of the frame parameter.
   * @return The dimension of the frame parameter.
   **/
  int dim_fparam() const {
    assert(inited);
    return dpbases[0]->dim_fparam();
  };
  /**
   * @brief Get the dimension of the atomic parameter.
   * @return The dimension of the atomic parameter.
   **/
  int dim_aparam() const {
    assert(inited);
    return dpbases[0]->dim_aparam();
  };
  /**
   * @brief Compute the average energy.
   * @param[out] dener The average energy.
   * @param[in] all_energy The energies of all models.
   **/
  template <typename VALUETYPE>
  void compute_avg(VALUETYPE& dener, const std::vector<VALUETYPE>& all_energy);
  /**
   * @brief Compute the average of vectors.
   * @param[out] avg The average of vectors.
   * @param[in] xx The vectors of all models.
   **/
  template <typename VALUETYPE>
  void compute_avg(std::vector<VALUETYPE>& avg,
                   const std::vector<std::vector<VALUETYPE>>& xx);
  /**
   * @brief Compute the standard deviation of vectors.
   * @param[out] std The standard deviation of vectors.
   * @param[in] avg The average of vectors.
   * @param[in] xx The vectors of all models.
   * @param[in] stride The stride to compute the deviation.
   **/
  template <typename VALUETYPE>
  void compute_std(std::vector<VALUETYPE>& std,
                   const std::vector<VALUETYPE>& avg,
                   const std::vector<std::vector<VALUETYPE>>& xx,
                   const int& stride);
  /**
   * @brief Compute the relative standard deviation of vectors.
   * @param[out] std The standard deviation of vectors.
   * @param[in] avg The average of vectors.
   * @param[in] eps The level parameter for computing the deviation.
   * @param[in] stride The stride to compute the deviation.
   **/
  template <typename VALUETYPE>
  void compute_relative_std(std::vector<VALUETYPE>& std,
                            const std::vector<VALUETYPE>& avg,
                            const VALUETYPE eps,
                            const int& stride);
  /**
   * @brief Compute the standard deviation of atomic energies.
   * @param[out] std The standard deviation of atomic energies.
   * @param[in] avg The average of atomic energies.
   * @param[in] xx The vectors of all atomic energies.
   **/
  template <typename VALUETYPE>
  void compute_std_e(std::vector<VALUETYPE>& std,
                     const std::vector<VALUETYPE>& avg,
                     const std::vector<std::vector<VALUETYPE>>& xx);
  /**
   * @brief Compute the standard deviation of forces.
   * @param[out] std The standard deviation of forces.
   * @param[in] avg The average of forces.
   * @param[in] xx The vectors of all forces.
   **/
  template <typename VALUETYPE>
  void compute_std_f(std::vector<VALUETYPE>& std,
                     const std::vector<VALUETYPE>& avg,
                     const std::vector<std::vector<VALUETYPE>>& xx);
  /**
   * @brief Compute the relative standard deviation of forces.
   * @param[out] std The relative standard deviation of forces.
   * @param[in] avg The relative average of forces.
   * @param[in] eps The level parameter for computing the deviation.
   **/
  template <typename VALUETYPE>
  void compute_relative_std_f(std::vector<VALUETYPE>& std,
                              const std::vector<VALUETYPE>& avg,
                              const VALUETYPE eps);
  /**
   * @brief Get whether the atom dimension of aparam is nall instead of fparam.
   * @param[out] aparam_nall whether the atom dimension of aparam is nall
   *instead of fparam.
   **/
  bool is_aparam_nall() const {
    assert(inited);
    return dpbases[0]->is_aparam_nall();
  };

 protected:
  unsigned numb_models;
  std::vector<std::shared_ptr<deepmd::DeepBaseModel>>
      dpbases;  // change to shared_ptr to make it inheritable
  bool inited;
};
}  // namespace deepmd
