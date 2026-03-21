// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PYTORCH
// AOTInductor package loader requires a header that may not exist on all
// platforms (e.g. macOS x86_64).  Disable pt_expt support when missing.
#if __has_include(<torch/csrc/inductor/aoti_package/model_package_loader.h>)
#define BUILD_PT_EXPT 1
#else
#define BUILD_PT_EXPT 0
#endif

#if BUILD_PT_EXPT

#include <torch/torch.h>

#include "DeepPot.h"

namespace torch::inductor {
class AOTIModelPackageLoader;
}

namespace deepmd {
/**
 * @brief PyTorch Exportable (AOTInductor .pt2) implementation for Deep
 *Potential.
 **/
class DeepPotPTExpt : public DeepPotBackend {
 public:
  /**
   * @brief DP constructor without initialization.
   **/
  DeepPotPTExpt();
  virtual ~DeepPotPTExpt();
  /**
   * @brief DP constructor with initialization.
   * @param[in] model The name of the .pt2 model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  DeepPotPTExpt(const std::string& model,
                const int& gpu_rank = 0,
                const std::string& file_content = "");
  /**
   * @brief Initialize the DP.
   * @param[in] model The name of the .pt2 model file.
   * @param[in] gpu_rank The GPU rank. Default is 0.
   * @param[in] file_content The content of the model file. If it is not empty,
   *DP will read from the string instead of the file.
   **/
  void init(const std::string& model,
            const int& gpu_rank = 0,
            const std::string& file_content = "");

 private:
  /**
   * @brief Evaluate with nlist (LAMMPS path — extended forces).
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
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
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);
  /**
   * @brief Evaluate without nlist (standalone — builds nlist, folds back).
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<int>& atype,
               const std::vector<VALUETYPE>& box,
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);

 public:
  double cutoff() const {
    assert(inited);
    return rcut;
  };
  int numb_types() const {
    assert(inited);
    return ntypes;
  };
  int numb_types_spin() const {
    assert(inited);
    return 0;
  };
  int dim_fparam() const {
    assert(inited);
    return dfparam;
  };
  int dim_aparam() const {
    assert(inited);
    return daparam;
  };
  void get_type_map(std::string& type_map);
  bool is_aparam_nall() const {
    assert(inited);
    return aparam_nall;
  };
  bool has_default_fparam() const {
    assert(inited);
    return has_default_fparam_;
  };

  // forward to template class
  void computew(std::vector<double>& ener,
                std::vector<double>& force,
                std::vector<double>& virial,
                std::vector<double>& atom_energy,
                std::vector<double>& atom_virial,
                const std::vector<double>& coord,
                const std::vector<int>& atype,
                const std::vector<double>& box,
                const std::vector<double>& fparam,
                const std::vector<double>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
                std::vector<float>& force,
                std::vector<float>& virial,
                std::vector<float>& atom_energy,
                std::vector<float>& atom_virial,
                const std::vector<float>& coord,
                const std::vector<int>& atype,
                const std::vector<float>& box,
                const std::vector<float>& fparam,
                const std::vector<float>& aparam,
                const bool atomic);
  void computew(std::vector<double>& ener,
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
                const bool atomic);
  void computew(std::vector<double>& ener,
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
                const bool atomic);
  void computew_mixed_type(std::vector<double>& ener,
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
                           const bool atomic);
  void computew_mixed_type(std::vector<double>& ener,
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
                           const bool atomic);

 private:
  bool inited;
  int ntypes;
  int dfparam;
  int daparam;
  bool aparam_nall;
  bool has_default_fparam_;
  double rcut;
  int gpu_id;
  bool gpu_enabled;
  std::vector<std::string> type_map;
  std::vector<std::string> output_keys;  // sorted internal output key names
  bool mixed_types;
  std::vector<int> sel;
  NeighborListData nlist_data;
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;

  /**
   * @brief Multi-frame loop for standalone compute (no nlist).
   */
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute_nframes(ENERGYVTYPE& ener,
                       std::vector<VALUETYPE>& force,
                       std::vector<VALUETYPE>& virial,
                       std::vector<VALUETYPE>& atom_energy,
                       std::vector<VALUETYPE>& atom_virial,
                       const int nframes,
                       const std::vector<VALUETYPE>& coord,
                       const std::vector<int>& atype,
                       const std::vector<VALUETYPE>& box,
                       const std::vector<VALUETYPE>& fparam,
                       const std::vector<VALUETYPE>& aparam,
                       const bool atomic);

  /**
   * @brief Mixed-type compute implementation (loops over frames).
   */
  template <typename VALUETYPE>
  void compute_mixed_type_impl(std::vector<double>& ener,
                               std::vector<VALUETYPE>& force,
                               std::vector<VALUETYPE>& virial,
                               std::vector<VALUETYPE>& atom_energy,
                               std::vector<VALUETYPE>& atom_virial,
                               const int& nframes,
                               const std::vector<VALUETYPE>& coord,
                               const std::vector<int>& atype,
                               const std::vector<VALUETYPE>& box,
                               const std::vector<VALUETYPE>& fparam,
                               const std::vector<VALUETYPE>& aparam,
                               const bool atomic);

  /**
   * @brief Run the .pt2 model and return flat output tensors.
   * @param[in] coord Extended coordinates tensor.
   * @param[in] atype Extended atom types tensor.
   * @param[in] nlist Neighbor list tensor.
   * @param[in] mapping Mapping tensor.
   * @param[in] fparam Frame parameter tensor (or empty).
   * @param[in] aparam Atomic parameter tensor (or empty).
   * @return Vector of output tensors in sorted key order.
   */
  std::vector<torch::Tensor> run_model(const torch::Tensor& coord,
                                       const torch::Tensor& atype,
                                       const torch::Tensor& nlist,
                                       const torch::Tensor& mapping,
                                       const torch::Tensor& fparam,
                                       const torch::Tensor& aparam);

  /**
   * @brief Extract outputs from flat tensor list using output_keys.
   */
  void extract_outputs(std::map<std::string, torch::Tensor>& output_map,
                       const std::vector<torch::Tensor>& flat_outputs);

  /**
   * @brief Translate PyTorch exceptions to DeePMD-kit exceptions.
   */
  void translate_error(std::function<void()> f);
};

}  // namespace deepmd

#endif  // BUILD_PT_EXPT
#endif  // BUILD_PYTORCH
