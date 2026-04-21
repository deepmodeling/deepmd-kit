// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#ifdef BUILD_PYTORCH
#if __has_include(<torch/csrc/inductor/aoti_package/model_package_loader.h>)
#define BUILD_PT_EXPT_SPIN 1
#else
#define BUILD_PT_EXPT_SPIN 0
#endif

#if BUILD_PT_EXPT_SPIN

#include <torch/torch.h>

#include "DeepSpin.h"

namespace torch::inductor {
class AOTIModelPackageLoader;
}

namespace deepmd {
/**
 * @brief PyTorch Exportable (AOTInductor .pt2) implementation for Deep
 *Potential with spin.
 **/
class DeepSpinPTExpt : public DeepSpinBackend {
 public:
  DeepSpinPTExpt();
  virtual ~DeepSpinPTExpt();
  DeepSpinPTExpt(const std::string& model,
                 const int& gpu_rank = 0,
                 const std::string& file_content = "");
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
               const std::vector<VALUETYPE>& fparam,
               const std::vector<VALUETYPE>& aparam,
               const bool atomic);
  /**
   * @brief Evaluate without nlist (standalone — builds nlist, folds back).
   **/
  template <typename VALUETYPE, typename ENERGYVTYPE>
  void compute(ENERGYVTYPE& ener,
               std::vector<VALUETYPE>& force,
               std::vector<VALUETYPE>& force_mag,
               std::vector<VALUETYPE>& virial,
               std::vector<VALUETYPE>& atom_energy,
               std::vector<VALUETYPE>& atom_virial,
               const std::vector<VALUETYPE>& coord,
               const std::vector<VALUETYPE>& spin,
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
    return ntypes_spin;
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
  std::vector<bool> get_use_spin() const override {
    assert(inited);
    return use_spin_;
  };
  bool has_default_fparam() const {
    assert(inited);
    return has_default_fparam_;
  };

  // forward to template class
  void computew(std::vector<double>& ener,
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
                const bool atomic);
  void computew(std::vector<double>& ener,
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
                const bool atomic);
  void computew(std::vector<double>& ener,
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
                const bool atomic);
  void computew(std::vector<double>& ener,
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
                const bool atomic);

 private:
  bool inited;
  int ntypes;
  int ntypes_spin;
  int dfparam;
  int daparam;
  bool aparam_nall;
  bool has_default_fparam_;
  std::vector<double> default_fparam_;
  std::vector<bool> use_spin_;
  double rcut;
  int gpu_id;
  bool gpu_enabled;
  std::vector<std::string> type_map;
  std::vector<std::string> output_keys;
  bool do_atomic_virial;  // whether model was exported with atomic virial corr
  int nnei;               // expected nlist nnei dimension (= sum(sel))
  NeighborListData nlist_data;
  at::Tensor mapping_tensor;     // cached mapping tensor (LAMMPS path)
  at::Tensor firstneigh_tensor;  // cached nlist tensor (LAMMPS path)
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader;

  std::vector<torch::Tensor> run_model(const torch::Tensor& coord,
                                       const torch::Tensor& atype,
                                       const torch::Tensor& spin,
                                       const torch::Tensor& nlist,
                                       const torch::Tensor& mapping,
                                       const torch::Tensor& fparam,
                                       const torch::Tensor& aparam);

  void extract_outputs(std::map<std::string, torch::Tensor>& output_map,
                       const std::vector<torch::Tensor>& flat_outputs);

  void translate_error(std::function<void()> f);
};

}  // namespace deepmd

#endif  // BUILD_PT_EXPT_SPIN
#endif  // BUILD_PYTORCH
