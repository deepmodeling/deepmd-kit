// SPDX-License-Identifier: LGPL-3.0-or-later
#ifndef LAMMPS_VERSION_NUMBER
#error Please define LAMMPS_VERSION_NUMBER to yyyymmdd
#endif

#ifndef LMP_PAIR_NNP_BASE_H
#define LMP_PAIR_NNP_BASE_H

#include "pair.h"
#ifdef DP_USE_CXX_API
#ifdef LMPPLUGIN
#include "DeepBaseModel.h"
#else
#include "deepmd/DeepBaseModel.h"
#endif
namespace deepmd_compat = deepmd;
#else
#ifdef LMPPLUGIN
#include "deepmd.hpp"
#else
#include "deepmd/deepmd.hpp"
#endif
namespace deepmd_compat = deepmd::hpp;
#endif
#include <iostream>
#include <map>

#include "model_deviation.h"
#define FLOAT_PREC double

namespace LAMMPS_NS {
class PairDeepBaseModel : public Pair {
 public:
  PairDeepBaseModel(class LAMMPS*, const char*, deepmd_compat::DeepBaseModel&);
  virtual ~PairDeepBaseModel() override;
  void* extract(const char*, int&) override;
  void init_style() override;
  void write_restart(FILE*) override;
  void read_restart(FILE*) override;
  double init_one(int i, int j) override;
  void print_summary(const std::string pre) const;
  int get_node_rank();
  void cum_sum(std::map<int, int>&, std::map<int, int>&);
  int get_dim_fparam() const { return dim_fparam; }

  std::string get_file_content(const std::string& model);
  std::vector<std::string> get_file_content(
      const std::vector<std::string>& models);
  std::vector<std::string> type_names;
  double ener_unit_cvt_factor, dist_unit_cvt_factor, force_unit_cvt_factor;

 protected:
  // Bound (not copied) to the derived class's own model member, the first
  // model of the ensemble. The derived members are still under construction
  // when this base constructor runs, so copying them would read uninitialized
  // state; a reference only records the address and resolves to the fully
  // constructed model by first use. Used for the model summary.
  deepmd_compat::DeepBaseModel& deep_base;
  virtual void allocate();
  double** scale;
  double cutoff;
  int numb_types;
  int numb_types_spin;
  // Options, outputs and file of the deviation of the ensemble of models.
  ModelDeviation model_deviation;
  int dim_fparam;
  int dim_aparam;
  int dim_chg_spin;
  bool is_restart;
  std::vector<double> virtual_len;
  std::vector<double> spin_norm;
  // for spin systems, search new index of atoms by their old index
  std::map<int, int> new_idx_map;
  std::map<int, int> old_idx_map;
  std::vector<double> fparam;
  std::vector<double> aparam;
  std::vector<double> charge_spin;

  void make_fparam_from_compute(std::vector<double>& fparam);
  bool do_compute_fparam;
  std::string compute_fparam_id;
  void make_fparam_from_fix(std::vector<double>& fparam);
  bool do_fix_fparam;
  std::string fix_fparam_id;
  int fix_fparam_index;
  void make_aparam_from_compute(std::vector<double>& aparam);
  bool do_compute_aparam;
  std::string compute_aparam_id;

  void parse_spin_vector_option(std::vector<double>& values,
                                const std::string& option,
                                int& iarg,
                                int narg,
                                char** arg,
                                bool (*is_key)(const std::string&));

  void make_ttm_fparam(std::vector<double>& fparam);

  void make_ttm_aparam(std::vector<double>& dparam);
  bool do_ttm;
  std::string ttm_fix_id;
  std::vector<int> type_idx_map;
};

}  // namespace LAMMPS_NS

void make_uniform_aparam(std::vector<double>& daparam,
                         const std::vector<double>& aparam,
                         const int& nlocal);

#endif
