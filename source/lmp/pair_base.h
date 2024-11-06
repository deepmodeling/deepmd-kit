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
#include <fstream>
#include <iostream>
#include <map>
#define FLOAT_PREC double

namespace LAMMPS_NS {
class PairDeepBaseModel : public Pair {
 public:
  PairDeepBaseModel(class LAMMPS *,
                    const char *,
                    deepmd_compat::DeepBaseModel &,
                    deepmd_compat::DeepBaseModelDevi &);
  virtual ~PairDeepBaseModel() override;
  void *extract(const char *, int &) override;
  void init_style() override;
  void write_restart(FILE *) override;
  void read_restart(FILE *) override;
  double init_one(int i, int j) override;
  void print_summary(const std::string pre) const;
  int get_node_rank();
  void cum_sum(std::map<int, int> &, std::map<int, int> &);

  std::string get_file_content(const std::string &model);
  std::vector<std::string> get_file_content(
      const std::vector<std::string> &models);
  std::vector<std::string> type_names;
  double ener_unit_cvt_factor, dist_unit_cvt_factor, force_unit_cvt_factor;

 protected:
  deepmd_compat::DeepBaseModel deep_base;
  deepmd_compat::DeepBaseModelDevi deep_base_model_devi;
  virtual void allocate();
  double **scale;
  unsigned numb_models;
  double cutoff;
  int numb_types;
  int numb_types_spin;
  std::vector<std::vector<double> > all_force;
  std::ofstream fp;
  int out_freq;
  std::string out_file;
  int dim_fparam;
  int dim_aparam;
  int out_each;
  int out_rel;
  int out_rel_v;
  int stdf_comm_buff_size;
  bool single_model;
  bool multi_models_mod_devi;
  bool multi_models_no_mod_devi;
  bool is_restart;
  std::vector<double> virtual_len;
  std::vector<double> spin_norm;
  // for spin systems, search new index of atoms by their old index
  std::map<int, int> new_idx_map;
  std::map<int, int> old_idx_map;
  std::vector<double> fparam;
  std::vector<double> aparam;
  double eps;
  double eps_v;

  void make_fparam_from_compute(std::vector<double> &fparam);
  bool do_compute_fparam;
  std::string compute_fparam_id;
  void make_aparam_from_compute(std::vector<double> &aparam);
  bool do_compute_aparam;
  std::string compute_aparam_id;

  void make_ttm_fparam(std::vector<double> &fparam);

  void make_ttm_aparam(std::vector<double> &dparam);
  bool do_ttm;
  std::string ttm_fix_id;
  int *counts, *displacements;
  tagint *tagsend, *tagrecv;
  double *stdfsend, *stdfrecv;
  std::vector<int> type_idx_map;
};

}  // namespace LAMMPS_NS

void make_uniform_aparam(std::vector<double> &daparam,
                         const std::vector<double> &aparam,
                         const int &nlocal);
void ana_st(double &max,
            double &min,
            double &sum,
            const std::vector<double> &vec,
            const int &nloc);

#endif
