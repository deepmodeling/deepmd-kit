// SPDX-License-Identifier: LGPL-3.0-or-later
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_map>

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "output.h"
#include "update.h"
#include "utils.h"
#if LAMMPS_VERSION_NUMBER >= 20210831
// in lammps #2902, fix_ttm members turns from private to protected
#define USE_TTM 1
#include "fix_ttm_dp.h"
#endif

#include "deepmd_version.h"
#include "pair_deepmd.h"

using namespace LAMMPS_NS;
using namespace std;

static const char cite_user_deepmd_package[] =
    "USER-DEEPMD package:\n\n"
    "@article{Wang_ComputPhysCommun_2018_v228_p178,\n"
    "  author = {Wang, Han and Zhang, Linfeng and Han, Jiequn and E, Weinan},\n"
    "  doi = {10.1016/j.cpc.2018.03.016},\n"
    "  url = {https://doi.org/10.1016/j.cpc.2018.03.016},\n"
    "  year = 2018,\n"
    "  month = {jul},\n"
    "  publisher = {Elsevier {BV}},\n"
    "  volume = 228,\n"
    "  journal = {Comput. Phys. Commun.},\n"
    "  title = {{DeePMD-kit: A deep learning package for many-body potential "
    "energy representation and molecular dynamics}},\n"
    "  pages = {178--184}\n"
    "}\n"
    "@article{Zeng_JChemPhys_2023_v159_p054801,\n"
    "  title  = {{DeePMD-kit v2: A software package for deep potential "
    "models}},\n"
    "  author =   {Jinzhe Zeng and Duo Zhang and Denghui Lu and Pinghui Mo and "
    "Zeyu Li\n"
    "         and Yixiao Chen and Mari{\\'a}n Rynik and Li'ang Huang and Ziyao "
    "Li and \n"
    "         Shaochen Shi and Yingze Wang and Haotian Ye and Ping Tuo and "
    "Jiabin\n"
    "         Yang and Ye Ding and Yifan Li and Davide Tisi and Qiyu Zeng and "
    "Han \n"
    "         Bao and Yu Xia and Jiameng Huang and Koki Muraoka and Yibo Wang "
    "and \n"
    "         Junhan Chang and Fengbo Yuan and Sigbj{\\o}rn L{\\o}land Bore "
    "and "
    "Chun\n"
    "         Cai and Yinnian Lin and Bo Wang and Jiayan Xu and Jia-Xin Zhu "
    "and \n"
    "         Chenxing Luo and Yuzhi Zhang and Rhys E A Goodall and Wenshuo "
    "Liang\n"
    "         and Anurag Kumar Singh and Sikai Yao and Jingchao Zhang and "
    "Renata\n"
    "         Wentzcovitch and Jiequn Han and Jie Liu and Weile Jia and Darrin "
    "M\n"
    "         York and Weinan E and Roberto Car and Linfeng Zhang and Han "
    "Wang},\n"
    "  journal =  {J. Chem. Phys.},\n"
    "  volume =   159,\n"
    "  issue =    5,  \n"
    "  year =    2023,\n"
    "  pages  =   054801,\n"
    "  doi =      {10.1063/5.0155600},\n"
    "}\n"
    "@Article{Zeng_JChemTheoryComput_2025_v21_p4375,\n"
    "  author =   {Jinzhe Zeng and Duo Zhang and Anyang Peng and Xiangyu "
    "Zhang and Sensen\n"
    "             He and Yan Wang and Xinzijian Liu and Hangrui Bi and Yifan "
    "Li and Chun\n"
    "             Cai and Chengqian Zhang and Yiming Du and Jia-Xin Zhu and "
    "Pinghui Mo\n"
    "             and Zhengtao Huang and Qiyu Zeng and Shaochen Shi and "
    "Xuejian Qin and\n"
    "             Zhaoxi Yu and Chenxing Luo and Ye Ding and Yun-Pei Liu and "
    "Ruosong Shi\n"
    "             and Zhenyu Wang and Sigbj{\\o}rn L{\\o}land Bore and Junhan "
    "Chang and\n"
    "             Zhe Deng and Zhaohan Ding and Siyuan Han and Wanrun Jiang "
    "and Guolin\n"
    "             Ke and Zhaoqing Liu and Denghui Lu and Koki Muraoka and "
    "Hananeh Oliaei\n"
    "             and Anurag Kumar Singh and Haohui Que and Weihong Xu and "
    "Zhangmancang\n"
    "             Xu and Yong-Bin Zhuang and Jiayu Dai and Timothy J. Giese "
    "and Weile\n"
    "             Jia and Ben Xu and Darrin M. York and Linfeng Zhang and Han "
    "Wang},\n"
    "    title =    {{DeePMD-kit v3: A Multiple-Backend Framework for Machine "
    "Learning\n"
    "             Potentials}},\n"
    "  journal =  {J. Chem. Theory Comput.},\n"
    "  year =     2025,\n"
    "  volume =   21,\n"
    "  number =   9,\n"
    "  pages =    {4375--4385},\n"
    "  doi =      {10.1021/acs.jctc.5c00340},\n"
    "}\n\n";

PairDeepMD::PairDeepMD(LAMMPS* lmp)
    : PairDeepBaseModel(
          lmp, cite_user_deepmd_package, deep_pot, deep_pot_model_devi),
      compact_selection_enabled_(false),
      compact_include_molecule_(true),
      compact_center_group_dynamic_(false),
      compact_center_group_bit_(0),
      compact_environment_cutoff_(0.0),
      compact_natoms_(0),
      commdata_(nullptr) {
  print_summary("  ");
}

PairDeepMD::~PairDeepMD() {
  // Ensure base class destructor is called
}

std::vector<tagint> PairDeepMD::allgather_unique_tagints(
    std::vector<tagint> local_values) const {
  std::sort(local_values.begin(), local_values.end());
  local_values.erase(std::unique(local_values.begin(), local_values.end()),
                     local_values.end());
  if (comm->nprocs == 1) {
    return local_values;
  }

  // LAMMPS serial MPI stubs declare send buffers as void*, so MPI send
  // scalars must remain mutable even though the collective does not alter them.
  int local_count = static_cast<int>(local_values.size());
  std::vector<int> counts(comm->nprocs, 0);
  std::vector<int> displacements(comm->nprocs, 0);
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, world);
  int total_count = 0;
  for (int rank = 0; rank < comm->nprocs; ++rank) {
    displacements[rank] = total_count;
    total_count += counts[rank];
  }
  if (total_count == 0) {
    return {};
  }

  std::vector<tagint> gathered(total_count);
  MPI_Allgatherv(local_values.data(), local_count, MPI_LMP_TAGINT,
                 gathered.data(), counts.data(), displacements.data(),
                 MPI_LMP_TAGINT, world);
  std::sort(gathered.begin(), gathered.end());
  gathered.erase(std::unique(gathered.begin(), gathered.end()), gathered.end());
  return gathered;
}

void PairDeepMD::refresh_compact_center_tags() {
  std::vector<tagint> local_center_tags;
  local_center_tags.reserve(atom->nlocal);
  for (int ii = 0; ii < atom->nlocal; ++ii) {
    if (atom->mask[ii] & compact_center_group_bit_) {
      local_center_tags.push_back(atom->tag[ii]);
    }
  }
  compact_center_tags_ = allgather_unique_tagints(std::move(local_center_tags));
  if (compact_center_tags_.empty()) {
    error->all(FLERR, "center_group for pair_style deepmd is empty");
  }
}

bool PairDeepMD::apply_compact_selection(std::vector<int>& model_types) {
  if (!compact_selection_enabled_) {
    return false;
  }

  if (compact_center_tags_.empty() || compact_center_group_dynamic_ ||
      neighbor->ago == 0) {
    refresh_compact_center_tags();
  }

  const int nlocal = atom->nlocal;
  const int nall = nlocal + atom->nghost;
  const int model_ntypes = deep_pot.numb_types();
  double** const x = atom->x;
  tagint* const tag = atom->tag;
  tagint* const molecule = atom->molecule;
  const auto is_model_atom = [&model_types, model_ntypes](int index) {
    return model_types[index] >= 0 && model_types[index] < model_ntypes;
  };

  // Atom order and the ghost set remain stable between neighbor rebuilds.
  // Dynamic groups are the exception because their membership may change on
  // any step even when the neighbor topology does not.
  if (compact_is_center_.size() != static_cast<size_t>(nall) ||
      compact_center_group_dynamic_ || neighbor->ago == 0) {
    compact_is_center_.resize(nall);
    for (int ii = 0; ii < nall; ++ii) {
      compact_is_center_[ii] = std::binary_search(
          compact_center_tags_.begin(), compact_center_tags_.end(), tag[ii]);
    }
  }

  int invalid_center_local = 0;
  for (int ii = 0; ii < nlocal; ++ii) {
    if (compact_is_center_[ii] && !is_model_atom(ii)) {
      invalid_center_local = 1;
    }
  }
  int invalid_center = 0;
  MPI_Allreduce(&invalid_center_local, &invalid_center, 1, MPI_INT, MPI_MAX,
                world);
  if (invalid_center) {
    error->all(FLERR,
               "center_group for pair_style deepmd contains an atom whose "
               "type is not represented by the DeepMD model");
  }

  if (!list) {
    error->all(FLERR,
               "compact pair_style deepmd requires an available pair "
               "neighbor list");
  }

  // The DeepMD full neighbor list already contains every ordinary atom pair
  // within the environment cutoff (plus skin), including the correct ghost
  // image in triclinic cells.  Walking only center rows avoids an O(Ncenter*N)
  // all-atom scan on every step.  Pairs removed by special_bonds are handled
  // separately below so compact selection remains independent of force-field
  // exclusions.
  const double environment_cutsq =
      compact_environment_cutoff_ * compact_environment_cutoff_;
  std::vector<tagint> local_selection_keys;
  int invalid_molecule_local = 0;
  const auto select_environment_atom = [&](int center, int environment,
                                           bool apply_minimum_image) {
    if (compact_is_center_[environment] || !is_model_atom(environment)) {
      return;
    }
    double dx = x[environment][0] - x[center][0];
    double dy = x[environment][1] - x[center][1];
    double dz = x[environment][2] - x[center][2];
    if (apply_minimum_image) {
      domain->minimum_image(FLERR, dx, dy, dz);
    }
    if (dx * dx + dy * dy + dz * dz >= environment_cutsq) {
      return;
    }
    if (compact_include_molecule_) {
      if (molecule[environment] <= 0) {
        invalid_molecule_local = 1;
        return;
      }
      local_selection_keys.push_back(molecule[environment]);
    } else {
      local_selection_keys.push_back(tag[environment]);
    }
  };

  for (int ii = 0; ii < nlocal; ++ii) {
    if (!compact_is_center_[ii]) {
      continue;
    }
    const int jnum = list->numneigh[ii];
    int* const jlist = list->firstneigh[ii];
    for (int jj = 0; jj < jnum; ++jj) {
      select_environment_atom(ii, jlist[jj] & NEIGHMASK, false);
    }
  }

  // Both zero-valued special_bonds factors remove the corresponding pair
  // from the neighbor list.  Recover only those few pairs by tag.  Building
  // the tag lookup is deferred until a non-center excluded partner is found;
  // the usual DPRc case has an entire bonded QM molecule in center_group and
  // therefore pays no hash-table cost.
  if (atom->molecular != Atom::ATOMIC && atom->special && atom->nspecial) {
    std::unordered_map<tagint, int> tag_to_index;
    const auto find_atom_by_tag = [&](tagint atom_tag) {
      if (tag_to_index.empty()) {
        tag_to_index.reserve(nall);
        for (int jj = 0; jj < nall; ++jj) {
          tag_to_index.emplace(tag[jj], jj);
        }
      }
      const auto found = tag_to_index.find(atom_tag);
      return found == tag_to_index.end() ? -1 : found->second;
    };

    for (int ii = 0; ii < nlocal; ++ii) {
      if (!compact_is_center_[ii]) {
        continue;
      }
      for (int level = 1; level <= 3; ++level) {
        if (force->special_lj[level] != 0.0 ||
            force->special_coul[level] != 0.0) {
          continue;
        }
        const int begin = level == 1 ? 0 : atom->nspecial[ii][level - 2];
        const int end = atom->nspecial[ii][level - 1];
        for (int jj = begin; jj < end; ++jj) {
          const tagint special_tag = atom->special[ii][jj];
          if (std::binary_search(compact_center_tags_.begin(),
                                 compact_center_tags_.end(), special_tag)) {
            continue;
          }
          const int special_index = find_atom_by_tag(special_tag);
          if (special_index >= 0) {
            select_environment_atom(ii, special_index, true);
          }
        }
      }
    }
  }
  int invalid_molecule = 0;
  MPI_Allreduce(&invalid_molecule_local, &invalid_molecule, 1, MPI_INT, MPI_MAX,
                world);
  if (invalid_molecule) {
    error->all(FLERR,
               "include_molecule yes requires positive molecule IDs for all "
               "environment atoms selected by pair_style deepmd");
  }

  const std::vector<tagint> selected_keys =
      allgather_unique_tagints(std::move(local_selection_keys));
  std::vector<unsigned char> selected(nall, 0);
  int selected_nlocal = 0;
  for (int ii = 0; ii < nall; ++ii) {
    const tagint key = compact_include_molecule_ ? molecule[ii] : tag[ii];
    const bool selected_environment =
        std::binary_search(selected_keys.begin(), selected_keys.end(), key);
    const bool active =
        is_model_atom(ii) && (compact_is_center_[ii] || selected_environment);
    selected[ii] = active;
    if (!active) {
      model_types[ii] = -1;
    } else if (ii < nlocal) {
      ++selected_nlocal;
    }
  }

  // Keep MPI send scalars mutable for compatibility with LAMMPS STUBS/mpi.h.
  bigint selected_local = selected_nlocal;
  MPI_Allreduce(&selected_local, &compact_natoms_, 1, MPI_LMP_BIGINT, MPI_SUM,
                world);
  if (compact_natoms_ == 0) {
    error->all(FLERR, "compact pair_style deepmd selected no model atoms");
  }

  int local_changed = compact_selected_ != selected;
  compact_selected_ = std::move(selected);
  int global_changed = 0;
  MPI_Allreduce(&local_changed, &global_changed, 1, MPI_INT, MPI_MAX, world);
  return global_changed != 0;
}

void PairDeepMD::analyze_model_deviation(double& max,
                                         double& min,
                                         double& sum,
                                         const std::vector<double>& deviation,
                                         int nlocal) const {
  if (!compact_selection_enabled_) {
    ana_st(max, min, sum, deviation, nlocal);
    return;
  }

  // If this rank owns no selected atom, preserve the caller's reduction-neutral
  // seeds (min = max double, max = 0, sum = 0) for the following MPI_Reduce.
  bool found = false;
  for (int ii = 0; ii < nlocal; ++ii) {
    if (!compact_selected_[ii]) {
      continue;
    }
    const double value = deviation[ii];
    if (!found) {
      max = min = sum = value;
      found = true;
    } else {
      max = std::max(max, value);
      min = std::min(min, value);
      sum += value;
    }
  }
}

void PairDeepMD::init_style() {
  PairDeepBaseModel::init_style();
  if (!compact_selection_enabled_) {
    return;
  }
  if (!atom->tag_enable) {
    error->all(FLERR,
               "compact pair_style deepmd requires atom IDs to be enabled");
  }
  const int center_group_index = group->find(compact_center_group_id_.c_str());
  if (center_group_index < 0) {
    error->all(FLERR, "center_group " + compact_center_group_id_ +
                          " for pair_style deepmd does not exist");
  }
  compact_center_group_bit_ = group->bitmask[center_group_index];
  compact_center_group_dynamic_ = group->dynamic[center_group_index] != 0;
  if (compact_include_molecule_ && !atom->molecule_flag) {
    error->all(FLERR,
               "include_molecule yes requires an atom style with molecule "
               "IDs");
  }
  refresh_compact_center_tags();
}

double PairDeepMD::init_one(int i, int j) {
  const double model_neighbor_cutoff = PairDeepBaseModel::init_one(i, j);
  if (!compact_selection_enabled_) {
    return model_neighbor_cutoff;
  }
  return std::max(model_neighbor_cutoff, compact_environment_cutoff_);
}

double PairDeepMD::eval_energy_with_fparam(
    const std::vector<double>& fparam_override) {
  if (numb_models != 1) {
    error->all(FLERR,
               "deepmd/fparam/dedn currently supports single-model pair_style "
               "only");
  }
  if (atom->sp_flag) {
    error->all(FLERR,
               "Pair style 'deepmd' does not support spin atoms, please use "
               "pair style 'deepspin' instead.");
  }

  bool do_ghost = true;
  commdata_ = (CommBrickDeepMD*)comm;
  double** x = atom->x;
  int* type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = 0;
  if (do_ghost) {
    nghost = atom->nghost;
  }
  int nall = nlocal + nghost;

  std::vector<int> dtype(nall);
  for (int ii = 0; ii < nall; ++ii) {
    dtype[ii] = type_idx_map[type[ii] - 1];
  }
  const bool compact_selection_changed = apply_compact_selection(dtype);

  double dener(0);
  std::vector<double> dforce(nall * 3);
  std::vector<double> dvirial(9, 0);
  std::vector<double> dcoord(nall * 3, 0.);
  std::vector<double> dbox(9, 0);
  std::vector<double> daparam;

  if (fparam_override.size() != static_cast<size_t>(dim_fparam)) {
    error->all(FLERR, "fparam override has the wrong dimension");
  }

  // get box
  dbox[0] = domain->h[0] / dist_unit_cvt_factor;  // xx
  dbox[4] = domain->h[1] / dist_unit_cvt_factor;  // yy
  dbox[8] = domain->h[2] / dist_unit_cvt_factor;  // zz
  dbox[7] = domain->h[3] / dist_unit_cvt_factor;  // zy
  dbox[6] = domain->h[4] / dist_unit_cvt_factor;  // zx
  dbox[3] = domain->h[5] / dist_unit_cvt_factor;  // yx

  // get coord
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dcoord[ii * 3 + dd] =
          (x[ii][dd] - domain->boxlo[dd]) / dist_unit_cvt_factor;
    }
  }

  // Owner mapping for message-passing .pt2 models that gather ghost features
  // through the LAMMPS atom map; unused by other models.
  std::vector<int> mapping_vec(nall, -1);
  if (comm->nprocs == 1 && atom->map_style != Atom::MAP_NONE) {
    for (size_t ii = 0; ii < nall; ++ii) {
      mapping_vec[ii] = atom->map(atom->tag[ii]);
    }
  }

  if (do_compute_aparam) {
    make_aparam_from_compute(daparam);
  } else if (aparam.size() > 0) {
    make_uniform_aparam(daparam, aparam, nlocal);
  } else if (do_ttm) {
#ifdef USE_TTM
    if (dim_aparam > 0) {
      make_ttm_aparam(daparam);
    }
#endif
  }
  int ago = neighbor->ago;
  if (compact_selection_changed) {
    ago = 0;
  }

  if (do_ghost) {
    if (!list) {
      error->all(FLERR,
                 "deepmd/fparam/dedn requires an available pair neighbor list");
    }
    deepmd_compat::InputNlist lmp_list(
        list->inum, list->ilist, list->numneigh, list->firstneigh,
        commdata_->nswap, commdata_->sendnum, commdata_->recvnum,
        commdata_->firstrecv, commdata_->sendlist, commdata_->sendproc,
        commdata_->recvproc, &world, comm->nprocs);
    lmp_list.set_mask(NEIGHMASK);
    if (comm->nprocs == 1 && atom->map_style != Atom::MAP_NONE) {
      lmp_list.set_mapping(mapping_vec.data());
    }

    try {
      deep_pot.compute(dener, dforce, dvirial, dcoord, dtype, dbox, nghost,
                       lmp_list, ago, fparam_override, daparam);
    } catch (deepmd_compat::deepmd_exception& e) {
      error->one(FLERR, e.what());
    }
  } else {
    error->all(FLERR, "unknown computational branch");
  }

  return scale[1][1] * dener * ener_unit_cvt_factor;
}

deepmd_compat::InputNlist PairDeepMD::make_comm_nlist() {
  commdata_ = (CommBrickDeepMD*)comm;
  return deepmd_compat::InputNlist(
      0, nullptr, nullptr, nullptr, commdata_->nswap, commdata_->sendnum,
      commdata_->recvnum, commdata_->firstrecv, commdata_->sendlist,
      commdata_->sendproc, commdata_->recvproc, &world, comm->nprocs);
}

void PairDeepMD::compute(int eflag, int vflag) {
  if (numb_models == 0) {
    return;
  }
  // See
  // https://docs.lammps.org/Developer_updating.html#use-ev-init-to-initialize-variables-derived-from-eflag-and-vflag
  ev_init(eflag, vflag);
  if (vflag_atom) {
    error->all(FLERR,
               "6-element atomic virial is not supported. Use compute "
               "centroid/stress/atom command for 9-element atomic virial.");
  }
  bool do_ghost = true;
  // Ghost communicator used to assemble the send/recv swap metadata.
  commdata_ = (CommBrickDeepMD*)comm;
  double** x = atom->x;
  double** f = atom->f;
  int* type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = 0;
  if (do_ghost) {
    nghost = atom->nghost;
  }
  int nall = nlocal + nghost;
  int newton_pair = force->newton_pair;

  if (atom->sp_flag) {
    error->all(
        FLERR,
        "Pair style 'deepmd' does not support spin atoms, please use pair "
        "style 'deepspin' instead.");
  }

  vector<int> dtype(nall);
  for (int ii = 0; ii < nall; ++ii) {
    dtype[ii] = type_idx_map[type[ii] - 1];
  }
  const bool compact_selection_changed = apply_compact_selection(dtype);

  double dener(0);
  vector<double> dforce(nall * 3);
  vector<double> dvirial(9, 0);
  vector<double> dcoord(nall * 3, 0.);
  vector<double> dbox(9, 0);
  vector<double> daparam;

  // get box
  dbox[0] = domain->h[0] / dist_unit_cvt_factor;  // xx
  dbox[4] = domain->h[1] / dist_unit_cvt_factor;  // yy
  dbox[8] = domain->h[2] / dist_unit_cvt_factor;  // zz
  dbox[7] = domain->h[3] / dist_unit_cvt_factor;  // zy
  dbox[6] = domain->h[4] / dist_unit_cvt_factor;  // zx
  dbox[3] = domain->h[5] / dist_unit_cvt_factor;  // yx

  // get coord
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dcoord[ii * 3 + dd] =
          (x[ii][dd] - domain->boxlo[dd]) / dist_unit_cvt_factor;
    }
  }

  // Owner mapping for message-passing .pt2 models that gather ghost features
  // through the LAMMPS atom map; unused by other models.
  std::vector<int> mapping_vec(nall, -1);
  if (comm->nprocs == 1 && atom->map_style != Atom::MAP_NONE) {
    for (size_t ii = 0; ii < nall; ++ii) {
      mapping_vec[ii] = atom->map(atom->tag[ii]);
    }
  }

  if (do_compute_aparam) {
    make_aparam_from_compute(daparam);
  } else if (aparam.size() > 0) {
    // uniform aparam
    make_uniform_aparam(daparam, aparam, nlocal);
  } else if (do_ttm) {
#ifdef USE_TTM
    if (dim_aparam > 0) {
      make_ttm_aparam(daparam);
    } else if (dim_fparam > 0) {
      make_ttm_fparam(fparam);
    }
#endif
  }

  if (do_compute_fparam) {
    make_fparam_from_compute(fparam);
  } else if (do_fix_fparam) {
    make_fparam_from_fix(fparam);
  }

  int ago = neighbor->ago;
  if (compact_selection_changed) {
    ago = 0;
  }
  if (numb_models > 1) {
    if (multi_models_no_mod_devi &&
        (out_freq > 0 && update->ntimestep % out_freq == 0)) {
      ago = 0;
    } else if (multi_models_mod_devi &&
               (out_freq == 0 || update->ntimestep % out_freq != 0)) {
      ago = 0;
    }
  }
  // compute
  single_model = (numb_models == 1);
  multi_models_no_mod_devi =
      (numb_models > 1 && (out_freq == 0 || update->ntimestep % out_freq != 0));
  multi_models_mod_devi =
      (numb_models > 1 && (out_freq > 0 && update->ntimestep % out_freq == 0));
  if (do_ghost) {
    deepmd_compat::InputNlist lmp_list(
        list->inum, list->ilist, list->numneigh, list->firstneigh,
        commdata_->nswap, commdata_->sendnum, commdata_->recvnum,
        commdata_->firstrecv, commdata_->sendlist, commdata_->sendproc,
        commdata_->recvproc, &world, comm->nprocs);
    lmp_list.set_mask(NEIGHMASK);
    if (comm->nprocs == 1 && atom->map_style != Atom::MAP_NONE) {
      lmp_list.set_mapping(mapping_vec.data());
    }
    deepmd_compat::InputNlist extend_lmp_list;
    if (single_model || multi_models_no_mod_devi) {
      // cvflag_atom is the right flag for the cvatom matrix
      if (!(eflag_atom || cvflag_atom)) {
        try {
          deep_pot.compute(dener, dforce, dvirial, dcoord, dtype, dbox, nghost,
                           lmp_list, ago, fparam, daparam, charge_spin);
        } catch (deepmd_compat::deepmd_exception& e) {
          error->one(FLERR, e.what());
        }
      }
      // do atomic energy and virial
      else {
        vector<double> deatom(nall * 1, 0);
        vector<double> dvatom(nall * 9, 0);
        try {
          deep_pot.compute(dener, dforce, dvirial, deatom, dvatom, dcoord,
                           dtype, dbox, nghost, lmp_list, ago, fparam, daparam,
                           charge_spin);
        } catch (deepmd_compat::deepmd_exception& e) {
          error->one(FLERR, e.what());
        }
        if (eflag_atom) {
          for (int ii = 0; ii < nlocal; ++ii) {
            eatom[ii] += scale[1][1] * deatom[ii] * ener_unit_cvt_factor;
          }
        }
        // Map the 9-component DeePMD atomic virial onto the LAMMPS centroid
        // per-atom virial (xx, yy, zz, xy, xz, yz, yx, zx, zy).
        if (cvflag_atom) {
          for (int ii = 0; ii < nall; ++ii) {
            cvatom[ii][0] +=
                scale[1][1] * dvatom[9 * ii + 0] * ener_unit_cvt_factor;  // xx
            cvatom[ii][1] +=
                scale[1][1] * dvatom[9 * ii + 4] * ener_unit_cvt_factor;  // yy
            cvatom[ii][2] +=
                scale[1][1] * dvatom[9 * ii + 8] * ener_unit_cvt_factor;  // zz
            cvatom[ii][3] +=
                scale[1][1] * dvatom[9 * ii + 3] * ener_unit_cvt_factor;  // xy
            cvatom[ii][4] +=
                scale[1][1] * dvatom[9 * ii + 6] * ener_unit_cvt_factor;  // xz
            cvatom[ii][5] +=
                scale[1][1] * dvatom[9 * ii + 7] * ener_unit_cvt_factor;  // yz
            cvatom[ii][6] +=
                scale[1][1] * dvatom[9 * ii + 1] * ener_unit_cvt_factor;  // yx
            cvatom[ii][7] +=
                scale[1][1] * dvatom[9 * ii + 2] * ener_unit_cvt_factor;  // zx
            cvatom[ii][8] +=
                scale[1][1] * dvatom[9 * ii + 5] * ener_unit_cvt_factor;  // zy
          }
        }
      }
    } else if (multi_models_mod_devi) {
      vector<double> deatom(nall * 1, 0);
      vector<double> dvatom(nall * 9, 0);
      vector<vector<double>> all_virial;
      vector<double> all_energy;
      vector<vector<double>> all_atom_energy;
      vector<vector<double>> all_atom_virial;
      if (!(eflag_atom || cvflag_atom)) {
        try {
          deep_pot_model_devi.compute(all_energy, all_force, all_virial, dcoord,
                                      dtype, dbox, nghost, lmp_list, ago,
                                      fparam, daparam, charge_spin);
        } catch (deepmd_compat::deepmd_exception& e) {
          error->one(FLERR, e.what());
        }
      } else {
        try {
          deep_pot_model_devi.compute(all_energy, all_force, all_virial,
                                      all_atom_energy, all_atom_virial, dcoord,
                                      dtype, dbox, nghost, lmp_list, ago,
                                      fparam, daparam, charge_spin);
        } catch (deepmd_compat::deepmd_exception& e) {
          error->one(FLERR, e.what());
        }
      }
      // deep_pot_model_devi.compute_avg (dener, all_energy);
      // deep_pot_model_devi.compute_avg (dforce, all_force);
      // deep_pot_model_devi.compute_avg (dvirial, all_virial);
      // deep_pot_model_devi.compute_avg (deatom, all_atom_energy);
      // deep_pot_model_devi.compute_avg (dvatom, all_atom_virial);
      dener = all_energy[0];
      dforce = all_force[0];
      dvirial = all_virial[0];
      if (eflag_atom) {
        deatom = all_atom_energy[0];
        for (int ii = 0; ii < nlocal; ++ii) {
          eatom[ii] += scale[1][1] * deatom[ii] * ener_unit_cvt_factor;
        }
      }
      // Map the 9-component DeePMD atomic virial onto the LAMMPS centroid
      // per-atom virial (xx, yy, zz, xy, xz, yz, yx, zx, zy).
      if (cvflag_atom) {
        dvatom = all_atom_virial[0];
        for (int ii = 0; ii < nall; ++ii) {
          cvatom[ii][0] +=
              scale[1][1] * dvatom[9 * ii + 0] * ener_unit_cvt_factor;  // xx
          cvatom[ii][1] +=
              scale[1][1] * dvatom[9 * ii + 4] * ener_unit_cvt_factor;  // yy
          cvatom[ii][2] +=
              scale[1][1] * dvatom[9 * ii + 8] * ener_unit_cvt_factor;  // zz
          cvatom[ii][3] +=
              scale[1][1] * dvatom[9 * ii + 3] * ener_unit_cvt_factor;  // xy
          cvatom[ii][4] +=
              scale[1][1] * dvatom[9 * ii + 6] * ener_unit_cvt_factor;  // xz
          cvatom[ii][5] +=
              scale[1][1] * dvatom[9 * ii + 7] * ener_unit_cvt_factor;  // yz
          cvatom[ii][6] +=
              scale[1][1] * dvatom[9 * ii + 1] * ener_unit_cvt_factor;  // yx
          cvatom[ii][7] +=
              scale[1][1] * dvatom[9 * ii + 2] * ener_unit_cvt_factor;  // zx
          cvatom[ii][8] +=
              scale[1][1] * dvatom[9 * ii + 5] * ener_unit_cvt_factor;  // zy
        }
      }
      if (out_freq > 0 && update->ntimestep % out_freq == 0) {
        int rank = comm->me;
        // std force
        if (newton_pair) {
#if LAMMPS_VERSION_NUMBER >= 20220324
          comm->reverse_comm(this);
#else
          comm->reverse_comm_pair(this);
#endif
        }
        vector<double> std_f;
        vector<double> tmp_avg_f;
        deep_pot_model_devi.compute_avg(tmp_avg_f, all_force);
        deep_pot_model_devi.compute_std_f(std_f, tmp_avg_f, all_force);
        if (out_rel == 1) {
          deep_pot_model_devi.compute_relative_std_f(std_f, tmp_avg_f, eps);
        }
        double min = numeric_limits<double>::max(), max = 0, avg = 0;
        analyze_model_deviation(max, min, avg, std_f, nlocal);
        double all_f_min = 0, all_f_max = 0, all_f_avg = 0;
        MPI_Reduce(&min, &all_f_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
        MPI_Reduce(&max, &all_f_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
        MPI_Reduce(&avg, &all_f_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        const double deviation_natoms =
            compact_selection_enabled_ ? static_cast<double>(compact_natoms_)
                                       : static_cast<double>(atom->natoms);
        all_f_avg /= deviation_natoms;
        // std v
        std::vector<double> send_v(9 * numb_models);
        std::vector<double> recv_v(9 * numb_models);
        for (int kk = 0; kk < numb_models; ++kk) {
          for (int ii = 0; ii < 9; ++ii) {
            send_v[kk * 9 + ii] = all_virial[kk][ii] / deviation_natoms;
          }
        }
        MPI_Reduce(&send_v[0], &recv_v[0], 9 * numb_models, MPI_DOUBLE, MPI_SUM,
                   0, world);
        std::vector<std::vector<double>> all_virial_1(numb_models);
        std::vector<double> avg_virial, std_virial;
        for (int kk = 0; kk < numb_models; ++kk) {
          all_virial_1[kk].resize(9);
          for (int ii = 0; ii < 9; ++ii) {
            all_virial_1[kk][ii] = recv_v[kk * 9 + ii];
          }
        }
        double all_v_min = numeric_limits<double>::max(), all_v_max = 0,
               all_v_avg = 0;
        if (rank == 0) {
          deep_pot_model_devi.compute_avg(avg_virial, all_virial_1);
          deep_pot_model_devi.compute_std(std_virial, avg_virial, all_virial_1,
                                          1);
          if (out_rel_v == 1) {
            deep_pot_model_devi.compute_relative_std(std_virial, avg_virial,
                                                     eps_v, 1);
          }
          for (int ii = 0; ii < 9; ++ii) {
            if (std_virial[ii] > all_v_max) {
              all_v_max = std_virial[ii];
            }
            if (std_virial[ii] < all_v_min) {
              all_v_min = std_virial[ii];
            }
            all_v_avg += std_virial[ii] * std_virial[ii];
          }
          all_v_avg = sqrt(all_v_avg / 9);
        }
        if (rank == 0) {
          all_v_max *= ener_unit_cvt_factor;
          all_v_min *= ener_unit_cvt_factor;
          all_v_avg *= ener_unit_cvt_factor;
          all_f_max *= force_unit_cvt_factor;
          all_f_min *= force_unit_cvt_factor;
          all_f_avg *= force_unit_cvt_factor;
          fp << setw(12) << update->ntimestep << " " << setw(18) << all_v_max
             << " " << setw(18) << all_v_min << " " << setw(18) << all_v_avg
             << " " << setw(18) << all_f_max << " " << setw(18) << all_f_min
             << " " << setw(18) << all_f_avg;
        }
        if (out_each == 1) {
          vector<double> std_f_all(atom->natoms);
          // Gather std_f and tags
          tagint* tag = atom->tag;
          int nprocs = comm->nprocs;
          ensure_model_deviation_buffers();
          for (int ii = 0; ii < nlocal; ii++) {
            tagsend[ii] = tag[ii];
            stdfsend[ii] = std_f[ii];
          }
          MPI_Gather(&nlocal, 1, MPI_INT, counts, 1, MPI_INT, 0, world);
          displacements[0] = 0;
          for (int ii = 0; ii < nprocs - 1; ii++) {
            displacements[ii + 1] = displacements[ii] + counts[ii];
          }
          MPI_Gatherv(tagsend, nlocal, MPI_LMP_TAGINT, tagrecv, counts,
                      displacements, MPI_LMP_TAGINT, 0, world);
          MPI_Gatherv(stdfsend, nlocal, MPI_DOUBLE, stdfrecv, counts,
                      displacements, MPI_DOUBLE, 0, world);
          if (rank == 0) {
            for (int dd = 0; dd < atom->natoms; ++dd) {
              std_f_all[tagrecv[dd] - 1] = stdfrecv[dd] * force_unit_cvt_factor;
            }
            for (int dd = 0; dd < atom->natoms; ++dd) {
              fp << " " << setw(18) << std_f_all[dd];
            }
          }
        }
        if (rank == 0) {
          fp << endl;
        }
      }
    } else {
      error->all(FLERR, "unknown computational branch");
    }
  } else {
    if (numb_models == 1) {
      try {
        deep_pot.compute(dener, dforce, dvirial, dcoord, dtype, dbox);
      } catch (deepmd_compat::deepmd_exception& e) {
        error->one(FLERR, e.what());
      }
    } else {
      error->all(FLERR, "Serial version does not support model devi");
    }
  }

  // get force
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      f[ii][dd] += scale[1][1] * dforce[3 * ii + dd] * force_unit_cvt_factor;
    }
  }

  // accumulate energy and virial
  if (eflag) {
    eng_vdwl += scale[1][1] * dener * ener_unit_cvt_factor;
  }
  if (vflag) {
    virial[0] += 1.0 * dvirial[0] * scale[1][1] * ener_unit_cvt_factor;
    virial[1] += 1.0 * dvirial[4] * scale[1][1] * ener_unit_cvt_factor;
    virial[2] += 1.0 * dvirial[8] * scale[1][1] * ener_unit_cvt_factor;
    virial[3] += 1.0 * dvirial[3] * scale[1][1] * ener_unit_cvt_factor;
    virial[4] += 1.0 * dvirial[6] * scale[1][1] * ener_unit_cvt_factor;
    virial[5] += 1.0 * dvirial[7] * scale[1][1] * ener_unit_cvt_factor;
  }
}

static bool is_key(const string& input) {
  vector<string> keys;
  keys.push_back("out_freq");
  keys.push_back("out_file");
  keys.push_back("fparam");
  keys.push_back("aparam");
  keys.push_back("fparam_from_compute");
  keys.push_back("fparam_from_fix");
  keys.push_back("aparam_from_compute");
  keys.push_back("charge_spin");
  keys.push_back("ttm");
  keys.push_back("atomic");
  keys.push_back("relative");
  keys.push_back("relative_v");
  keys.push_back("virtual_len");
  keys.push_back("spin_norm");
  keys.push_back("center_group");
  keys.push_back("environment_cutoff");
  keys.push_back("include_molecule");

  for (int ii = 0; ii < keys.size(); ++ii) {
    if (input == keys[ii]) {
      return true;
    }
  }
  return false;
}

void PairDeepMD::settings(int narg, char** arg) {
  if (narg <= 0) {
    error->all(FLERR, "Illegal pair_style command");
  }

  vector<string> models;
  int iarg = 0;
  while (iarg < narg) {
    if (is_key(arg[iarg])) {
      break;
    }
    iarg++;
  }
  for (int ii = 0; ii < iarg; ++ii) {
    models.push_back(arg[ii]);
  }
  numb_models = models.size();
  if (numb_models == 1) {
    try {
      deep_pot.init(arg[0], get_node_rank(), get_file_content(arg[0]));
    } catch (deepmd_compat::deepmd_exception& e) {
      error->one(FLERR, e.what());
    }
    cutoff = deep_pot.cutoff() * dist_unit_cvt_factor;
    numb_types = deep_pot.numb_types();
    numb_types_spin = deep_pot.numb_types_spin();
    dim_fparam = deep_pot.dim_fparam();
    dim_aparam = deep_pot.dim_aparam();
    dim_chg_spin = deep_pot.dim_chg_spin();
  } else {
    try {
      deep_pot.init(arg[0], get_node_rank(), get_file_content(arg[0]));
      deep_pot_model_devi.init(models, get_node_rank(),
                               get_file_content(models));
    } catch (deepmd_compat::deepmd_exception& e) {
      error->one(FLERR, e.what());
    }
    cutoff = deep_pot_model_devi.cutoff() * dist_unit_cvt_factor;
    numb_types = deep_pot_model_devi.numb_types();
    numb_types_spin = deep_pot_model_devi.numb_types_spin();
    dim_fparam = deep_pot_model_devi.dim_fparam();
    dim_aparam = deep_pot_model_devi.dim_aparam();
    dim_chg_spin = deep_pot_model_devi.dim_chg_spin();
    assert(cutoff == deep_pot.cutoff() * dist_unit_cvt_factor);
    assert(numb_types == deep_pot.numb_types());
    assert(numb_types_spin == deep_pot.numb_types_spin());
    assert(dim_fparam == deep_pot.dim_fparam());
    assert(dim_aparam == deep_pot.dim_aparam());
    assert(dim_chg_spin == deep_pot.dim_chg_spin());
  }

  out_freq = 100;
  out_file = "model_devi.out";
  out_each = 0;
  out_rel = 0;
  eps = 0.;
  fparam.clear();
  aparam.clear();
  charge_spin.clear();
  compact_selection_enabled_ = false;
  compact_include_molecule_ = true;
  compact_center_group_dynamic_ = false;
  compact_center_group_bit_ = 0;
  compact_environment_cutoff_ = 0.0;
  compact_natoms_ = 0;
  compact_center_group_id_.clear();
  compact_center_tags_.clear();
  compact_selected_.clear();
  bool center_group_set = false;
  bool environment_cutoff_set = false;
  bool include_molecule_set = false;
  while (iarg < narg) {
    if (!is_key(arg[iarg])) {
      error->all(FLERR,
                 "Illegal pair_style command\nwrong number of parameters\n");
    }
    if (string(arg[iarg]) == string("out_freq")) {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "Illegal out_freq, not provided");
      }
      out_freq = atoi(arg[iarg + 1]);
      iarg += 2;
    } else if (string(arg[iarg]) == string("out_file")) {
      if (iarg + 1 >= narg) {
        error->all(FLERR, "Illegal out_file, not provided");
      }
      out_file = string(arg[iarg + 1]);
      iarg += 2;
    } else if (string(arg[iarg]) == string("center_group")) {
      if (center_group_set) {
        error->all(FLERR, "center_group may be specified only once");
      }
      if (iarg + 1 >= narg || is_key(arg[iarg + 1])) {
        error->all(FLERR, "Illegal center_group, group ID is not provided");
      }
      compact_selection_enabled_ = true;
      compact_center_group_id_ = arg[iarg + 1];
      center_group_set = true;
      iarg += 2;
    } else if (string(arg[iarg]) == string("environment_cutoff")) {
      if (environment_cutoff_set) {
        error->all(FLERR, "environment_cutoff may be specified only once");
      }
      if (iarg + 1 >= narg || is_key(arg[iarg + 1])) {
        error->all(FLERR, "Illegal environment_cutoff, value is not provided");
      }
      compact_environment_cutoff_ =
          utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (!std::isfinite(compact_environment_cutoff_) ||
          compact_environment_cutoff_ <= 0.0) {
        error->all(FLERR,
                   "environment_cutoff must be a finite value greater than "
                   "zero");
      }
      environment_cutoff_set = true;
      iarg += 2;
    } else if (string(arg[iarg]) == string("include_molecule")) {
      if (include_molecule_set) {
        error->all(FLERR, "include_molecule may be specified only once");
      }
      if (iarg + 1 >= narg || is_key(arg[iarg + 1])) {
        error->all(FLERR, "Illegal include_molecule, yes/no is not provided");
      }
      compact_include_molecule_ =
          utils::logical(FLERR, arg[iarg + 1], false, lmp) != 0;
      include_molecule_set = true;
      iarg += 2;
    } else if (string(arg[iarg]) == string("fparam")) {
      for (int ii = 0; ii < dim_fparam; ++ii) {
        if (iarg + 1 + ii >= narg || is_key(arg[iarg + 1 + ii])) {
          char tmp[1024];
          sprintf(tmp, "Illegal fparam, the dimension should be %d",
                  dim_fparam);
          error->all(FLERR, tmp);
        }
        fparam.push_back(atof(arg[iarg + 1 + ii]));
      }
      iarg += 1 + dim_fparam;
    } else if (string(arg[iarg]) == string("aparam")) {
      for (int ii = 0; ii < dim_aparam; ++ii) {
        if (iarg + 1 + ii >= narg || is_key(arg[iarg + 1 + ii])) {
          char tmp[1024];
          sprintf(tmp, "Illegal aparam, the dimension should be %d",
                  dim_aparam);
          error->all(FLERR, tmp);
        }
        aparam.push_back(atof(arg[iarg + 1 + ii]));
      }
      iarg += 1 + dim_aparam;
    } else if (string(arg[iarg]) == string("ttm")) {
#ifdef USE_TTM
      for (int ii = 0; ii < 1; ++ii) {
        if (iarg + 1 + ii >= narg || is_key(arg[iarg + 1 + ii])) {
          error->all(FLERR, "invalid ttm key: should be ttm ttm_fix_id(str)");
        }
      }
      do_ttm = true;
      ttm_fix_id = arg[iarg + 1];
      iarg += 1 + 1;
#else
      error->all(FLERR,
                 "The deepmd-kit was compiled without support for TTM, please "
                 "rebuild it with LAMMPS version >=20210831");
#endif
    }

    ///////////////////////////////////////////////
    // pair_style     deepmd cp.pb fparam_from_compute TEMP
    // compute        TEMP all temp
    //////////////////////////////////////////////
    else if (string(arg[iarg]) == string("fparam_from_compute")) {
      for (int ii = 0; ii < 1; ++ii) {
        if (iarg + 1 + ii >= narg || is_key(arg[iarg + 1 + ii])) {
          error->all(FLERR,
                     "invalid fparam_from_compute key: should be "
                     "fparam_from_compute compute_fparam_id(str)");
        }
      }
      do_compute_fparam = true;
      compute_fparam_id = arg[iarg + 1];
      iarg += 1 + 1;
    } else if (string(arg[iarg]) == string("fparam_from_fix")) {
      if (iarg + 1 >= narg || is_key(arg[iarg + 1])) {
        error->all(FLERR,
                   "invalid fparam_from_fix key: should be "
                   "fparam_from_fix fix_fparam_id(str) [fix_vector_index]");
      }
      do_fix_fparam = true;
      fix_fparam_id = arg[iarg + 1];
      fix_fparam_index = -1;
      if (iarg + 2 < narg && !is_key(arg[iarg + 2])) {
        char* endptr = nullptr;
        errno = 0;
        long one_based = std::strtol(arg[iarg + 2], &endptr, 10);
        if (endptr == arg[iarg + 2] || *endptr != '\0' || errno == ERANGE ||
            one_based < 1 ||
            one_based > static_cast<long>(std::numeric_limits<int>::max())) {
          error->all(FLERR,
                     "invalid fparam_from_fix key: vector index must be a "
                     "positive 1-based integer");
        }
        fix_fparam_index = static_cast<int>(one_based - 1);
        iarg += 3;
      } else {
        iarg += 2;
      }
    } else if (string(arg[iarg]) == string("aparam_from_compute")) {
      for (int ii = 0; ii < 1; ++ii) {
        if (iarg + 1 + ii >= narg || is_key(arg[iarg + 1 + ii])) {
          error->all(FLERR,
                     "invalid aparam_from_compute key: should be "
                     "aparam_from_compute compute_aparam_id(str)");
        }
      }
      do_compute_aparam = true;
      compute_aparam_id = arg[iarg + 1];
      iarg += 1 + 1;
    } else if (string(arg[iarg]) == string("charge_spin")) {
      for (int ii = 0; ii < dim_chg_spin; ++ii) {
        if (iarg + 1 + ii >= narg || is_key(arg[iarg + 1 + ii])) {
          char tmp[1024];
          sprintf(tmp, "Illegal charge_spin, the dimension should be %d",
                  dim_chg_spin);
          error->all(FLERR, tmp);
        }
        charge_spin.push_back(atof(arg[iarg + 1 + ii]));
      }
      iarg += 1 + dim_chg_spin;
    } else if (string(arg[iarg]) == string("atomic")) {
      out_each = 1;
      iarg += 1;
    } else if (string(arg[iarg]) == string("relative")) {
      if (iarg + 1 >= narg || is_key(arg[iarg + 1])) {
        error->all(FLERR, "Illegal relative, not provided");
      }
      out_rel = 1;
      eps = utils::numeric(FLERR, arg[iarg + 1], false, lmp) /
            ener_unit_cvt_factor;
      iarg += 2;
    } else if (string(arg[iarg]) == string("relative_v")) {
      if (iarg + 1 >= narg || is_key(arg[iarg + 1])) {
        error->all(FLERR, "Illegal relative_v, not provided");
      }
      out_rel_v = 1;
      eps_v = utils::numeric(FLERR, arg[iarg + 1], false, lmp) /
              ener_unit_cvt_factor;
      iarg += 2;
    } else if (string(arg[iarg]) == string("virtual_len")) {
      parse_spin_vector_option(virtual_len, "virtual_len", iarg, narg, arg,
                               is_key);
    } else if (string(arg[iarg]) == string("spin_norm")) {
      parse_spin_vector_option(spin_norm, "spin_norm", iarg, narg, arg, is_key);
    }
  }

  if (out_freq < 0) {
    error->all(FLERR, "Illegal out_freq, should be >= 0");
  }
  if (compact_selection_enabled_ && !environment_cutoff_set) {
    error->all(FLERR,
               "center_group requires environment_cutoff in pair_style "
               "deepmd");
  }
  if (!compact_selection_enabled_ &&
      (environment_cutoff_set || include_molecule_set)) {
    error->all(FLERR,
               "environment_cutoff and include_molecule require center_group "
               "in pair_style deepmd");
  }
  if ((int)do_ttm + (int)do_compute_aparam + (int)(aparam.size() > 0) > 1) {
    error->all(FLERR,
               "aparam, aparam_from_compute, and ttm should NOT be set "
               "simultaneously");
  }
  if (do_compute_fparam && fparam.size() > 0) {
    error->all(
        FLERR,
        "fparam and fparam_from_compute should NOT be set simultaneously");
  }
  if (do_fix_fparam && fparam.size() > 0) {
    error->all(FLERR,
               "fparam and fparam_from_fix should NOT be set simultaneously");
  }
  if (do_fix_fparam && do_compute_fparam) {
    error->all(FLERR,
               "fparam_from_compute and fparam_from_fix should NOT be set "
               "simultaneously");
  }

  if (comm->me == 0) {
    if (numb_models > 1 && out_freq > 0) {
      if (!is_restart) {
        fp.open(out_file);
        fp << scientific;
        fp << "#" << setw(12 - 1) << "step" << setw(18 + 1) << "max_devi_v"
           << setw(18 + 1) << "min_devi_v" << setw(18 + 1) << "avg_devi_v"
           << setw(18 + 1) << "max_devi_f" << setw(18 + 1) << "min_devi_f"
           << setw(18 + 1) << "avg_devi_f";
        if (out_each) {
          // The atom count is not known when the header is written.
          fp << setw(18 + 1) << "atm_devi_f(N)";
        }
        fp << endl;
      } else {
        fp.open(out_file, std::ofstream::out | std::ofstream::app);
        fp << scientific;
      }
    }
    string pre = "  ";
    cout << pre << ">>> Info of model(s):" << endl
         << pre << "using " << setw(3) << numb_models << " model(s): ";
    if (narg == 1) {
      cout << arg[0] << " ";
    } else {
      for (int ii = 0; ii < models.size(); ++ii) {
        cout << models[ii] << " ";
      }
    }
    cout << endl
         << pre << "rcut in model:      " << cutoff << endl
         << pre << "ntypes in model:    " << numb_types << endl;
    if (compact_selection_enabled_) {
      cout << pre << "compact center group: " << compact_center_group_id_
           << endl
           << pre << "environment cutoff: " << compact_environment_cutoff_
           << endl
           << pre << "include molecules:  "
           << (compact_include_molecule_ ? "yes" : "no") << endl;
    }
    if (fparam.size() > 0) {
      cout << pre << "using fparam(s):    ";
      for (int ii = 0; ii < dim_fparam; ++ii) {
        cout << fparam[ii] << "  ";
      }
      cout << endl;
    }
    if (do_compute_fparam) {
      cout << pre << "using compute id (fparam):      ";
      cout << compute_fparam_id << "  " << endl;
    }
    if (do_fix_fparam) {
      cout << pre << "using fix id (fparam):          ";
      cout << fix_fparam_id;
      if (fix_fparam_index >= 0) {
        cout << "[" << fix_fparam_index + 1 << "]";
      }
      cout << "  " << endl;
    }
    if (do_compute_aparam) {
      cout << pre << "using compute id (aparam):      ";
      cout << compute_aparam_id << "  " << endl;
    }
    if (aparam.size() > 0) {
      cout << pre << "using aparam(s):    ";
      for (int ii = 0; ii < aparam.size(); ++ii) {
        cout << aparam[ii] << "  ";
      }
      cout << endl;
    }
    if (do_ttm) {
      cout << pre << "using ttm fix:      ";
      cout << ttm_fix_id << "  ";
      if (dim_fparam > 0) {
        cout << "(fparam)" << endl;
      } else if (dim_aparam > 0) {
        cout << "(aparam)" << endl;
      }
    }
  }

  comm_reverse = numb_models * 3;
  all_force.resize(numb_models);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDeepMD::coeff(int narg, char** arg) {
  if (!allocated) {
    allocate();
  }

  int n = atom->ntypes;
  int ilo, ihi, jlo, jhi;
  ilo = 0;
  jlo = 0;
  ihi = n;
  jhi = n;
  if (narg >= 2) {
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
    if (ilo != 1 || jlo != 1 || ihi != n || jhi != n) {
      error->all(FLERR,
                 "deepmd requires that the scale should be set to all atom "
                 "types, i.e. pair_coeff * *.");
    }
  }
  if (narg <= 2) {
    // A bare `pair_coeff * *` maps LAMMPS atom types onto the model's first
    // ntypes elements by position. When the model type_map has more entries
    // than the system has atom types (e.g. a periodic-table pretrained or
    // fine-tuned model), that positional mapping may mislabel the species, so
    // warn and recommend naming the elements explicitly.
    std::string type_map_str;
    deep_pot.get_type_map(type_map_str);
    std::istringstream iss(type_map_str);
    std::string type_name;
    int model_ntypes = 0;
    while (iss >> type_name) {
      ++model_ntypes;
    }
    if (model_ntypes > n) {
      error->warning(
          FLERR, "pair_coeff * * maps the system atom types onto the first " +
                     std::to_string(n) + " of the model's " +
                     std::to_string(model_ntypes) +
                     " element types; list the elements explicitly, e.g. "
                     "pair_coeff * * O H, to avoid a possible mislabeling.");
    }
    type_idx_map.resize(n);
    for (int ii = 0; ii < n; ++ii) {
      type_idx_map[ii] = ii;
    }
  } else {
    int iarg = 2;

    // type_map is a list of strings with undetermined length
    // note: although we have numb_types from the model, we do not require
    // the number of types in the system matches that in the model
    std::vector<std::string> type_map;
    std::string type_map_str;
    deep_pot.get_type_map(type_map_str);
    // convert the string to a vector of strings
    std::istringstream iss(type_map_str);
    std::string type_name;
    while (iss >> type_name) {
      type_map.push_back(type_name);
    }

    type_idx_map.clear();
    type_names.clear();
    while (iarg < narg) {
      std::string type_name = arg[iarg];
      type_names.push_back(type_name);
      bool found_element = false;
      for (int ii = 0; ii < type_map.size(); ++ii) {
        if (type_map[ii] == type_name) {
          type_idx_map.push_back(ii);
          found_element = true;
          break;
        }
      }
      if (!found_element && "NULL" == type_name) {
        type_idx_map.push_back(-1);  // virtual atom
        found_element = true;
      }
      if (!found_element) {
        error->all(FLERR, "pair_coeff: element " + type_name +
                              " not found in the model");
      }
      iarg += 1;
    }
    numb_types = type_idx_map.size();
    if (numb_types < n) {
      type_idx_map.resize(n);
      for (int ii = numb_types; ii < n; ++ii) {
        type_idx_map[ii] = -1;
      }
    }
  }
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
      scale[i][j] = 1.0;
      if (i > numb_types || j > numb_types) {
        char warning_msg[1024];
        sprintf(warning_msg,
                "Interaction between types %d and %d is set with deepmd, but "
                "will be ignored.\n Deepmd model has only %d types, it only "
                "computes the mulitbody interaction of types: 1-%d.",
                i, j, numb_types, numb_types);
        error->warning(FLERR, warning_msg);
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

int PairDeepMD::pack_reverse_comm(int n, int first, double* buf) {
  int i, m, last;

  m = 0;
  last = first + n;
  if (atom->sp_flag) {
    error->all(
        FLERR,
        "Pair style 'deepmd' does not support spin atoms, please use pair "
        "style 'deepspin' instead.");
  } else {
    for (i = first; i < last; i++) {
      for (int dd = 0; dd < numb_models; ++dd) {
        buf[m++] = all_force[dd][3 * i + 0];
        buf[m++] = all_force[dd][3 * i + 1];
        buf[m++] = all_force[dd][3 * i + 2];
      }
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::unpack_reverse_comm(int n, int* list, double* buf) {
  int i, j, m;

  m = 0;
  if (atom->sp_flag) {
    error->all(
        FLERR,
        "Pair style 'deepmd' does not support spin atoms, please use pair "
        "style 'deepspin' instead.");
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (int dd = 0; dd < numb_models; ++dd) {
        all_force[dd][3 * j + 0] += buf[m++];
        all_force[dd][3 * j + 1] += buf[m++];
        all_force[dd][3 * j + 2] += buf[m++];
      }
    }
  }
}
