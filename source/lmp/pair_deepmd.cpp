// SPDX-License-Identifier: LGPL-3.0-or-later
#include <string.h>

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "compute.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
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
    : PairDeepBaseModel(lmp, cite_user_deepmd_package, deep_pot),
      commdata_(nullptr) {
  print_summary("  ");
}

PairDeepMD::~PairDeepMD() {
  // Ensure base class destructor is called
}

double PairDeepMD::eval_energy_with_fparam(
    const std::vector<double>& fparam_override) {
  if (!ensemble.empty()) {
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
  // A long-range module reads the cell of a periodic system only; the empty
  // box marks a non-periodic one.
  if (long_range && domain->nonperiodic) {
    dbox.clear();
  }

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
  // A long-range module reads the cell of a periodic system only; the empty
  // box marks a non-periodic one.
  if (long_range && domain->nonperiodic) {
    dbox.clear();
  }

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
        deep_pot.compute(dener, dforce, dvirial, deatom, dvatom, dcoord, dtype,
                         dbox, nghost, lmp_list, ago, fparam, daparam,
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
    // The further models of the ensemble are evaluated on the same neighbor
    // list at the steps that write the model deviation.
    if (model_deviation.due()) {
      model_deviation.store(0, dforce, dvirial);
      const int ensemble_ago = model_deviation.ensemble_ago();
      for (std::size_t kk = 0; kk < ensemble.size(); ++kk) {
        double energy = 0.0;
        vector<double> model_force, model_virial;
        try {
          ensemble[kk].compute(energy, model_force, model_virial, dcoord, dtype,
                               dbox, nghost, lmp_list, ensemble_ago, fparam,
                               daparam, charge_spin);
        } catch (deepmd_compat::deepmd_exception& e) {
          error->one(FLERR, e.what());
        }
        model_deviation.store(kk + 1, model_force, model_virial);
      }
      if (newton_pair) {
#if LAMMPS_VERSION_NUMBER >= 20220324
        comm->reverse_comm(this);
#else
        comm->reverse_comm_pair(this);
#endif
      }
      model_deviation.write();
    }
  } else {
    if (ensemble.empty()) {
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
  if (ModelDeviation::is_option(input)) {
    return true;
  }
  vector<string> keys;
  keys.push_back("fparam");
  keys.push_back("aparam");
  keys.push_back("fparam_from_compute");
  keys.push_back("fparam_from_fix");
  keys.push_back("aparam_from_compute");
  keys.push_back("charge_spin");
  keys.push_back("ttm");
  keys.push_back("virtual_len");
  keys.push_back("spin_norm");

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
  const double ensemble_cutoff = load_ensemble(
      models, deep_pot, ensemble,
      [&](deepmd_compat::DeepPot& model, const string& path) {
        try {
          model.init(path, get_node_rank(), get_file_content(path));
        } catch (deepmd_compat::deepmd_exception& e) {
          error->one(FLERR, e.what());
        }
      },
      force->pair_style, error);
  cutoff = ensemble_cutoff * dist_unit_cvt_factor;
  numb_types = deep_pot.numb_types();
  numb_types_spin = deep_pot.numb_types_spin();
  dim_fparam = deep_pot.dim_fparam();
  dim_aparam = deep_pot.dim_aparam();
  dim_chg_spin = deep_pot.dim_chg_spin();

  model_deviation.reset();
  fparam.clear();
  aparam.clear();
  charge_spin.clear();
  while (iarg < narg) {
    if (!is_key(arg[iarg])) {
      error->all(FLERR,
                 "Illegal pair_style command\nwrong number of parameters\n");
    }
    if (const int used = model_deviation.parse(narg, arg, iarg,
                                               ener_unit_cvt_factor, is_key)) {
      iarg += used;
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
    } else if (string(arg[iarg]) == string("virtual_len")) {
      parse_spin_vector_option(virtual_len, "virtual_len", iarg, narg, arg,
                               is_key);
    } else if (string(arg[iarg]) == string("spin_norm")) {
      parse_spin_vector_option(spin_norm, "spin_norm", iarg, narg, arg, is_key);
    }
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

  // A charge/spin condition named on the pair_style line holds for the whole
  // run, so it is handed to the model once here instead of being resupplied
  // every step.  This is also what lets a compressed model serve it at all:
  // there the condition lives inside frozen tables, which are rebuilt here
  // and cannot be rebuilt per step at a sensible cost.
  if (!charge_spin.empty()) {
    try {
      deep_pot.set_charge_spin(charge_spin);
      for (auto& model : ensemble) {
        model.set_charge_spin(charge_spin);
      }
    } catch (deepmd_compat::deepmd_exception& e) {
      error->one(FLERR, e.what());
    }
  }

  model_deviation.setup(1 + ensemble.size(), false, is_restart,
                        ener_unit_cvt_factor, force_unit_cvt_factor);
  if (comm->me == 0) {
    string pre = "  ";
    cout << pre << ">>> Info of model(s):" << endl
         << pre << "using " << setw(3) << model_deviation.models()
         << " model(s): ";
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

  comm_reverse = model_deviation.width();
}

void PairDeepMD::init_style() {
  PairDeepBaseModel::init_style();
  // A long-range module sums over every pair of atoms of a non-periodic
  // system, or over the images of the cell of a system periodic in every
  // dimension; the atoms of the other ranks reach it through the
  // communicator. The model checks which of the two its module serves.
  const auto has_long_range = [](const deepmd_compat::DeepPot& model) {
    return model.has_long_range();
  };
  long_range = deep_pot.has_long_range() ||
               std::any_of(ensemble.begin(), ensemble.end(), has_long_range);
  const int periodic_dimensions =
      domain->xperiodic + domain->yperiodic + domain->zperiodic;
  if (long_range && periodic_dimensions != 0 && periodic_dimensions != 3) {
    error->all(FLERR,
               "pair style deepmd: a model with a long-range module evaluates "
               "a system periodic in every dimension or in none; use p in "
               "every dimension, or f, s or m in every dimension.");
  }
  // A non-periodic system reaches an ensemble that holds a long-range module
  // without a cell, which only models with such a module accept.
  if (long_range && periodic_dimensions == 0 &&
      !(deep_pot.has_long_range() &&
        std::all_of(ensemble.begin(), ensemble.end(), has_long_range))) {
    error->all(FLERR,
               "pair style deepmd: an ensemble that holds a model with a "
               "long-range module evaluates a non-periodic system only when "
               "every model carries one; use models with long-range modules "
               "throughout, or a periodic system.");
  }
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
  return model_deviation.pack_reverse(n, first, buf);
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::unpack_reverse_comm(int n, int* list, double* buf) {
  model_deviation.unpack_reverse(n, list, buf);
}
