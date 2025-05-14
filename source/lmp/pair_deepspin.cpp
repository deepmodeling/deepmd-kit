// SPDX-License-Identifier: LGPL-3.0-or-later
#include <string.h>

#include <cassert>
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
#if LAMMPS_VERSION_NUMBER >= 20210831
// in lammps #2902, fix_ttm members turns from private to protected
#define USE_TTM 1
#include "fix_ttm_dp.h"
#endif

#include "deepmd_version.h"
#include "pair_deepspin.h"

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

PairDeepSpin::PairDeepSpin(LAMMPS *lmp)
    : PairDeepBaseModel(
          lmp, cite_user_deepmd_package, deep_spin, deep_spin_model_devi) {
  // Constructor body can be empty
}

PairDeepSpin::~PairDeepSpin() {
  // Ensure base class destructor is called
}

void PairDeepSpin::compute(int eflag, int vflag) {
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
  //  dpa2 communication
  commdata_ = (CommBrickDeepSpin *)comm;
  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = 0;
  if (do_ghost) {
    nghost = atom->nghost;
  }
  int nall = nlocal + nghost;
  int newton_pair = force->newton_pair;

  vector<double> dspin(nall * 3, 0.);
  vector<double> dfm(nall * 3, 0.);
  double **sp = atom->sp;
  double **fm = atom->fm;
  // spin initialize
  if (atom->sp_flag) {
    // get spin
    for (int ii = 0; ii < nall; ++ii) {
      for (int dd = 0; dd < 3; ++dd) {
        dspin[ii * 3 + dd] = sp[ii][dd] * sp[ii][3];  // get real spin vector
      }
    }
  } else {
    error->all(
        FLERR,
        "Pair style 'deepspin' only supports spin atoms, please use pair style "
        "'deepmd' instead.");
  }

  vector<int> dtype(nall);
  for (int ii = 0; ii < nall; ++ii) {
    dtype[ii] = type_idx_map[type[ii] - 1];
  }

  double dener(0);
  vector<double> dforce(nall * 3);
  vector<double> dforce_mag(nall * 3);
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
  }

  // int ago = numb_models > 1 ? 0 : neighbor->ago;
  int ago = neighbor->ago;
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
        commdata_->recvproc, &world);
    lmp_list.set_mask(NEIGHMASK);
    if (single_model || multi_models_no_mod_devi) {
      // cvflag_atom is the right flag for the cvatom matrix
      if (!(eflag_atom || cvflag_atom)) {
        try {
          deep_spin.compute(dener, dforce, dforce_mag, dvirial, dcoord, dspin,
                            dtype, dbox, nghost, lmp_list, ago, fparam,
                            daparam);
        } catch (deepmd_compat::deepmd_exception &e) {
          error->one(FLERR, e.what());
        }
      }
      // do atomic energy and virial
      else {
        vector<double> deatom(nall * 1, 0);
        vector<double> dvatom(nall * 9, 0);
        try {
          deep_spin.compute(dener, dforce, dforce_mag, dvirial, deatom, dvatom,
                            dcoord, dspin, dtype, dbox, nghost, lmp_list, ago,
                            fparam, daparam);
        } catch (deepmd_compat::deepmd_exception &e) {
          error->one(FLERR, e.what());
        }
        if (eflag_atom) {
          for (int ii = 0; ii < nlocal; ++ii) {
            eatom[ii] += scale[1][1] * deatom[ii] * ener_unit_cvt_factor;
          }
        }
        // Added by Davide Tisi 2020
        // interface the atomic virial computed by DeepMD
        // with the one used in centroid atoms
        if (cvflag_atom) {
          for (int ii = 0; ii < nall; ++ii) {
            // vatom[ii][0] += 1.0 * dvatom[9*ii+0];
            // vatom[ii][1] += 1.0 * dvatom[9*ii+4];
            // vatom[ii][2] += 1.0 * dvatom[9*ii+8];
            // vatom[ii][3] += 1.0 * dvatom[9*ii+3];
            // vatom[ii][4] += 1.0 * dvatom[9*ii+6];
            // vatom[ii][5] += 1.0 * dvatom[9*ii+7];
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
          deep_spin_model_devi.compute(all_energy, all_force, all_force_mag,
                                       all_virial, dcoord, dspin, dtype, dbox,
                                       nghost, lmp_list, ago, fparam, daparam);
        } catch (deepmd_compat::deepmd_exception &e) {
          error->one(FLERR, e.what());
        }
      } else {
        try {
          deep_spin_model_devi.compute(
              all_energy, all_force, all_force_mag, all_virial, all_atom_energy,
              all_atom_virial, dcoord, dspin, dtype, dbox, nghost, lmp_list,
              ago, fparam, daparam);
        } catch (deepmd_compat::deepmd_exception &e) {
          error->one(FLERR, e.what());
        }
      }
      // deep_spin_model_devi.compute_avg (dener, all_energy);
      // deep_spin_model_devi.compute_avg (dforce, all_force);
      // deep_spin_model_devi.compute_avg (dvirial, all_virial);
      // deep_spin_model_devi.compute_avg (deatom, all_atom_energy);
      // deep_spin_model_devi.compute_avg (dvatom, all_atom_virial);
      dener = all_energy[0];
      dforce = all_force[0];
      dforce_mag = all_force_mag[0];
      dvirial = all_virial[0];
      if (eflag_atom) {
        deatom = all_atom_energy[0];
        for (int ii = 0; ii < nlocal; ++ii) {
          eatom[ii] += scale[1][1] * deatom[ii] * ener_unit_cvt_factor;
        }
      }
      // Added by Davide Tisi 2020
      // interface the atomic virial computed by DeepMD
      // with the one used in centroid atoms
      if (cvflag_atom) {
        dvatom = all_atom_virial[0];
        for (int ii = 0; ii < nall; ++ii) {
          // vatom[ii][0] += 1.0 * dvatom[9*ii+0];
          // vatom[ii][1] += 1.0 * dvatom[9*ii+4];
          // vatom[ii][2] += 1.0 * dvatom[9*ii+8];
          // vatom[ii][3] += 1.0 * dvatom[9*ii+3];
          // vatom[ii][4] += 1.0 * dvatom[9*ii+6];
          // vatom[ii][5] += 1.0 * dvatom[9*ii+7];
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
        vector<double> std_fm;
        vector<double> tmp_avg_fm;
        deep_spin_model_devi.compute_avg(tmp_avg_f, all_force);
        deep_spin_model_devi.compute_std_f(std_f, tmp_avg_f, all_force);
        if (out_rel == 1) {
          deep_spin_model_devi.compute_relative_std_f(std_f, tmp_avg_f, eps);
        }
        double min = numeric_limits<double>::max(), max = 0, avg = 0;
        ana_st(max, min, avg, std_f, nlocal);
        double all_f_min = 0, all_f_max = 0, all_f_avg = 0;
        double all_fm_min = 0, all_fm_max = 0, all_fm_avg = 0;
        MPI_Reduce(&min, &all_f_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
        MPI_Reduce(&max, &all_f_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
        MPI_Reduce(&avg, &all_f_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        all_f_avg /= double(atom->natoms);
        deep_spin_model_devi.compute_avg(tmp_avg_fm, all_force_mag);
        deep_spin_model_devi.compute_std_f(std_fm, tmp_avg_fm, all_force_mag);
        if (out_rel == 1) {
          deep_spin_model_devi.compute_relative_std_f(std_fm, tmp_avg_fm, eps);
        }
        min = numeric_limits<double>::max(), max = 0, avg = 0;
        ana_st(max, min, avg, std_fm, nlocal);
        MPI_Reduce(&min, &all_fm_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
        MPI_Reduce(&max, &all_fm_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
        MPI_Reduce(&avg, &all_fm_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        // need modified for only spin atoms
        all_fm_avg /= double(atom->natoms);
        // std v
        std::vector<double> send_v(9 * numb_models);
        std::vector<double> recv_v(9 * numb_models);
        for (int kk = 0; kk < numb_models; ++kk) {
          for (int ii = 0; ii < 9; ++ii) {
            send_v[kk * 9 + ii] = all_virial[kk][ii] / double(atom->natoms);
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
          deep_spin_model_devi.compute_avg(avg_virial, all_virial_1);
          deep_spin_model_devi.compute_std(std_virial, avg_virial, all_virial_1,
                                           1);
          if (out_rel_v == 1) {
            deep_spin_model_devi.compute_relative_std(std_virial, avg_virial,
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
          all_fm_max *= force_unit_cvt_factor;
          all_fm_min *= force_unit_cvt_factor;
          all_fm_avg *= force_unit_cvt_factor;
          fp << setw(12) << update->ntimestep << " " << setw(18) << all_v_max
             << " " << setw(18) << all_v_min << " " << setw(18) << all_v_avg
             << " " << setw(18) << all_f_max << " " << setw(18) << all_f_min
             << " " << setw(18) << all_f_avg << " " << setw(18) << all_fm_max
             << " " << setw(18) << all_fm_min << " " << setw(18) << all_fm_avg;
        }
        if (out_each == 1) {
          // need support for spin atomic force.
          vector<double> std_f_all(atom->natoms);
          // Gather std_f and tags
          tagint *tag = atom->tag;
          int nprocs = comm->nprocs;
          // Grow arrays if necessary
          if (atom->natoms > stdf_comm_buff_size) {
            stdf_comm_buff_size = atom->natoms;
            memory->destroy(stdfsend);
            memory->destroy(stdfrecv);
            memory->destroy(tagsend);
            memory->destroy(tagrecv);
            memory->create(stdfsend, stdf_comm_buff_size, "deepmd:stdfsendall");
            memory->create(stdfrecv, stdf_comm_buff_size, "deepmd:stdfrecvall");
            memory->create(tagsend, stdf_comm_buff_size, "deepmd:tagsendall");
            memory->create(tagrecv, stdf_comm_buff_size, "deepmd:tagrecvall");
          }
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
        deep_spin.compute(dener, dforce, dforce_mag, dvirial, dcoord, dspin,
                          dtype, dbox);
      } catch (deepmd_compat::deepmd_exception &e) {
        error->one(FLERR, e.what());
      }
    } else {
      error->all(FLERR, "Serial version does not support model devi");
    }
  }

  // get force
  // unit_factor = hbar / spin_norm;
  const double hbar = 6.5821191e-04;
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      f[ii][dd] += scale[1][1] * dforce[3 * ii + dd] * force_unit_cvt_factor;
      fm[ii][dd] += scale[1][1] * dforce_mag[3 * ii + dd] / (hbar / sp[ii][3]) *
                    force_unit_cvt_factor;
    }
  }

  std::map<int, int>().swap(new_idx_map);
  std::map<int, int>().swap(old_idx_map);
  // malloc_trim(0);

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

static bool is_key(const string &input) {
  vector<string> keys;
  keys.push_back("out_freq");
  keys.push_back("out_file");
  keys.push_back("fparam");
  keys.push_back("aparam");
  keys.push_back("fparam_from_compute");
  keys.push_back("aparam_from_compute");
  keys.push_back("ttm");
  keys.push_back("atomic");
  keys.push_back("relative");
  keys.push_back("relative_v");
  keys.push_back("virtual_len");
  keys.push_back("spin_norm");

  for (int ii = 0; ii < keys.size(); ++ii) {
    if (input == keys[ii]) {
      return true;
    }
  }
  return false;
}

void PairDeepSpin::settings(int narg, char **arg) {
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
      deep_spin.init(arg[0], get_node_rank(), get_file_content(arg[0]));
    } catch (deepmd_compat::deepmd_exception &e) {
      error->one(FLERR, e.what());
    }
    cutoff = deep_spin.cutoff() * dist_unit_cvt_factor;
    numb_types = deep_spin.numb_types();
    numb_types_spin = deep_spin.numb_types_spin();
    dim_fparam = deep_spin.dim_fparam();
    dim_aparam = deep_spin.dim_aparam();
  } else {
    try {
      deep_spin.init(arg[0], get_node_rank(), get_file_content(arg[0]));
      deep_spin_model_devi.init(models, get_node_rank(),
                                get_file_content(models));
    } catch (deepmd_compat::deepmd_exception &e) {
      error->one(FLERR, e.what());
    }
    cutoff = deep_spin_model_devi.cutoff() * dist_unit_cvt_factor;
    numb_types = deep_spin_model_devi.numb_types();
    numb_types_spin = deep_spin_model_devi.numb_types_spin();
    dim_fparam = deep_spin_model_devi.dim_fparam();
    dim_aparam = deep_spin_model_devi.dim_aparam();
    assert(cutoff == deep_spin.cutoff() * dist_unit_cvt_factor);
    assert(numb_types == deep_spin.numb_types());
    assert(numb_types_spin == deep_spin.numb_types_spin());
    assert(dim_fparam == deep_spin.dim_fparam());
    assert(dim_aparam == deep_spin.dim_aparam());
  }

  out_freq = 100;
  out_file = "model_devi.out";
  out_each = 0;
  out_rel = 0;
  eps = 0.;
  fparam.clear();
  aparam.clear();
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
    } else if (string(arg[iarg]) == string("atomic")) {
      out_each = 1;
      iarg += 1;
    } else if (string(arg[iarg]) == string("relative")) {
      out_rel = 1;
      eps = atof(arg[iarg + 1]) / ener_unit_cvt_factor;
      iarg += 2;
    } else if (string(arg[iarg]) == string("relative_v")) {
      out_rel_v = 1;
      eps_v = atof(arg[iarg + 1]) / ener_unit_cvt_factor;
      iarg += 2;
    } else if (string(arg[iarg]) == string("virtual_len")) {
      virtual_len.resize(numb_types_spin);
      for (int ii = 0; ii < numb_types_spin; ++ii) {
        virtual_len[ii] = atof(arg[iarg + ii + 1]);
      }
      iarg += numb_types_spin + 1;
    } else if (string(arg[iarg]) == string("spin_norm")) {
      spin_norm.resize(numb_types_spin);
      for (int ii = 0; ii < numb_types_spin; ++ii) {
        spin_norm[ii] = atof(arg[iarg + ii + 1]);
      }
      iarg += numb_types_spin + 1;
    }
  }

  if (out_freq < 0) {
    error->all(FLERR, "Illegal out_freq, should be >= 0");
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

  if (comm->me == 0) {
    if (numb_models > 1 && out_freq > 0) {
      if (!is_restart) {
        fp.open(out_file);
        fp << scientific;
        fp << "#" << setw(12 - 1) << "step" << setw(18 + 1) << "max_devi_v"
           << setw(18 + 1) << "min_devi_v" << setw(18 + 1) << "avg_devi_v"
           << setw(18 + 1) << "max_devi_fr" << setw(18 + 1) << "min_devi_fr"
           << setw(18 + 1) << "avg_devi_fr" << setw(18 + 1) << "max_devi_fm"
           << setw(18 + 1) << "min_devi_fm" << setw(18 + 1) << "avg_devi_fm"
           << endl;
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

  comm_reverse = numb_models * 3 * 2;
  all_force.resize(numb_models);
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDeepSpin::coeff(int narg, char **arg) {
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
    deep_spin.get_type_map(type_map_str);
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
        type_idx_map.push_back(type_map.size());  // ghost type
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

int PairDeepSpin::pack_reverse_comm(int n, int first, double *buf) {
  int i, m, last;

  m = 0;
  last = first + n;
  if (!atom->sp_flag) {
    error->all(
        FLERR,
        "Pair style 'deepspin' only supports spin atoms, please use pair style "
        "'deepmd' instead.");
  } else {
    for (i = first; i < last; i++) {
      for (int dd = 0; dd < numb_models; ++dd) {
        buf[m++] = all_force[dd][3 * i + 0];
        buf[m++] = all_force[dd][3 * i + 1];
        buf[m++] = all_force[dd][3 * i + 2];
        buf[m++] = all_force_mag[dd][3 * i + 0];
        buf[m++] = all_force_mag[dd][3 * i + 1];
        buf[m++] = all_force_mag[dd][3 * i + 2];
      }
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairDeepSpin::unpack_reverse_comm(int n, int *list, double *buf) {
  int i, j, m;

  m = 0;
  if (!atom->sp_flag) {
    error->all(
        FLERR,
        "Pair style 'deepspin' only supports spin atoms, please use pair style "
        "'deepmd' instead.");
  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      for (int dd = 0; dd < numb_models; ++dd) {
        all_force[dd][3 * j + 0] += buf[m++];
        all_force[dd][3 * j + 1] += buf[m++];
        all_force[dd][3 * j + 2] += buf[m++];
        all_force_mag[dd][3 * j + 0] += buf[m++];
        all_force_mag[dd][3 * j + 1] += buf[m++];
        all_force_mag[dd][3 * j + 2] += buf[m++];
      }
    }
  }
}
