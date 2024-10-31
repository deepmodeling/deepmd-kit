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
    "@misc{Zeng_JChemPhys_2023_v159_p054801,\n"
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
    "}\n\n";

PairDeepMD::PairDeepMD(LAMMPS *lmp)
    : PairDeepMDBase(lmp, cite_user_deepmd_package)
{
  // Constructor body can be empty
}

PairDeepMD::~PairDeepMD() {
  // Ensure base class destructor is called 
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
  //  dpa2 communication
  commdata_ = (CommBrickDeepMD *)comm;
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
  if (atom->sp_flag) {
      std::cout << "Pair style 'deepmd' does not support spin atoms, please use pair style 'deepspin' instead." << std::endl;
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
    deepmd_compat::InputNlist extend_lmp_list;
    if (single_model || multi_models_no_mod_devi) {
      // cvflag_atom is the right flag for the cvatom matrix
      if (!(eflag_atom || cvflag_atom)) {
          try {
            deep_pot.compute(dener, dforce, dvirial, dcoord, dtype, dbox,
                             nghost, lmp_list, ago, fparam, daparam);
          } catch (deepmd_compat::deepmd_exception &e) {
            error->one(FLERR, e.what());
          }
      }
      // do atomic energy and virial
      else {
        vector<double> deatom(nall * 1, 0);
        vector<double> dvatom(nall * 9, 0);
        try {
          deep_pot.compute(dener, dforce, dvirial, deatom, dvatom, dcoord,
                            dtype, dbox, nghost, lmp_list, ago, fparam,
                            daparam);
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
          deep_pot_model_devi.compute(all_energy, all_force, all_virial,
                                      dcoord, dtype, dbox, nghost, lmp_list,
                                      ago, fparam, daparam);
        } catch (deepmd_compat::deepmd_exception &e) {
          error->one(FLERR, e.what());
        }
      } else {
        try {
          deep_pot_model_devi.compute(all_energy, all_force, all_virial,
                                      all_atom_energy, all_atom_virial,
                                      dcoord, dtype, dbox, nghost, lmp_list,
                                      ago, fparam, daparam);
        } catch (deepmd_compat::deepmd_exception &e) {
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
        deep_pot_model_devi.compute_avg(tmp_avg_f, all_force);
        deep_pot_model_devi.compute_std_f(std_f, tmp_avg_f, all_force);
        if (out_rel == 1) {
          deep_pot_model_devi.compute_relative_std_f(std_f, tmp_avg_f, eps);
        }
        double min = numeric_limits<double>::max(), max = 0, avg = 0;
        ana_st(max, min, avg, std_f, nlocal);
        double all_f_min = 0, all_f_max = 0, all_f_avg = 0;
        double all_fm_min = 0, all_fm_max = 0, all_fm_avg = 0;
        MPI_Reduce(&min, &all_f_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
        MPI_Reduce(&max, &all_f_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
        MPI_Reduce(&avg, &all_f_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
        all_f_avg /= double(atom->natoms);
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
        deep_pot.compute(dener, dforce, dvirial, dcoord, dtype, dbox);
      } catch (deepmd_compat::deepmd_exception &e) {
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

/* ---------------------------------------------------------------------- */

int PairDeepMD::pack_reverse_comm(int n, int first, double *buf) {
  int i, m, last;

  m = 0;
  last = first + n;
  if (atom->sp_flag) {
    std::cout << "Pair style 'deepmd' does not support spin atoms, please use pair style 'deepspin' instead." << std::endl;
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

void PairDeepMD::unpack_reverse_comm(int n, int *list, double *buf) {
  int i, j, m;

  m = 0;
  if (atom->sp_flag) {
     std::cout << "Pair style 'deepmd' does not support spin atoms, please use pair style 'deepspin' instead." << std::endl;
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