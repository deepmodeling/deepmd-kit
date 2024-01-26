// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pair_deepmd.h"

#include <string.h>

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

using namespace LAMMPS_NS;
using namespace std;

static bool is_key(const string &input) {
  vector<string> keys;
  keys.push_back("out_freq");
  keys.push_back("out_file");
  keys.push_back("fparam");
  keys.push_back("aparam");
  keys.push_back("fparam_from_compute");
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

static void ana_st(double &max, double &min, double &sum,
                   const vector<double> &vec, const int &nloc) {
  if (nloc == 0)
    return;
  max = vec[0];
  min = vec[0];
  sum = vec[0];
  for (unsigned ii = 1; ii < nloc; ++ii) {
    if (vec[ii] > max)
      max = vec[ii];
    if (vec[ii] < min)
      min = vec[ii];
    sum += vec[ii];
  }
}

PairDeepMD::PairDeepMD(LAMMPS *lmp)
    : Pair(lmp)

{
  if (strcmp(update->unit_style, "metal") != 0) {
    error->all(
        FLERR,
        "Pair deepmd requires metal unit, please set it by \"units metal\"");
  }
  numb_models = 0;
}

/* ---------------------------------------------------------------------- */

PairDeepMD::~PairDeepMD() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::settings(int narg, char **arg) {
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
    models.push_back(std::string(arg[ii]));
  }
  numb_models = models.size();
  if (numb_models == 1) {
    // try {
    deep_pot.init<double>(std::string(arg[0]));
    // } catch (deepmd_compat::deepmd_exception &e) {
    // error->one(FLERR, e.what());
    // }
  } else {
    // try {
    deep_pot.init<double>(std::string(arg[0]));
    deep_pot_model_devi.init<double>(models);
    // } catch (deepmd_compat::deepmd_exception &e) {
    // error->one(FLERR, e.what());
    // }
  }

  out_freq = 100;
  out_file = "model_devi.out";
  out_each = 0;
  out_rel = 0;
  eps = 0.;
  // fparam.clear();
  // aparam.clear();
  while (iarg < narg) {
    if (!is_key(arg[iarg])) {
      error->all(FLERR,
                 "Illegal pair_style command\nwrong number of parameters\n");
    }
    if (string(arg[iarg]) == string("out_freq")) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Illegal out_freq, not provided");
      out_freq = atoi(arg[iarg + 1]);
      iarg += 2;
    } else if (string(arg[iarg]) == string("out_file")) {
      if (iarg + 1 >= narg)
        error->all(FLERR, "Illegal out_file, not provided");
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
          error->all(FLERR, "invalid fparam_from_compute key: should be "
                            "fparam_from_compute compute_id(str)");
        }
      }
      do_compute = true;
      compute_id = arg[iarg + 1];
      iarg += 1 + 1;
    }

    else if (string(arg[iarg]) == string("atomic")) {
      out_each = 1;
      iarg += 1;
    } else if (string(arg[iarg]) == string("relative")) {
      out_rel = 1;
      eps = atof(arg[iarg + 1]);
      iarg += 2;
    } else if (string(arg[iarg]) == string("relative_v")) {
      out_rel_v = 1;
      eps_v = atof(arg[iarg + 1]);
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

  if (out_freq < 0)
    error->all(FLERR, "Illegal out_freq, should be >= 0");
  if (do_ttm && aparam.size() > 0) {
    error->all(FLERR, "aparam and ttm should NOT be set simultaneously");
  }
  if (do_compute && fparam.size() > 0) {
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
           << setw(18 + 1) << "max_devi_f" << setw(18 + 1) << "min_devi_f"
           << setw(18 + 1) << "avg_devi_f" << endl;
      } else {
        fp.open(out_file, std::ofstream::out | std::ofstream::app);
        fp << scientific;
      }
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::compute(int eflag, int vflag) {
  if (numb_models == 0) {
    return;
  }
  if (eflag || vflag) {
    ev_setup(eflag, vflag);
  }
  if (vflag_atom) {
    error->all(FLERR,
               "6-element atomic virial is not supported. Use compute "
               "centroid/stress/atom command for 9-element atomic virial.");
  }

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  double dener(0);
  vector<double> dforce(nlocal * 3);
  vector<double> dvirial(9, 0);
  vector<double> dcoord(nlocal * 3, 0.);
  vector<int> dtype(nlocal);
  vector<double> dbox(9, 0);

  for (int ii = 0; ii < nlocal; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      dcoord[3 * ii + jj] = x[ii][jj];
    }
    dtype[ii] = type[ii] - 1;
  }

  // get box
  dbox[0] = domain->h[0]; // xx
  dbox[4] = domain->h[1]; // yy
  dbox[8] = domain->h[2]; // zz
  dbox[7] = domain->h[3]; // zy
  dbox[6] = domain->h[4]; // zx
  dbox[3] = domain->h[5]; // yx

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

  if (single_model || multi_models_no_mod_devi) {
    deep_pot.compute<double, double>(dener, dforce, dvirial, dcoord, dtype,
                                     dbox);
  } else if (multi_models_mod_devi) {
    vector<vector<double>> all_virial;
    vector<double> all_energy;
    deep_pot_model_devi.compute<double, double>(
        all_energy, all_force, all_virial, dcoord, dtype, dbox);
    dener = all_energy[0];
    dforce = all_force[0];
    dvirial = all_virial[0];
    if (out_freq > 0 && update->ntimestep % out_freq == 0) {
      int rank = comm->me;
      vector<double> std_f;
      vector<double> tmp_avg_f;
      deep_pot_model_devi.compute_avg(tmp_avg_f, all_force);
      deep_pot_model_devi.compute_std_f(std_f, tmp_avg_f, all_force);
      if (out_rel == 1) {
        deep_pot_model_devi.compute_relative_std_f(std_f, tmp_avg_f, eps);
      }
      double min = numeric_limits<double>::max(), max = 0, avg = 0;
      ana_st(max, min, avg, std_f, nlocal);
      int all_nlocal = 0;
      MPI_Reduce(&nlocal, &all_nlocal, 1, MPI_INT, MPI_SUM, 0, world);
      double all_f_min = 0, all_f_max = 0, all_f_avg = 0;
      MPI_Reduce(&min, &all_f_min, 1, MPI_DOUBLE, MPI_MIN, 0, world);
      MPI_Reduce(&max, &all_f_max, 1, MPI_DOUBLE, MPI_MAX, 0, world);
      MPI_Reduce(&avg, &all_f_avg, 1, MPI_DOUBLE, MPI_SUM, 0, world);
      all_f_avg /= double(all_nlocal);
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
        fp << setw(12) << update->ntimestep << " " << setw(18) << all_v_max
           << " " << setw(18) << all_v_min << " " << setw(18) << all_v_avg
           << " " << setw(18) << all_f_max << " " << setw(18) << all_f_min
           << " " << setw(18) << all_f_avg;
      }
      if (out_each == 1) {
        vector<double> std_f_all(all_nlocal);
        // Gather std_f and tags
        tagint *tag = atom->tag;
        int nprocs = comm->nprocs;
        for (int ii = 0; ii < nlocal; ii++) {
          tagsend[ii] = tag[ii];
          stdfsend[ii] = std_f[ii];
        }
        MPI_Gather(&nlocal, 1, MPI_INT, counts, 1, MPI_INT, 0, world);
        displacements[0] = 0;
        for (int ii = 0; ii < nprocs - 1; ii++)
          displacements[ii + 1] = displacements[ii] + counts[ii];
        MPI_Gatherv(tagsend, nlocal, MPI_LMP_TAGINT, tagrecv, counts,
                    displacements, MPI_LMP_TAGINT, 0, world);
        MPI_Gatherv(stdfsend, nlocal, MPI_DOUBLE, stdfrecv, counts,
                    displacements, MPI_DOUBLE, 0, world);
        if (rank == 0) {
          for (int dd = 0; dd < all_nlocal; ++dd) {
            std_f_all[tagrecv[dd] - 1] = stdfrecv[dd];
          }
          for (int dd = 0; dd < all_nlocal; ++dd) {
            fp << " " << setw(18) << std_f_all[dd];
          }
        }
      }
      if (rank == 0) {
        fp << endl;
      }
    }
  }

  for (int ii = 0; ii < nlocal; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      f[ii][jj] = dforce[3 * ii + jj];
    }
  }

  // accumulate energy and virial
  if (eflag) {
    eng_vdwl += dener;
  }
  if (vflag) {
    virial[0] += 1.0 * dvirial[0];
    virial[1] += 1.0 * dvirial[4];
    virial[2] += 1.0 * dvirial[8];
    virial[3] += 1.0 * dvirial[3];
    virial[4] += 1.0 * dvirial[6];
    virial[5] += 1.0 * dvirial[7];
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDeepMD::coeff(int narg, char **arg) {
  // if (narg < 4 || narg > 5) error->all(FLERR, "Incorrect args for pair
  // coefficients");
  if (!allocated)
    allocate();

  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      // epsilon[i][j] = epsilon_one;
      // sigma[i][j] = sigma_one;
      // cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0)
    error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::allocate() {
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(scale, n + 1, n + 1, "pair:scale");

  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) {
      setflag[i][j] = 0;
      scale[i][j] = 0;
    }
  }
  for (int i = 1; i <= numb_types; ++i) {
    if (i > n) {
      continue;
    }
    for (int j = i; j <= numb_types; ++j) {
      if (j > n) {
        continue;
      }
      setflag[i][j] = 1;
      scale[i][j] = 1;
    }
  }
}
