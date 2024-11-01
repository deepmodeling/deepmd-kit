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
#include "pair_base.h"

using namespace LAMMPS_NS;
using namespace std;

static int stringCmp(const void *a, const void *b) {
  char *m = (char *)a;
  char *n = (char *)b;
  int i, sum = 0;

  for (i = 0; i < MPI_MAX_PROCESSOR_NAME; i++) {
    if (m[i] == n[i]) {
      continue;
    } else {
      sum = m[i] - n[i];
      break;
    }
  }
  return sum;
}

int PairDeepMDBase::get_node_rank() {
  char host_name[MPI_MAX_PROCESSOR_NAME];
  memset(host_name, '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
  char(*host_names)[MPI_MAX_PROCESSOR_NAME];
  int n, namelen, color, rank, nprocs, myrank;
  size_t bytes;
  MPI_Comm nodeComm;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Get_processor_name(host_name, &namelen);

  bytes = nprocs * sizeof(char[MPI_MAX_PROCESSOR_NAME]);
  host_names = (char(*)[MPI_MAX_PROCESSOR_NAME])malloc(bytes);
  for (int ii = 0; ii < nprocs; ii++) {
    memset(host_names[ii], '\0', sizeof(char) * MPI_MAX_PROCESSOR_NAME);
  }

  strcpy(host_names[rank], host_name);

  for (n = 0; n < nprocs; n++) {
    MPI_Bcast(&(host_names[n]), MPI_MAX_PROCESSOR_NAME, MPI_CHAR, n,
              MPI_COMM_WORLD);
  }
  qsort(host_names, nprocs, sizeof(char[MPI_MAX_PROCESSOR_NAME]), stringCmp);

  color = 0;
  for (n = 0; n < nprocs - 1; n++) {
    if (strcmp(host_name, host_names[n]) == 0) {
      break;
    }
    if (strcmp(host_names[n], host_names[n + 1])) {
      color++;
    }
  }

  MPI_Comm_split(MPI_COMM_WORLD, color, 0, &nodeComm);
  MPI_Comm_rank(nodeComm, &myrank);

  MPI_Barrier(MPI_COMM_WORLD);
  int looprank = myrank;
  // printf (" Assigning device %d  to process on node %s rank %d,
  // OK\n",looprank,  host_name, rank );
  free(host_names);
  return looprank;
}

std::string PairDeepMDBase::get_file_content(const std::string &model) {
  int myrank = 0, root = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  int nchar = 0;
  std::string file_content;
  if (myrank == root) {
    deepmd_compat::read_file_to_string(model, file_content);
    nchar = file_content.size();
  }
  MPI_Bcast(&nchar, 1, MPI_INT, root, MPI_COMM_WORLD);
  char *buff = (char *)malloc(sizeof(char) * nchar);
  if (myrank == root) {
    memcpy(buff, file_content.c_str(), sizeof(char) * nchar);
  }
  MPI_Bcast(buff, nchar, MPI_CHAR, root, MPI_COMM_WORLD);
  file_content.resize(nchar);
  for (unsigned ii = 0; ii < nchar; ++ii) {
    file_content[ii] = buff[ii];
  }
  free(buff);
  return file_content;
}

std::vector<std::string> PairDeepMDBase::get_file_content(
    const std::vector<std::string> &models) {
  std::vector<std::string> file_contents(models.size());
  for (unsigned ii = 0; ii < models.size(); ++ii) {
    file_contents[ii] = get_file_content(models[ii]);
  }
  return file_contents;
}

void PairDeepMDBase::make_fparam_from_compute(vector<double> &fparam) {
  assert(do_compute_fparam);

  int icompute = modify->find_compute(compute_fparam_id);
  Compute *compute = modify->compute[icompute];

  if (!compute) {
    error->all(FLERR, "compute id is not found: " + compute_fparam_id);
  }
  fparam.resize(dim_fparam);

  if (dim_fparam == 1) {
    if (!(compute->invoked_flag & Compute::INVOKED_SCALAR)) {
      compute->compute_scalar();
      compute->invoked_flag |= Compute::INVOKED_SCALAR;
    }
    fparam[0] = compute->scalar;
  } else if (dim_fparam > 1) {
    if (!(compute->invoked_flag & Compute::INVOKED_VECTOR)) {
      compute->compute_vector();
      compute->invoked_flag |= Compute::INVOKED_VECTOR;
    }
    double *cvector = compute->vector;
    for (int jj = 0; jj < dim_fparam; ++jj) {
      fparam[jj] = cvector[jj];
    }
  }
}

void PairDeepMDBase::make_aparam_from_compute(vector<double> &aparam) {
  assert(do_compute_aparam);

  int icompute = modify->find_compute(compute_aparam_id);
  Compute *compute = modify->compute[icompute];

  if (!compute) {
    error->all(FLERR, "compute id is not found: " + compute_aparam_id);
  }
  int nlocal = atom->nlocal;
  aparam.resize(static_cast<size_t>(dim_aparam) * nlocal);

  if (!(compute->invoked_flag & Compute::INVOKED_PERATOM)) {
    compute->compute_peratom();
    compute->invoked_flag |= Compute::INVOKED_PERATOM;
  }
  if (dim_aparam == 1) {
    double *cvector = compute->vector_atom;
    aparam.assign(cvector, cvector + nlocal);
  } else if (dim_aparam > 1) {
    double **carray = compute->array_atom;
    for (int ii = 0; ii < nlocal; ++ii) {
      for (int jj = 0; jj < dim_aparam; ++jj) {
        aparam[ii * dim_aparam + jj] = carray[ii][jj];
      }
    }
  }
}

#ifdef USE_TTM
void PairDeepMDBase::make_ttm_fparam(vector<double> &fparam) {
  assert(do_ttm);
  // get ttm_fix
  const FixTTMDP *ttm_fix = NULL;
  for (int ii = 0; ii < modify->nfix; ii++) {
    if (string(modify->fix[ii]->id) == ttm_fix_id) {
      ttm_fix = dynamic_cast<FixTTMDP *>(modify->fix[ii]);
    }
  }
  if (!ttm_fix) {
    error->all(FLERR, "fix ttm id is not found: " + ttm_fix_id);
  }

  fparam.resize(dim_fparam);

  vector<int> nnodes = ttm_fix->get_nodes();
  int nxnodes = nnodes[0];
  int nynodes = nnodes[1];
  int nznodes = nnodes[2];
  double ***const T_electron = ttm_fix->get_T_electron();

  int numb_effective_nodes = 0;
  double total_Te = 0;

  // loop over grids to get average electron temperature
  for (int ixnode = 0; ixnode < nxnodes; ixnode++) {
    for (int iynode = 0; iynode < nynodes; iynode++) {
      for (int iznode = 0; iznode < nznodes; iznode++) {
        if (T_electron[ixnode][iynode][iznode] != 0) {
          numb_effective_nodes += 1;
          total_Te += T_electron[ixnode][iynode][iznode];
        }
      }
    }
  }

  fparam[0] = total_Te / numb_effective_nodes;
}
#endif

#ifdef USE_TTM
void PairDeepMDBase::make_ttm_aparam(vector<double> &daparam) {
  assert(do_ttm);
  // get ttm_fix
  const FixTTMDP *ttm_fix = NULL;
  for (int ii = 0; ii < modify->nfix; ii++) {
    if (string(modify->fix[ii]->id) == ttm_fix_id) {
      ttm_fix = dynamic_cast<FixTTMDP *>(modify->fix[ii]);
    }
  }
  if (!ttm_fix) {
    error->all(FLERR, "fix ttm id is not found: " + ttm_fix_id);
  }
  // modify
  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  vector<int> nnodes = ttm_fix->get_nodes();
  int nxnodes = nnodes[0];
  int nynodes = nnodes[1];
  int nznodes = nnodes[2];
  double ***const T_electron = ttm_fix->get_T_electron();
  double dx = domain->xprd / nxnodes;
  double dy = domain->yprd / nynodes;
  double dz = domain->zprd / nynodes;
  // resize daparam
  daparam.resize(nlocal);
  // loop over atoms to assign aparam
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & ttm_fix->groupbit) {
      double xscale = (x[ii][0] - domain->boxlo[0]) / domain->xprd;
      double yscale = (x[ii][1] - domain->boxlo[1]) / domain->yprd;
      double zscale = (x[ii][2] - domain->boxlo[2]) / domain->zprd;
      int ixnode = static_cast<int>(xscale * nxnodes);
      int iynode = static_cast<int>(yscale * nynodes);
      int iznode = static_cast<int>(zscale * nznodes);
      // https://stackoverflow.com/a/1907585/9567349
      ixnode = ((ixnode % nxnodes) + nxnodes) % nxnodes;
      iynode = ((iynode % nynodes) + nynodes) % nynodes;
      iznode = ((iznode % nznodes) + nznodes) % nznodes;
      daparam[ii] = T_electron[ixnode][iynode][iznode];
    }
  }
}
#endif

void PairDeepMDBase::cum_sum(std::map<int, int> &sum, std::map<int, int> &vec) {
  sum[0] = 0;
  for (int ii = 1; ii < vec.size(); ++ii) {
    sum[ii] = sum[ii - 1] + vec[ii - 1];
  }
}

PairDeepMDBase::PairDeepMDBase(LAMMPS *lmp, const char *cite_user_package)
    : Pair(lmp)

{
  if (lmp->citeme) {
    lmp->citeme->add(cite_user_package);
  }
  if (strcmp(update->unit_style, "lj") == 0) {
    error->all(FLERR,
               "Pair deepmd does not support unit style lj. Please use other "
               "unit styles like metal or real unit instead. You may set it by "
               "\"units metal\" or \"units real\"");
  }
  ener_unit_cvt_factor = force->boltz / 8.617343e-5;
  dist_unit_cvt_factor = force->angstrom;
  force_unit_cvt_factor = ener_unit_cvt_factor / dist_unit_cvt_factor;

  restartinfo = 1;
#if LAMMPS_VERSION_NUMBER >= 20201130
  centroidstressflag =
      CENTROID_AVAIL;  // set centroidstressflag = CENTROID_AVAIL to allow the
                       // use of the centroid/stress/atom. Added by Davide Tisi
#else
  centroidstressflag = 2;  // set centroidstressflag = 2 to allow the use of the
                           // centroid/stress/atom. Added by Davide Tisi
#endif
  pppmflag = 1;
  respa_enable = 0;
  writedata = 0;

  cutoff = 0.;
  numb_types = 0;
  numb_types_spin = 0;
  numb_models = 0;
  out_freq = 0;
  out_each = 0;
  out_rel = 0;
  out_rel_v = 0;
  stdf_comm_buff_size = 0;
  eps = 0.;
  eps_v = 0.;
  scale = NULL;
  do_ttm = false;
  do_compute_fparam = false;
  do_compute_aparam = false;
  single_model = false;
  multi_models_mod_devi = false;
  multi_models_no_mod_devi = false;
  is_restart = false;
  // set comm size needed by this Pair
  comm_reverse = 1;

  print_summary("  ");
}

void PairDeepMDBase::print_summary(const string pre) const {
  if (comm->me == 0) {
    // capture cout to a string, then call LAMMPS's utils::logmesg
    // https://stackoverflow.com/a/4043813/9567349
    std::stringstream buffer;
    std::streambuf *sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());

    cout << "Summary of lammps deepmd module ..." << endl;
    cout << pre << ">>> Info of deepmd-kit:" << endl;
    deep_pot.print_summary(pre);
    cout << pre << ">>> Info of lammps module:" << endl;
    cout << pre << "use deepmd-kit at:  " << STR_DEEPMD_ROOT << endl;
    cout << pre << "source:             " << STR_GIT_SUMM << endl;
    cout << pre << "source branch:      " << STR_GIT_BRANCH << endl;
    cout << pre << "source commit:      " << STR_GIT_HASH << endl;
    cout << pre << "source commit at:   " << STR_GIT_DATE << endl;
    cout << pre << "build with inc:     " << STR_BACKEND_INCLUDE_DIRS << endl;
    cout << pre << "build with lib:     " << STR_BACKEND_LIBRARY_PATH << endl;

    std::cout.rdbuf(sbuf);
    utils::logmesg(lmp, buffer.str());
  }
}

PairDeepMDBase::~PairDeepMDBase() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

void PairDeepMDBase::allocate() {
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
      scale[i][j] = 1.0;
    }
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

void PairDeepMDBase::settings(int narg, char **arg) {
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
    } catch (deepmd_compat::deepmd_exception &e) {
      error->one(FLERR, e.what());
    }
    cutoff = deep_pot.cutoff() * dist_unit_cvt_factor;
    numb_types = deep_pot.numb_types();
    numb_types_spin = deep_pot.numb_types_spin();
    dim_fparam = deep_pot.dim_fparam();
    dim_aparam = deep_pot.dim_aparam();
  } else {
    try {
      deep_pot.init(arg[0], get_node_rank(), get_file_content(arg[0]));
      deep_pot_model_devi.init(models, get_node_rank(),
                               get_file_content(models));
    } catch (deepmd_compat::deepmd_exception &e) {
      error->one(FLERR, e.what());
    }
    cutoff = deep_pot_model_devi.cutoff() * dist_unit_cvt_factor;
    numb_types = deep_pot_model_devi.numb_types();
    numb_types_spin = deep_pot_model_devi.numb_types_spin();
    dim_fparam = deep_pot_model_devi.dim_fparam();
    dim_aparam = deep_pot_model_devi.dim_aparam();
    assert(cutoff == deep_pot.cutoff() * dist_unit_cvt_factor);
    assert(numb_types == deep_pot.numb_types());
    assert(numb_types_spin == deep_pot.numb_types_spin());
    assert(dim_fparam == deep_pot.dim_fparam());
    assert(dim_aparam == deep_pot.dim_aparam());
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
        if (!atom->sp_flag) {
          fp << "#" << setw(12 - 1) << "step" << setw(18 + 1) << "max_devi_v"
             << setw(18 + 1) << "min_devi_v" << setw(18 + 1) << "avg_devi_v"
             << setw(18 + 1) << "max_devi_f" << setw(18 + 1) << "min_devi_f"
             << setw(18 + 1) << "avg_devi_f";
          if (out_each) {
            // at this time, we don't know how many atoms
            fp << setw(18 + 1) << "atm_devi_f(N)";
          }
          fp << endl;
        } else {
          fp << "#" << setw(12 - 1) << "step" << setw(18 + 1) << "max_devi_v"
             << setw(18 + 1) << "min_devi_v" << setw(18 + 1) << "avg_devi_v"
             << setw(18 + 1) << "max_devi_fr" << setw(18 + 1) << "min_devi_fr"
             << setw(18 + 1) << "avg_devi_fr" << setw(18 + 1) << "max_devi_fm"
             << setw(18 + 1) << "min_devi_fm" << setw(18 + 1) << "avg_devi_fm"
             << endl;
        }
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

  // comm_reverse = numb_models * 3;
  if (atom->sp_flag) {
    comm_reverse = numb_models * 3 * 2;
  } else {
    comm_reverse = numb_models * 3;
  }
  all_force.resize(numb_models);
}

void PairDeepMDBase::read_restart(FILE *) { is_restart = true; }

void PairDeepMDBase::write_restart(FILE *) {
  // pass
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDeepMDBase::coeff(int narg, char **arg) {
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

void PairDeepMDBase::init_style() {
#if LAMMPS_VERSION_NUMBER >= 20220324
  neighbor->add_request(this, NeighConst::REQ_FULL);
#else
  int irequest = neighbor->request(this, instance_me);
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  // neighbor->requests[irequest]->newton = 2;
#endif
  if (out_each == 1) {
    int ntotal = atom->natoms;
    int nprocs = comm->nprocs;
    if (ntotal > stdf_comm_buff_size) {
      stdf_comm_buff_size = ntotal;
    }
    memory->create(counts, nprocs, "deepmd:counts");
    memory->create(displacements, nprocs, "deepmd:displacements");
    memory->create(stdfsend, ntotal, "deepmd:stdfsendall");
    memory->create(stdfrecv, ntotal, "deepmd:stdfrecvall");
    memory->create(tagsend, ntotal, "deepmd:tagsendall");
    memory->create(tagrecv, ntotal, "deepmd:tagrecvall");
  }
}

double PairDeepMDBase::init_one(int i, int j) {
  if (i > numb_types || j > numb_types) {
    char warning_msg[1024];
    sprintf(warning_msg,
            "Interaction between types %d and %d is set with deepmd, but will "
            "be ignored.\n Deepmd model has only %d types, it only computes "
            "the mulitbody interaction of types: 1-%d.",
            i, j, numb_types, numb_types);
    error->warning(FLERR, warning_msg);
  }

  if (setflag[i][j] == 0) {
    scale[i][j] = 1.0;
  }
  scale[j][i] = scale[i][j];

  return cutoff;
}

void *PairDeepMDBase::extract(const char *str, int &dim) {
  if (strcmp(str, "cut_coul") == 0) {
    dim = 0;
    return (void *)&cutoff;
  }
  if (strcmp(str, "scale") == 0) {
    dim = 2;
    return (void *)scale;
  }
  return NULL;
}

void ana_st(double &max,
            double &min,
            double &sum,
            const vector<double> &vec,
            const int &nloc) {
  if (nloc == 0) {
    return;
  }
  max = vec[0];
  min = vec[0];
  sum = vec[0];
  for (unsigned ii = 1; ii < nloc; ++ii) {
    if (vec[ii] > max) {
      max = vec[ii];
    }
    if (vec[ii] < min) {
      min = vec[ii];
    }
    sum += vec[ii];
  }
}

void make_uniform_aparam(vector<double> &daparam,
                         const vector<double> &aparam,
                         const int &nlocal) {
  unsigned dim_aparam = aparam.size();
  daparam.resize(static_cast<size_t>(dim_aparam) * nlocal);
  for (int ii = 0; ii < nlocal; ++ii) {
    for (int jj = 0; jj < dim_aparam; ++jj) {
      daparam[ii * dim_aparam + jj] = aparam[jj];
    }
  }
}
