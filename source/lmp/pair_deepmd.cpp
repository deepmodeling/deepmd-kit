// SPDX-License-Identifier: LGPL-3.0-or-later
#include <string.h>

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

int PairDeepMD::get_node_rank() {
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

std::string PairDeepMD::get_file_content(const std::string &model) {
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

std::vector<std::string> PairDeepMD::get_file_content(
    const std::vector<std::string> &models) {
  std::vector<std::string> file_contents(models.size());
  for (unsigned ii = 0; ii < models.size(); ++ii) {
    file_contents[ii] = get_file_content(models[ii]);
  }
  return file_contents;
}

static void ana_st(double &max,
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

static void make_uniform_aparam(vector<double> &daparam,
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

void PairDeepMD::make_fparam_from_compute(vector<double> &fparam) {
  assert(do_compute_fparam);

  int icompute = modify->find_compute(compute_fparam_id);
  Compute *compute = modify->compute[icompute];

  assert(compute);
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

void PairDeepMD::make_aparam_from_compute(vector<double> &aparam) {
  assert(do_compute_aparam);

  int icompute = modify->find_compute(compute_aparam_id);
  Compute *compute = modify->compute[icompute];

  assert(compute);
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
void PairDeepMD::make_ttm_fparam(vector<double> &fparam) {
  assert(do_ttm);
  // get ttm_fix
  const FixTTMDP *ttm_fix = NULL;
  for (int ii = 0; ii < modify->nfix; ii++) {
    if (string(modify->fix[ii]->id) == ttm_fix_id) {
      ttm_fix = dynamic_cast<FixTTMDP *>(modify->fix[ii]);
    }
  }
  assert(ttm_fix);

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
void PairDeepMD::make_ttm_aparam(vector<double> &daparam) {
  assert(do_ttm);
  // get ttm_fix
  const FixTTMDP *ttm_fix = NULL;
  for (int ii = 0; ii < modify->nfix; ii++) {
    if (string(modify->fix[ii]->id) == ttm_fix_id) {
      ttm_fix = dynamic_cast<FixTTMDP *>(modify->fix[ii]);
    }
  }
  assert(ttm_fix);
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

void PairDeepMD::cum_sum(std::map<int, int> &sum, std::map<int, int> &vec) {
  sum[0] = 0;
  for (int ii = 1; ii < vec.size(); ++ii) {
    sum[ii] = sum[ii - 1] + vec[ii - 1];
  }
}

PairDeepMD::PairDeepMD(LAMMPS *lmp)
    : Pair(lmp)

{
  if (lmp->citeme) {
    lmp->citeme->add(cite_user_deepmd_package);
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

void PairDeepMD::print_summary(const string pre) const {
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
    cout << pre << "build float prec:   " << STR_FLOAT_PREC << endl;
    cout << pre << "build with tf inc:  " << STR_TensorFlow_INCLUDE_DIRS
         << endl;
    cout << pre << "build with tf lib:  " << STR_TensorFlow_LIBRARY << endl;

    std::cout.rdbuf(sbuf);
    utils::logmesg(lmp, buffer.str());
  }
}

PairDeepMD::~PairDeepMD() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
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
        dspin[ii * 3 + dd] = sp[ii][dd];
      }
    }
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
    deepmd_compat::InputNlist lmp_list(list->inum, list->ilist, list->numneigh,
                                       list->firstneigh);
    deepmd_compat::InputNlist extend_lmp_list;
    if (atom->sp_flag) {
      extend(extend_inum, extend_ilist, extend_numneigh, extend_neigh,
             extend_firstneigh, extend_dcoord, extend_dtype, extend_nghost,
             new_idx_map, old_idx_map, lmp_list, dcoord, dtype, nghost, dspin,
             numb_types, numb_types_spin, virtual_len);
      extend_lmp_list =
          deepmd_compat::InputNlist(extend_inum, &extend_ilist[0],
                                    &extend_numneigh[0], &extend_firstneigh[0]);
    }
    if (single_model || multi_models_no_mod_devi) {
      // cvflag_atom is the right flag for the cvatom matrix
      if (!(eflag_atom || cvflag_atom)) {
        if (!atom->sp_flag) {
          try {
            deep_pot.compute(dener, dforce, dvirial, dcoord, dtype, dbox,
                             nghost, lmp_list, ago, fparam, daparam);
          } catch (deepmd_compat::deepmd_exception &e) {
            error->one(FLERR, e.what());
          }
        } else {
          dforce.resize(static_cast<size_t>(extend_inum + extend_nghost) * 3);
          try {
            deep_pot.compute(dener, dforce, dvirial, extend_dcoord,
                             extend_dtype, dbox, extend_nghost, extend_lmp_list,
                             ago, fparam, daparam);
          } catch (deepmd_compat::deepmd_exception &e) {
            error->one(FLERR, e.what());
          }
        }
      }
      // do atomic energy and virial
      else {
        vector<double> deatom(nall * 1, 0);
        vector<double> dvatom(nall * 9, 0);
        if (!atom->sp_flag) {
          try {
            deep_pot.compute(dener, dforce, dvirial, deatom, dvatom, dcoord,
                             dtype, dbox, nghost, lmp_list, ago, fparam,
                             daparam);
          } catch (deepmd_compat::deepmd_exception &e) {
            error->one(FLERR, e.what());
          }
        } else {
          dforce.resize(static_cast<size_t>(extend_inum + extend_nghost) * 3);
          try {
            deep_pot.compute(dener, dforce, dvirial, extend_dcoord,
                             extend_dtype, dbox, extend_nghost, extend_lmp_list,
                             ago, fparam, daparam);
          } catch (deepmd_compat::deepmd_exception &e) {
            error->one(FLERR, e.what());
          }
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
      try {
        deep_pot_model_devi.compute(
            all_energy, all_force, all_virial, all_atom_energy, all_atom_virial,
            dcoord, dtype, dbox, nghost, lmp_list, ago, fparam, daparam);
      } catch (deepmd_compat::deepmd_exception &e) {
        error->one(FLERR, e.what());
      }
      // deep_pot_model_devi.compute_avg (dener, all_energy);
      // deep_pot_model_devi.compute_avg (dforce, all_force);
      // deep_pot_model_devi.compute_avg (dvirial, all_virial);
      // deep_pot_model_devi.compute_avg (deatom, all_atom_energy);
      // deep_pot_model_devi.compute_avg (dvatom, all_atom_virial);
      dener = all_energy[0];
      dforce = all_force[0];
      dvirial = all_virial[0];
      deatom = all_atom_energy[0];
      dvatom = all_atom_virial[0];
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
        ana_st(max, min, avg, std_f, nlocal);
        double all_f_min = 0, all_f_max = 0, all_f_avg = 0;
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
  if (!atom->sp_flag) {
    for (int ii = 0; ii < nall; ++ii) {
      for (int dd = 0; dd < 3; ++dd) {
        f[ii][dd] += scale[1][1] * dforce[3 * ii + dd] * force_unit_cvt_factor;
      }
    }
  } else {
    // unit_factor = hbar / spin_norm;
    const double hbar = 6.5821191e-04;
    for (int ii = 0; ii < nall; ++ii) {
      for (int dd = 0; dd < 3; ++dd) {
        int new_idx = new_idx_map[ii];
        f[ii][dd] +=
            scale[1][1] * dforce[3 * new_idx + dd] * force_unit_cvt_factor;
        if (dtype[ii] < numb_types_spin && ii < nlocal) {
          fm[ii][dd] += scale[1][1] * dforce[3 * (new_idx + nlocal) + dd] /
                        (hbar / spin_norm[dtype[ii]]) * force_unit_cvt_factor;
        } else if (dtype[ii] < numb_types_spin) {
          fm[ii][dd] += scale[1][1] * dforce[3 * (new_idx + nghost) + dd] /
                        (hbar / spin_norm[dtype[ii]]) * force_unit_cvt_factor;
        }
      }
    }
  }

  if (atom->sp_flag) {
    std::map<int, int>().swap(new_idx_map);
    std::map<int, int>().swap(old_idx_map);
    // malloc_trim(0);
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

  comm_reverse = numb_models * 3;
  all_force.resize(numb_models);
}

void PairDeepMD::read_restart(FILE *) { is_restart = true; }

void PairDeepMD::write_restart(FILE *) {
  // pass
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDeepMD::coeff(int narg, char **arg) {
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

void PairDeepMD::init_style() {
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

double PairDeepMD::init_one(int i, int j) {
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

/* ---------------------------------------------------------------------- */

int PairDeepMD::pack_reverse_comm(int n, int first, double *buf) {
  int i, m, last;

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    for (int dd = 0; dd < numb_models; ++dd) {
      buf[m++] = all_force[dd][3 * i + 0];
      buf[m++] = all_force[dd][3 * i + 1];
      buf[m++] = all_force[dd][3 * i + 2];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void PairDeepMD::unpack_reverse_comm(int n, int *list, double *buf) {
  int i, j, m;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    for (int dd = 0; dd < numb_models; ++dd) {
      all_force[dd][3 * j + 0] += buf[m++];
      all_force[dd][3 * j + 1] += buf[m++];
      all_force[dd][3 * j + 2] += buf[m++];
    }
  }
}

void *PairDeepMD::extract(const char *str, int &dim) {
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

void PairDeepMD::extend(int &extend_inum,
                        std::vector<int> &extend_ilist,
                        std::vector<int> &extend_numneigh,
                        std::vector<vector<int>> &extend_neigh,
                        std::vector<int *> &extend_firstneigh,
                        std::vector<double> &extend_dcoord,
                        std::vector<int> &extend_atype,
                        int &extend_nghost,
                        std::map<int, int> &new_idx_map,
                        std::map<int, int> &old_idx_map,
                        const deepmd_compat::InputNlist &lmp_list,
                        const std::vector<double> &dcoord,
                        const std::vector<int> &atype,
                        const int nghost,
                        const std::vector<double> &spin,
                        const int numb_types,
                        const int numb_types_spin,
                        const std::vector<double> &virtual_len) {
  extend_ilist.clear();
  extend_numneigh.clear();
  extend_neigh.clear();
  extend_firstneigh.clear();
  extend_dcoord.clear();
  extend_atype.clear();

  int nall = dcoord.size() / 3;
  int nloc = nall - nghost;
  assert(nloc == lmp_list.inum);

  // record numb_types_real and nloc_virt
  int numb_types_real = numb_types - numb_types_spin;
  std::map<int, int> loc_type_count;
  std::map<int, int>::iterator iter = loc_type_count.begin();
  for (int i = 0; i < nloc; i++) {
    iter = loc_type_count.find(atype[i]);
    if (iter != loc_type_count.end()) {
      iter->second += 1;
    } else {
      loc_type_count.insert(pair<int, int>(atype[i], 1));
    }
  }
  assert(numb_types_real - 1 == loc_type_count.rbegin()->first);
  int nloc_virt = 0;
  for (int i = 0; i < numb_types_spin; i++) {
    nloc_virt += loc_type_count[i];
  }

  // record nghost_virt
  std::map<int, int> ghost_type_count;
  for (int i = nloc; i < nall; i++) {
    iter = ghost_type_count.find(atype[i]);
    if (iter != ghost_type_count.end()) {
      iter->second += 1;
    } else {
      ghost_type_count.insert(pair<int, int>(atype[i], 1));
    }
  }
  int nghost_virt = 0;
  for (int i = 0; i < numb_types_spin; i++) {
    nghost_virt += ghost_type_count[i];
  }

  // for extended system, search new index by old index, and vice versa
  extend_nghost = nghost + nghost_virt;
  int extend_nloc = nloc + nloc_virt;
  int extend_nall = extend_nloc + extend_nghost;
  std::map<int, int> cum_loc_type_count;
  std::map<int, int> cum_ghost_type_count;
  cum_sum(cum_loc_type_count, loc_type_count);
  cum_sum(cum_ghost_type_count, ghost_type_count);
  std::vector<int> loc_type_reset(numb_types_real, 0);
  std::vector<int> ghost_type_reset(numb_types_real, 0);

  new_idx_map.clear();
  old_idx_map.clear();
  for (int ii = 0; ii < nloc; ii++) {
    int new_idx = cum_loc_type_count[atype[ii]] + loc_type_reset[atype[ii]];
    new_idx_map[ii] = new_idx;
    old_idx_map[new_idx] = ii;
    loc_type_reset[atype[ii]]++;
  }
  for (int ii = nloc; ii < nall; ii++) {
    int new_idx = cum_ghost_type_count[atype[ii]] +
                  ghost_type_reset[atype[ii]] + extend_nloc;
    new_idx_map[ii] = new_idx;
    old_idx_map[new_idx] = ii;
    ghost_type_reset[atype[ii]]++;
  }

  // extend lmp_list
  extend_inum = extend_nloc;

  extend_ilist.resize(extend_nloc);
  for (int ii = 0; ii < extend_nloc; ii++) {
    extend_ilist[ii] = ii;
  }

  extend_neigh.resize(extend_nloc);
  for (int ii = 0; ii < nloc; ii++) {
    int jnum = lmp_list.numneigh[old_idx_map[ii]];
    const int *jlist = lmp_list.firstneigh[old_idx_map[ii]];
    if (atype[old_idx_map[ii]] < numb_types_spin) {
      extend_neigh[ii].push_back(ii + nloc);
    }
    for (int jj = 0; jj < jnum; jj++) {
      int new_idx = new_idx_map[jlist[jj]];
      extend_neigh[ii].push_back(new_idx);
      if (atype[jlist[jj]] < numb_types_spin && jlist[jj] < nloc) {
        extend_neigh[ii].push_back(new_idx + nloc);
      } else if (atype[jlist[jj]] < numb_types_spin && jlist[jj] < nall) {
        extend_neigh[ii].push_back(new_idx + nghost);
      }
    }
  }
  for (int ii = nloc; ii < extend_nloc; ii++) {
    extend_neigh[ii].assign(extend_neigh[ii - nloc].begin(),
                            extend_neigh[ii - nloc].end());
    std::vector<int>::iterator it =
        find(extend_neigh[ii].begin(), extend_neigh[ii].end(), ii);
    *it = ii - nloc;
  }

  extend_firstneigh.resize(extend_nloc);
  extend_numneigh.resize(extend_nloc);
  for (int ii = 0; ii < extend_nloc; ii++) {
    extend_firstneigh[ii] = &extend_neigh[ii][0];
    extend_numneigh[ii] = extend_neigh[ii].size();
  }

  // extend coord
  extend_dcoord.resize(static_cast<size_t>(extend_nall) * 3);
  for (int ii = 0; ii < nloc; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      extend_dcoord[new_idx_map[ii] * 3 + jj] = dcoord[ii * 3 + jj];
      if (atype[ii] < numb_types_spin) {
        double temp_dcoord =
            dcoord[ii * 3 + jj] + spin[ii * 3 + jj] * virtual_len[atype[ii]];
        extend_dcoord[(new_idx_map[ii] + nloc) * 3 + jj] = temp_dcoord;
      }
    }
  }
  for (int ii = nloc; ii < nall; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      extend_dcoord[new_idx_map[ii] * 3 + jj] = dcoord[ii * 3 + jj];
      if (atype[ii] < numb_types_spin) {
        double temp_dcoord =
            dcoord[ii * 3 + jj] + spin[ii * 3 + jj] * virtual_len[atype[ii]];
        extend_dcoord[(new_idx_map[ii] + nghost) * 3 + jj] = temp_dcoord;
      }
    }
  }

  // extend atype
  extend_atype.resize(extend_nall);
  for (int ii = 0; ii < nall; ii++) {
    extend_atype[new_idx_map[ii]] = atype[ii];
    if (atype[ii] < numb_types_spin) {
      if (ii < nloc) {
        extend_atype[new_idx_map[ii] + nloc] = atype[ii] + numb_types_real;
      } else {
        extend_atype[new_idx_map[ii] + nghost] = atype[ii] + numb_types_real;
      }
    }
  }
}
