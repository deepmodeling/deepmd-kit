// SPDX-License-Identifier: LGPL-3.0-or-later
#include "fix_dplr.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "input.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pppm_dplr.h"
#include "update.h"
#include "variable.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

static bool is_key(const string &input) {
  vector<string> keys;
  keys.push_back("model");
  keys.push_back("type_associate");
  keys.push_back("bond_type");
  keys.push_back("efield");
  for (int ii = 0; ii < keys.size(); ++ii) {
    if (input == keys[ii]) {
      return true;
    }
  }
  return false;
}

FixDPLR::FixDPLR(LAMMPS *lmp, int narg, char **arg)
    : Fix(lmp, narg, arg),
      xstr(nullptr),
      ystr(nullptr),
      zstr(nullptr),
      efield(3, 0.0),
      efield_fsum(4, 0.0),
      efield_fsum_all(4, 0.0),
      efield_force_flag(0) {
#if LAMMPS_VERSION_NUMBER >= 20210210
  // lammps/lammps#2560
  energy_global_flag = 1;
  virial_global_flag = 1;
#else
  virial_flag = 1;
#endif

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 3;
  qe2f = force->qe2f;
  xstyle = ystyle = zstyle = NONE;

  if (strcmp(update->unit_style, "lj") == 0) {
    error->all(FLERR,
               "Fix dplr does not support unit style lj. Please use other "
               "unit styles like metal or real unit instead. You may set it by "
               "\"units metal\" or \"units real\"");
  }

  int iarg = 3;
  vector<int> map_vec;
  bond_type.clear();
  while (iarg < narg) {
    if (!is_key(arg[iarg])) {
      error->all(FLERR, "Illegal fix command\nwrong number of parameters\n");
    }
    if (string(arg[iarg]) == string("model")) {
      if (iarg + 1 > narg) {
        error->all(FLERR, "Illegal fix adapt command");
      }
      model = string(arg[iarg + 1]);
      iarg += 2;
    } else if (string(arg[iarg]) == string("efield")) {
      if (iarg + 3 > narg) {
        error->all(FLERR,
                   "Illegal fix adapt command, efield should be provided 3 "
                   "float numbers");
      }
      if (utils::strmatch(arg[iarg + 1], "^v_")) {
        xstr = utils::strdup(arg[iarg + 1] + 2);
      } else {
        efield[0] = qe2f * utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        xstyle = CONSTANT;
      }

      if (utils::strmatch(arg[iarg + 2], "^v_")) {
        ystr = utils::strdup(arg[iarg + 2] + 2);
      } else {
        efield[1] = qe2f * utils::numeric(FLERR, arg[iarg + 2], false, lmp);
        ystyle = CONSTANT;
      }

      if (utils::strmatch(arg[iarg + 3], "^v_")) {
        zstr = utils::strdup(arg[iarg + 3] + 2);
      } else {
        efield[2] = qe2f * utils::numeric(FLERR, arg[iarg + 3], false, lmp);
        zstyle = CONSTANT;
      }
      iarg += 4;
    } else if (string(arg[iarg]) == string("type_associate")) {
      int iend = iarg + 1;
      while (iend < narg && (!is_key(arg[iend]))) {
        map_vec.push_back(atoi(arg[iend]) - 1);
        iend++;
      }
      iarg = iend;
    } else if (string(arg[iarg]) == string("bond_type")) {
      int iend = iarg + 1;
      while (iend < narg && (!is_key(arg[iend]))) {
        bond_type.push_back(atoi(arg[iend]) - 1);
        iend++;
      }
      sort(bond_type.begin(), bond_type.end());
      iarg = iend;
    } else {
      break;
    }
  }
  assert(map_vec.size() % 2 == 0 &&
         "number of ints provided by type_associate should be even");

  // dpt.init(model);
  // dtm.init("frozen_model.pb");
  try {
    dpt.init(model, 0, "dipole_charge");
    dtm.init(model, 0, "dipole_charge");
  } catch (deepmd_compat::deepmd_exception &e) {
    error->one(FLERR, e.what());
  }

  pair_deepmd = (PairDeepMD *)force->pair_match("deepmd", 1);
  if (!pair_deepmd) {
    error->all(FLERR, "pair_style deepmd should be set before this fix\n");
  }
  ener_unit_cvt_factor = pair_deepmd->ener_unit_cvt_factor;
  dist_unit_cvt_factor = pair_deepmd->dist_unit_cvt_factor;
  force_unit_cvt_factor = pair_deepmd->force_unit_cvt_factor;

  int n = atom->ntypes;
  std::vector<std::string> type_names = pair_deepmd->type_names;
  std::vector<std::string> type_map;
  std::string type_map_str;
  dpt.get_type_map(type_map_str);
  // convert the string to a vector of strings
  std::istringstream iss(type_map_str);
  std::string type_name;
  while (iss >> type_name) {
    type_map.push_back(type_name);
  }
  if (type_names.size() == 0 || type_map.size() == 0) {
    type_idx_map.resize(n);
    for (int ii = 0; ii < n; ++ii) {
      type_idx_map[ii] = ii;
    }
  } else {
    type_idx_map.clear();
    for (std::string type_name : type_names) {
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
                              " not found in the DPLR model");
      }
    }
    int numb_types = type_idx_map.size();
    if (numb_types < n) {
      type_idx_map.resize(n);
      for (int ii = numb_types; ii < n; ++ii) {
        type_idx_map[ii] = ii;
      }
    }
  }

  for (int ii = 0; ii < map_vec.size() / 2; ++ii) {
    type_asso[type_idx_map[map_vec[ii * 2 + 0]]] =
        type_idx_map[map_vec[ii * 2 + 1]];
    bk_type_asso[type_idx_map[map_vec[ii * 2 + 1]]] =
        type_idx_map[map_vec[ii * 2 + 0]];
  }

  sel_type = dpt.sel_types();
  sort(sel_type.begin(), sel_type.end());
  dpl_type.clear();
  for (int ii = 0; ii < sel_type.size(); ++ii) {
    dpl_type.push_back(type_asso[sel_type[ii]]);
  }

  // set comm size needed by this fix
  comm_reverse = 3;
}

/* ---------------------------------------------------------------------- */

FixDPLR::~FixDPLR() {
  delete[] xstr;
  delete[] ystr;
  delete[] zstr;
}

/* ---------------------------------------------------------------------- */

int FixDPLR::setmask() {
  int mask = 0;
#if LAMMPS_VERSION_NUMBER < 20210210
  // THERMO_ENERGY removed in lammps/lammps#2560
  mask |= THERMO_ENERGY;
#endif
  mask |= POST_INTEGRATE;
  mask |= PRE_FORCE;
  mask |= POST_FORCE;
  mask |= MIN_PRE_EXCHANGE;
  mask |= MIN_PRE_FORCE;
  mask |= MIN_POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixDPLR::init() {
  // double **xx = atom->x;
  // double **vv = atom->v;
  // int nlocal = atom->nlocal;
  // for (int ii = 0; ii < nlocal; ++ii){
  //   cout << xx[ii][0] << " "
  // 	 << xx[ii][1] << " "
  // 	 << xx[ii][2] << "   "
  // 	 << vv[ii][0] << " "
  // 	 << vv[ii][1] << " "
  // 	 << vv[ii][2] << " "
  // 	 << endl;
  // }
  // check variables

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0) {
      error->all(FLERR, "Variable {} for x-field in fix {} does not exist",
                 xstr, style);
    }
    if (input->variable->equalstyle(xvar)) {
      xstyle = EQUAL;
    } else {
      error->all(FLERR, "Variable {} for x-field in fix {} is invalid style",
                 xstr, style);
    }
  }

  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0) {
      error->all(FLERR, "Variable {} for y-field in fix {} does not exist",
                 ystr, style);
    }
    if (input->variable->equalstyle(yvar)) {
      ystyle = EQUAL;
    } else {
      error->all(FLERR, "Variable {} for y-field in fix {} is invalid style",
                 ystr, style);
    }
  }

  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0) {
      error->all(FLERR, "Variable {} for z-field in fix {} does not exist",
                 zstr, style);
    }
    if (input->variable->equalstyle(zvar)) {
      zstyle = EQUAL;
    } else {
      error->all(FLERR, "Variable {} for z-field in fix {} is invalid style",
                 zstr, style);
    }
  }

  if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL) {
    varflag = EQUAL;
  } else {
    varflag = CONSTANT;
  }
}

/* ---------------------------------------------------------------------- */

void FixDPLR::setup_pre_force(int vflag) { pre_force(vflag); }

/* ---------------------------------------------------------------------- */

void FixDPLR::setup(int vflag) {
  // if (strstr(update->integrate_style,"verlet"))
  post_force(vflag);
  // else {
  //   error->all(FLERR, "respa is not supported by this fix");
  // }
}

/* ---------------------------------------------------------------------- */

void FixDPLR::min_setup(int vflag) { setup(vflag); }

/* ---------------------------------------------------------------------- */

void FixDPLR::get_valid_pairs(vector<pair<int, int> > &pairs) {
  pairs.clear();

  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  vector<int> dtype(nall);
  // get type
  int *type = atom->type;
  for (int ii = 0; ii < nall; ++ii) {
    dtype[ii] = type_idx_map[type[ii] - 1];
  }

  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  for (int ii = 0; ii < nbondlist; ++ii) {
    int idx0 = -1, idx1 = -1;
    int bd_type = bondlist[ii][2] - 1;
    if (!binary_search(bond_type.begin(), bond_type.end(), bd_type)) {
      continue;
    }
    std::vector<int>::iterator it =
        find(sel_type.begin(), sel_type.end(), dtype[bondlist[ii][0]]);
    if (it != sel_type.end()) {
      int idx_type = distance(sel_type.begin(), it);
      if (dtype[bondlist[ii][1]] == dpl_type[idx_type]) {
        idx0 = bondlist[ii][0];
        idx1 = bondlist[ii][1];
      } else {
        char str[300];
        sprintf(str,
                "Invalid pair: %d %d \n       A virtual atom of type %d is "
                "expected, but the type of atom %d is "
                "%d.\n       Please check your data file carefully.\n",
                atom->tag[bondlist[ii][0]], atom->tag[bondlist[ii][1]],
                dpl_type[idx_type] + 1, atom->tag[bondlist[ii][1]],
                type[bondlist[ii][1]]);
        error->all(FLERR, str);
      }
    } else {
      it = find(sel_type.begin(), sel_type.end(), dtype[bondlist[ii][1]]);
      if (it != sel_type.end()) {
        int idx_type = distance(sel_type.begin(), it);
        if (dtype[bondlist[ii][0]] == dpl_type[idx_type]) {
          idx0 = bondlist[ii][1];
          idx1 = bondlist[ii][0];
        } else {
          char str[300];
          sprintf(str,
                  "Invalid pair: %d %d \n       A virtual atom of type %d is "
                  "expected, but the type of atom %d is %d.\n       Please "
                  "check your data file carefully.\n",
                  atom->tag[bondlist[ii][0]], atom->tag[bondlist[ii][1]],
                  dpl_type[idx_type] + 1, atom->tag[bondlist[ii][0]],
                  type[bondlist[ii][0]]);
          error->all(FLERR, str);
        }
      } else {
        char str[300];
        sprintf(str,
                "Invalid pair: %d %d \n       They are not expected to have "
                "Wannier centroid.\n       Please check your data file "
                "carefully.\n",
                atom->tag[bondlist[ii][0]], atom->tag[bondlist[ii][1]]);
        error->all(FLERR, str);
      }
    }
    if (!(idx0 < nlocal && idx1 < nlocal)) {
      error->all(FLERR,
                 "find a bonded pair that is not on the same processor, "
                 "something should not happen");
    }
    pairs.push_back(pair<int, int>(idx0, idx1));
  }
}

/* ---------------------------------------------------------------------- */

void FixDPLR::post_integrate() {
  double **x = atom->x;
  double **v = atom->v;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  vector<pair<int, int> > valid_pairs;
  get_valid_pairs(valid_pairs);

  for (int ii = 0; ii < valid_pairs.size(); ++ii) {
    int idx0 = valid_pairs[ii].first;
    int idx1 = valid_pairs[ii].second;
    for (int dd = 0; dd < 3; ++dd) {
      x[idx1][dd] = x[idx0][dd];
      v[idx1][dd] = v[idx0][dd];
      // v[idx1][dd] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixDPLR::pre_force(int vflag) {
  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // if (eflag_atom) {
  //   error->all(FLERR,"atomic energy calculation is not supported by this
  //   fix\n");
  // }

  // declear inputs
  vector<int> dtype(nall);
  vector<FLOAT_PREC> dbox(9, 0);
  vector<FLOAT_PREC> dcoord(nall * 3, 0.);
  // get type
  for (int ii = 0; ii < nall; ++ii) {
    dtype[ii] = type_idx_map[type[ii] - 1];
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
  // get lammps nlist
  NeighList *list = pair_deepmd->list;
  deepmd_compat::InputNlist lmp_list(list->inum, list->ilist, list->numneigh,
                                     list->firstneigh);
  // declear output
  vector<FLOAT_PREC> tensor;
  // compute
  try {
    dpt.compute(tensor, dcoord, dtype, dbox, nghost, lmp_list);
  } catch (deepmd_compat::deepmd_exception &e) {
    error->one(FLERR, e.what());
  }
  // cout << "tensor of size " << tensor.size() << endl;
  // cout << "nghost " << nghost << endl;
  // cout << "nall " << dtype.size() << endl;
  // cout << "nloc " << nlocal << endl;
  // for (int ii = 0; ii < tensor.size(); ++ii){
  //   if (ii%3 == 0){
  //     cout << endl;
  //   }
  //   cout << tensor[ii] << "\t";
  // }
  // cout << endl;
  // for (int ii = 0; ii < nlocal * 3; ++ii){
  //   if (ii%3 == 0){
  //     cout << endl;
  //   }
  //   cout << dcoord[ii] << "\t";
  // }
  // int max_type = 0;
  // for (int ii = 0; ii < dtype.size(); ++ii){
  //   if (dtype[ii] > max_type) {
  //     max_type = dtype[ii];
  //   }
  // }

  vector<int> sel_fwd, sel_bwd;
  int sel_nghost;
  deepmd_compat::select_by_type(sel_fwd, sel_bwd, sel_nghost, dcoord, dtype,
                                nghost, sel_type);
  int sel_nall = sel_bwd.size();
  int sel_nloc = sel_nall - sel_nghost;
  vector<int> sel_type(sel_bwd.size());
  deepmd_compat::select_map<int>(sel_type, dtype, sel_fwd, 1);

  // Yixiao: because the deeptensor already return the correct order, the
  // following map is no longer needed deepmd_compat::AtomMap<FLOAT_PREC>
  // atom_map(sel_type.begin(), sel_type.begin() + sel_nloc); const
  // vector<int> & sort_fwd_map(atom_map.get_fwd_map());

  vector<pair<int, int> > valid_pairs;
  get_valid_pairs(valid_pairs);

  int odim = dpt.output_dim();
  assert(odim == 3);
  dipole_recd.resize(static_cast<size_t>(nall) * 3);
  fill(dipole_recd.begin(), dipole_recd.end(), 0.0);
  for (int ii = 0; ii < valid_pairs.size(); ++ii) {
    int idx0 = valid_pairs[ii].first;
    int idx1 = valid_pairs[ii].second;
    assert(idx0 < sel_fwd.size());  // && sel_fwd[idx0] < sort_fwd_map.size());
    // Yixiao: the sort map is no longer needed
    // int res_idx = sort_fwd_map[sel_fwd[idx0]];
    int res_idx = sel_fwd[idx0];
    // int ret_idx = dpl_bwd[res_idx];
    atom->image[idx1] = atom->image[idx0];
    for (int dd = 0; dd < 3; ++dd) {
      x[idx1][dd] =
          x[idx0][dd] + tensor[res_idx * 3 + dd] * dist_unit_cvt_factor;
      // res_buff[idx1 * odim + dd] = tensor[res_idx * odim + dd];
      dipole_recd[idx0 * 3 + dd] =
          tensor[res_idx * 3 + dd] * dist_unit_cvt_factor;
    }
  }
  // cout << "-------------------- fix/dplr: pre force " << endl;
  // for (int ii = 0; ii < nlocal; ++ii){
  //   cout << ii << "    ";
  //   for (int dd = 0; dd < 3; ++dd){
  //     cout << x[ii][dd] << " " ;
  //   }
  //   cout << endl;
  // }
}

/* ---------------------------------------------------------------------- */

void FixDPLR::post_force(int vflag) {
  if (vflag) {
    v_setup(vflag);
  } else {
    evflag = 0;
  }
  if (vflag_atom) {
    error->all(FLERR,
               "atomic virial calculation is not supported by this fix\n");
  }

  if (!(varflag == CONSTANT)) {
    update_efield_variables();
  }

  PPPMDPLR *pppm_dplr = (PPPMDPLR *)force->kspace_match("pppm/dplr", 1);
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  vector<FLOAT_PREC> dcoord(nall * 3, 0.0), dbox(9, 0.0),
      dfele(nlocal * 3, 0.0);
  vector<int> dtype(nall, 0);
  // set values for dcoord, dbox, dfele
  {
    int *type = atom->type;
    for (int ii = 0; ii < nall; ++ii) {
      dtype[ii] = type_idx_map[type[ii] - 1];
    }
    dbox[0] = domain->h[0] / dist_unit_cvt_factor;  // xx
    dbox[4] = domain->h[1] / dist_unit_cvt_factor;  // yy
    dbox[8] = domain->h[2] / dist_unit_cvt_factor;  // zz
    dbox[7] = domain->h[3] / dist_unit_cvt_factor;  // zy
    dbox[6] = domain->h[4] / dist_unit_cvt_factor;  // zx
    dbox[3] = domain->h[5] / dist_unit_cvt_factor;  // yx
    // get coord
    double **x = atom->x;
    for (int ii = 0; ii < nall; ++ii) {
      for (int dd = 0; dd < 3; ++dd) {
        dcoord[ii * 3 + dd] =
            (x[ii][dd] - domain->boxlo[dd]) / dist_unit_cvt_factor;
      }
    }
    // revise force according to efield
    if (pppm_dplr) {
      const vector<double> &dfele_(pppm_dplr->get_fele());
      assert(dfele_.size() == nlocal * 3);
      for (int ii = 0; ii < nlocal * 3; ++ii) {
        dfele[ii] += dfele_[ii];
      }
    }
    // revise force and virial according to efield
    double *q = atom->q;
    imageint *image = atom->image;
    double unwrap[3];
    double v[6];
    efield_fsum[0] = efield_fsum[1] = efield_fsum[2] = efield_fsum[3] = 0.0;
    efield_force_flag = 0;
    for (int ii = 0; ii < nlocal; ++ii) {
      double tmpf[3];
      for (int dd = 0; dd < 3; ++dd) {
        tmpf[dd] = q[ii] * efield[dd] * force->qe2f;
      }
      for (int dd = 0; dd < 3; ++dd) {
        dfele[ii * 3 + dd] += tmpf[dd];
      }
      domain->unmap(x[ii], image[ii], unwrap);
      efield_fsum[0] -=
          tmpf[0] * unwrap[0] + tmpf[1] * unwrap[1] + tmpf[2] * unwrap[2];
      efield_fsum[1] += tmpf[0];
      efield_fsum[2] += tmpf[1];
      efield_fsum[3] += tmpf[2];
      if (evflag) {
        v[0] = tmpf[0] * unwrap[0];
        v[1] = tmpf[1] * unwrap[1];
        v[2] = tmpf[2] * unwrap[2];
        v[3] = tmpf[0] * unwrap[1];
        v[4] = tmpf[0] * unwrap[2];
        v[5] = tmpf[1] * unwrap[2];
        v_tally(ii, v);
      }
    }
  }
  // lmp nlist
  NeighList *list = pair_deepmd->list;
  deepmd_compat::InputNlist lmp_list(list->inum, list->ilist, list->numneigh,
                                     list->firstneigh);
  // bonded pairs
  vector<pair<int, int> > valid_pairs;
  get_valid_pairs(valid_pairs);
  // output vects
  vector<FLOAT_PREC> dfcorr, dvcorr;
  // compute
  try {
    for (int ii = 0; ii < nlocal * 3; ++ii) {
      dfele[ii] /= force_unit_cvt_factor;
    }
    dtm.compute(dfcorr, dvcorr, dcoord, dtype, dbox, valid_pairs, dfele, nghost,
                lmp_list);
    for (int ii = 0; ii < nlocal * 3; ++ii) {
      dfcorr[ii] *= force_unit_cvt_factor;
    }
    for (int ii = 0; ii < 9; ++ii) {
      dvcorr[ii] *= ener_unit_cvt_factor;
    }
  } catch (deepmd_compat::deepmd_exception &e) {
    error->one(FLERR, e.what());
  }
  assert(dfcorr.size() == dcoord.size());
  assert(dfcorr.size() == nall * 3);
  // backward communication of fcorr
  dfcorr_buff.resize(dfcorr.size());
  copy(dfcorr.begin(), dfcorr.end(), dfcorr_buff.begin());
#if LAMMPS_VERSION_NUMBER >= 20220324
  comm->reverse_comm(this, 3);
#else
  comm->reverse_comm_fix(this, 3);
#endif
  copy(dfcorr_buff.begin(), dfcorr_buff.end(), dfcorr.begin());
  // // check and print
  // cout << "-------------------- fix/dplr: post force " << endl;
  // cout << "dfcorr.size() " << dfcorr.size() << endl;
  // cout << "dcoord.size() " << dcoord.size() << endl;
  // for (int ii = 0; ii < nlocal; ++ii){
  //   cout << ii << "\t x: ";
  //   for (int dd = 0; dd < 3; ++dd){
  //     cout << dcoord[ii*3+dd] << " \t " ;
  //   }
  //   cout << ii << "\t f: ";
  //   for (int dd = 0; dd < 3; ++dd){
  //     cout << dfcorr[ii*3+dd] << " \t " ;
  //   }
  //   cout << endl;
  // }
  // apply the force correction
  double **f = atom->f;
  for (int ii = 0; ii < nlocal; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      f[ii][dd] += dfcorr[ii * 3 + dd];
    }
  }
  // cout << "virial corr1 ";
  // for (int ii = 0; ii < 9; ++ii){
  //   cout << dvcorr[ii] << " " ;
  // }
  // cout << endl;
  for (int ii = 0; ii < valid_pairs.size(); ++ii) {
    int idx0 = valid_pairs[ii].first;
    int idx1 = valid_pairs[ii].second;
    for (int dd0 = 0; dd0 < 3; ++dd0) {
      for (int dd1 = 0; dd1 < 3; ++dd1) {
        dvcorr[dd0 * 3 + dd1] -=
            dfele[idx1 * 3 + dd0] * dipole_recd[idx0 * 3 + dd1];
      }
    }
  }
  // cout << "virial corr2 ";
  // for (int ii = 0; ii < 9; ++ii){
  //   cout << dvcorr[ii] << " " ;
  // }
  // cout << endl;
  if (evflag) {
    double vv[6] = {0.0};
    vv[0] += dvcorr[0];
    vv[1] += dvcorr[4];
    vv[2] += dvcorr[8];
    vv[3] += dvcorr[3];
    vv[4] += dvcorr[6];
    vv[5] += dvcorr[7];
    v_tally(0, vv);
  }
}

/* ---------------------------------------------------------------------- */

void FixDPLR::min_pre_exchange() { post_integrate(); }

/* ---------------------------------------------------------------------- */

void FixDPLR::min_pre_force(int vflag) { pre_force(vflag); }

/* ---------------------------------------------------------------------- */

void FixDPLR::min_post_force(int vflag) { post_force(vflag); }

/* ---------------------------------------------------------------------- */

int FixDPLR::pack_reverse_comm(int n, int first, double *buf) {
  int m = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    buf[m++] = dfcorr_buff[3 * i + 0];
    buf[m++] = dfcorr_buff[3 * i + 1];
    buf[m++] = dfcorr_buff[3 * i + 2];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixDPLR::unpack_reverse_comm(int n, int *list, double *buf) {
  int m = 0;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    dfcorr_buff[3 * j + 0] += buf[m++];
    dfcorr_buff[3 * j + 1] += buf[m++];
    dfcorr_buff[3 * j + 2] += buf[m++];
  }
}

/* ----------------------------------------------------------------------
   return energy added by fix
------------------------------------------------------------------------- */

double FixDPLR::compute_scalar(void) {
  if (efield_force_flag == 0) {
    MPI_Allreduce(&efield_fsum[0], &efield_fsum_all[0], 4, MPI_DOUBLE, MPI_SUM,
                  world);
    efield_force_flag = 1;
  }
  return efield_fsum_all[0];
}

/* ----------------------------------------------------------------------
   return total extra force due to fix
------------------------------------------------------------------------- */

double FixDPLR::compute_vector(int n) {
  if (efield_force_flag == 0) {
    MPI_Allreduce(&efield_fsum[0], &efield_fsum_all[0], 4, MPI_DOUBLE, MPI_SUM,
                  world);
    efield_force_flag = 1;
  }
  return efield_fsum_all[n + 1];
}

/* ----------------------------------------------------------------------
   update efield variables without doing anything else
------------------------------------------------------------------------- */

void FixDPLR::update_efield_variables() {
  modify->clearstep_compute();

  if (xstyle == EQUAL) {
    efield[0] = qe2f * input->variable->compute_equal(xvar);
  }
  if (ystyle == EQUAL) {
    efield[1] = qe2f * input->variable->compute_equal(yvar);
  }
  if (zstyle == EQUAL) {
    efield[2] = qe2f * input->variable->compute_equal(zvar);
  }

  modify->addstep_compute(update->ntimestep + 1);
}
