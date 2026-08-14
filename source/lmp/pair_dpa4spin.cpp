// SPDX-License-Identifier: LGPL-3.0-or-later
#include "pair_dpa4spin.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "atom.h"
#include "citeme.h"
#include "comm.h"
#include "deepmd_version.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;

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

namespace {
// Reduced Planck constant in eV.ps. The model reports the magnetic force as
// the energy gradient with respect to the magnetic moment, while LAMMPS stores
// the precession force; the two differ by the factor hbar / |m|.
constexpr double kHBar = 6.5821191e-04;
// Positions of the LAMMPS global virial components (xx, yy, zz, xy, xz, yz)
// within the nine-component tensor the model reports.
constexpr int kGlobalVirialMap[6] = {0, 4, 8, 3, 6, 7};
// Positions of the LAMMPS centroid per-atom virial components
// (xx, yy, zz, xy, xz, yz, yx, zx, zy) within the same tensor.
constexpr int kCentroidVirialMap[9] = {0, 4, 8, 3, 6, 7, 1, 2, 5};
}  // namespace

PairDPA4Spin::PairDPA4Spin(LAMMPS* lmp)
    : Pair(lmp), scale(nullptr), cutoff(0.0), commdata_(nullptr) {
  if (lmp->citeme) {
    lmp->citeme->add(cite_user_deepmd_package);
  }
  if (strcmp(update->unit_style, "lj") == 0) {
    error->all(FLERR,
               "pair style dpa4spin does not support unit style lj; use a "
               "physical unit style such as metal or real.");
  }
  ener_unit_cvt_factor = force->boltz / 8.617343e-5;
  dist_unit_cvt_factor = force->angstrom;
  force_unit_cvt_factor = ener_unit_cvt_factor / dist_unit_cvt_factor;

  // The artifact is identified by a path that a restart cannot carry, so the
  // input has to re-issue pair_style and pair_coeff after read_restart.
  restartinfo = 0;
  // The model reports a nine-component atomic virial, which feeds compute
  // centroid/stress/atom.
  centroidstressflag = CENTROID_AVAIL;
  respa_enable = 0;
  writedata = 0;

  print_summary("  ");
}

PairDPA4Spin::~PairDPA4Spin() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

void PairDPA4Spin::print_summary(const std::string& pre) const {
  if (comm->me != 0) {
    return;
  }
  // The DeePMD-kit banner is written to std::cout by the library. Capture it
  // so that the whole summary reaches the LAMMPS screen and log file together.
  std::stringstream buffer;
  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());

  std::cout << "Summary of lammps deepmd module ..." << std::endl;
  std::cout << pre << ">>> Info of deepmd-kit:" << std::endl;
  deep_spin.print_summary(pre);
  std::cout << pre << ">>> Info of lammps module:" << std::endl;
  std::cout << pre << "use deepmd-kit at:  " << STR_DEEPMD_ROOT << std::endl;
  std::cout << pre << "source:             " << STR_GIT_SUMM << std::endl;
  std::cout << pre << "source branch:      " << STR_GIT_BRANCH << std::endl;
  std::cout << pre << "source commit:      " << STR_GIT_HASH << std::endl;
  std::cout << pre << "source commit at:   " << STR_GIT_DATE << std::endl;
  std::cout << pre << "build with inc:     " << STR_BACKEND_INCLUDE_DIRS
            << std::endl;
  std::cout << pre << "build with lib:     " << STR_BACKEND_LIBRARY_PATH
            << std::endl;

  std::cout.rdbuf(sbuf);
  utils::logmesg(lmp, buffer.str());
}

int PairDPA4Spin::get_node_rank() const {
  int rank = 0;
  MPI_Comm_rank(world, &rank);

#ifdef MPI_COMM_TYPE_SHARED
  // LAMMPS may run on a partition or on an embedding-provided subcommunicator.
  // Splitting that communicator by shared-memory domain keeps independent
  // LAMMPS instances out of collectives on MPI_COMM_WORLD.
  MPI_Comm node_comm;
  MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &node_comm);
  int node_rank = 0;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_free(&node_comm);
  return node_rank;
#else
  // The serial MPI stubs of LAMMPS predate MPI-3 and provide no
  // MPI_Comm_split_type. Their only communicator holds a single rank, so its
  // communicator rank is also the node-local rank.
  return rank;
#endif
}

void PairDPA4Spin::allocate() {
  allocated = 1;
  const int ntypes = atom->ntypes;

  memory->create(setflag, ntypes + 1, ntypes + 1, "pair:setflag");
  memory->create(cutsq, ntypes + 1, ntypes + 1, "pair:cutsq");
  memory->create(scale, ntypes + 1, ntypes + 1, "pair:scale");

  for (int ii = 1; ii <= ntypes; ++ii) {
    for (int jj = ii; jj <= ntypes; ++jj) {
      setflag[ii][jj] = 0;
      scale[ii][jj] = 0.0;
    }
  }
}

void PairDPA4Spin::settings(int narg, char** arg) {
  // Name whichever style the input selected, so the Kokkos variant reports
  // itself rather than its host base.
  const std::string style = force->pair_style;
  if (narg < 1) {
    error->all(FLERR, "Illegal pair_style command: pair style " + style +
                          " evaluates a single native-spin artifact and takes "
                          "its path as the first argument.");
  }

  try {
    deep_spin.init(arg[0], get_node_rank());
  } catch (deepmd_compat::deepmd_exception& e) {
    error->one(FLERR, e.what());
  }
  cutoff = deep_spin.cutoff() * dist_unit_cvt_factor;

  // How many values name a charge state is a property of the artifact, so the
  // keyword can only be read once the model is loaded.
  const int dim_chg_spin = deep_spin.dim_chg_spin();
  std::vector<double> charge_spin;
  int iarg = 1;
  while (iarg < narg) {
    if (std::string(arg[iarg]) != "charge_spin") {
      error->all(FLERR, "Illegal pair_style command: pair style " + style +
                            " takes the artifact path followed by the optional "
                            "keyword charge_spin, not '" +
                            std::string(arg[iarg]) + "'.");
    }
    // One charge state holds for the whole run, so a second occurrence names
    // no state the style could serve.  Values already read mark the keyword
    // as seen: a successful read always contributes at least one value.
    if (!charge_spin.empty()) {
      error->all(FLERR,
                 "Illegal pair_style command: keyword charge_spin names the "
                 "single charge state of the whole run, so it may appear "
                 "only once.");
    }
    if (dim_chg_spin == 0) {
      error->all(FLERR,
                 "Illegal pair_style command: the artifact served by pair "
                 "style " +
                     style +
                     " carries no charge/spin condition, so keyword "
                     "charge_spin names nothing it can serve.");
    }
    if (iarg + dim_chg_spin >= narg) {
      error->all(FLERR,
                 "Illegal pair_style command: keyword charge_spin names a "
                 "charge state with " +
                     std::to_string(dim_chg_spin) + " value(s).");
    }
    for (int ii = 0; ii < dim_chg_spin; ++ii) {
      charge_spin.push_back(
          utils::numeric(FLERR, arg[iarg + 1 + ii], false, lmp));
    }
    iarg += 1 + dim_chg_spin;
  }

  // A charge/spin condition named on the pair_style line holds for the whole
  // run, so it is handed to the model once here instead of being resupplied
  // every step.  This is also what lets a compressed model serve it at all:
  // there the condition lives inside frozen tables, which are rebuilt here
  // and cannot be rebuilt per step at a sensible cost.
  if (!charge_spin.empty()) {
    try {
      deep_spin.set_charge_spin(charge_spin);
    } catch (deepmd_compat::deepmd_exception& e) {
      error->one(FLERR, e.what());
    }
  }

  utils::logmesg(lmp,
                 "  >>> Info of model(s):\n"
                 "  using 1 model(s): {}\n"
                 "  rcut in model:      {}\n"
                 "  ntypes in model:    {}\n",
                 arg[0], cutoff, deep_spin.numb_types());
  if (!charge_spin.empty()) {
    std::string values;
    for (const double value : charge_spin) {
      values += fmt::format("{}  ", value);
    }
    utils::logmesg(lmp, "  using charge_spin:  {}\n", values);
  }
}

/* ----------------------------------------------------------------------
   map the atom types onto the elements of the model
------------------------------------------------------------------------- */

void PairDPA4Spin::coeff(int narg, char** arg) {
  if (narg < 2) {
    error->all(FLERR, "Incorrect args for pair coefficients");
  }
  if (!allocated) {
    allocate();
  }

  const int ntypes = atom->ntypes;
  int ilo, ihi, jlo, jhi;
  utils::bounds(FLERR, arg[0], 1, ntypes, ilo, ihi, error);
  utils::bounds(FLERR, arg[1], 1, ntypes, jlo, jhi, error);
  if (ilo != 1 || jlo != 1 || ihi != ntypes || jhi != ntypes) {
    error->all(FLERR,
               "pair style dpa4spin sets one scale for every atom type, i.e. "
               "pair_coeff * *.");
  }

  // Element names the artifact was trained on, in model type order.
  std::vector<std::string> model_types;
  std::string type_map_str;
  deep_spin.get_type_map(type_map_str);
  std::istringstream type_map_stream(type_map_str);
  std::string element;
  while (type_map_stream >> element) {
    model_types.push_back(element);
  }
  const int model_ntypes = static_cast<int>(model_types.size());

  type_idx_map.assign(ntypes, -1);
  if (narg == 2) {
    // A bare `pair_coeff * *` maps the atom types onto the leading model
    // elements by position, which is only meaningful when the model has at
    // least as many elements as the system has atom types.
    if (model_ntypes < ntypes) {
      error->all(FLERR,
                 "pair_coeff * * maps atom type i onto model element i, but "
                 "the system has " +
                     std::to_string(ntypes) +
                     " atom types and the model only " +
                     std::to_string(model_ntypes) +
                     "; list the elements explicitly, e.g. pair_coeff * * Fe "
                     "C.");
    }
    if (model_ntypes > ntypes) {
      error->warning(
          FLERR, "pair_coeff * * maps the system atom types onto the first " +
                     std::to_string(ntypes) + " of the model's " +
                     std::to_string(model_ntypes) +
                     " element types; list the elements explicitly, e.g. "
                     "pair_coeff * * Fe C, to avoid a possible mislabeling.");
    }
    for (int ii = 0; ii < ntypes; ++ii) {
      type_idx_map[ii] = ii;
    }
  } else {
    // An explicit element list names the model element of each atom type in
    // turn. NULL, and any atom type past the end of the list, denotes a type
    // the model never sees.
    if (narg - 2 > ntypes) {
      error->all(FLERR,
                 "pair_coeff lists more elements than the system has atom "
                 "types.");
    }
    for (int ii = 0; ii + 2 < narg; ++ii) {
      const std::string name = arg[ii + 2];
      if (name == "NULL") {
        continue;
      }
      const auto found =
          std::find(model_types.begin(), model_types.end(), name);
      if (found == model_types.end()) {
        error->all(FLERR,
                   "pair_coeff: element " + name + " not found in the model");
      }
      type_idx_map[ii] = static_cast<int>(found - model_types.begin());
    }
  }

  std::string excluded;
  for (int ii = 0; ii < ntypes; ++ii) {
    if (type_idx_map[ii] < 0) {
      excluded += " " + std::to_string(ii + 1);
    }
  }
  if (!excluded.empty()) {
    error->warning(FLERR, "pair style dpa4spin ignores atom type(s)" +
                              excluded + ": they map to no model element.");
  }

  for (int ii = 1; ii <= ntypes; ++ii) {
    for (int jj = ii; jj <= ntypes; ++jj) {
      setflag[ii][jj] = 1;
      scale[ii][jj] = 1.0;
    }
  }
}

void PairDPA4Spin::init_style() {
  neighbor->add_request(this, NeighConst::REQ_FULL);

  const std::string style = force->pair_style;
  if (!atom->sp_flag) {
    error->all(FLERR, "pair style " + style +
                          " only supports spin atoms, please use pair style "
                          "deepmd instead.");
  }
  // The scheme, not the lower schema, decides which style serves an artifact:
  // this one marshals one node per atom and reads the magnetic force straight
  // off the model, neither of which a virtual-atom model provides.
  if (!deep_spin.uses_native_spin_scheme()) {
    error->all(FLERR,
               "pair style " + style +
                   " serves the native-spin scheme without cross-rank message "
                   "passing; a virtual-atom spin model, and a native-spin "
                   "model whose descriptor exchanges intermediate features "
                   "between ranks, are served by pair style deepspin.");
  }
  // A single rank folds ghost neighbours onto the local atom that owns them,
  // which is resolved through the atom map. Domain decomposition gives every
  // ghost its own node and needs no map.
  if (comm->nprocs == 1 && atom->map_style == Atom::MAP_NONE) {
    error->all(FLERR, "pair style " + style +
                          " needs an atom map on a single rank; add "
                          "'atom_modify map yes' to the input.");
  }
}

double PairDPA4Spin::init_one(int i, int j) {
  if (setflag[i][j] == 0) {
    scale[i][j] = 1.0;
  }
  scale[j][i] = scale[i][j];

  return cutoff;
}

void PairDPA4Spin::compute(int eflag, int vflag) {
  ev_init(eflag, vflag);
  if (vflag_atom) {
    error->all(FLERR,
               "6-element atomic virial is not supported. Use compute "
               "centroid/stress/atom command for 9-element atomic virial.");
  }

  const int nlocal = atom->nlocal;
  const int nghost = atom->nghost;
  const int nall = nlocal + nghost;
  if (nall == 0) {
    // A rank holding no atom at all contributes nothing to any output.
    return;
  }

  double** x = atom->x;
  double** f = atom->f;
  double** sp = atom->sp;
  double** fm = atom->fm;
  const int* type = atom->type;

  // === Step 1. Marshal the model inputs ===
  // Coordinates are referred to the box origin, and the moment is the LAMMPS
  // unit direction scaled by its magnitude.
  std::vector<double> dcoord(static_cast<size_t>(nall) * 3);
  std::vector<double> dspin(static_cast<size_t>(nall) * 3);
  std::vector<int> dtype(nall);
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dcoord[ii * 3 + dd] =
          (x[ii][dd] - domain->boxlo[dd]) / dist_unit_cvt_factor;
      dspin[ii * 3 + dd] = sp[ii][dd] * sp[ii][3];
    }
    dtype[ii] = type_idx_map[type[ii] - 1];
  }

  std::vector<double> dbox(9, 0.0);
  dbox[0] = domain->h[0] / dist_unit_cvt_factor;  // xx
  dbox[4] = domain->h[1] / dist_unit_cvt_factor;  // yy
  dbox[8] = domain->h[2] / dist_unit_cvt_factor;  // zz
  dbox[7] = domain->h[3] / dist_unit_cvt_factor;  // zy
  dbox[6] = domain->h[4] / dist_unit_cvt_factor;  // zx
  dbox[3] = domain->h[5] / dist_unit_cvt_factor;  // yx

  commdata_ = (CommBrickDPA4Spin*)comm;
  deepmd_compat::InputNlist lmp_list(
      list->inum, list->ilist, list->numneigh, list->firstneigh,
      commdata_->nswap, commdata_->sendnum, commdata_->recvnum,
      commdata_->firstrecv, commdata_->sendlist, commdata_->sendproc,
      commdata_->recvproc, &world, comm->nprocs);
  lmp_list.set_mask(NEIGHMASK);
  // A single rank folds every ghost onto the local atom that owns it, an
  // ownership the atom map resolves. Domain decomposition gives each ghost a
  // node of its own and needs no such map.
  std::vector<int> mapping;
  if (comm->nprocs == 1) {
    mapping.resize(nall);
    for (int ii = 0; ii < nall; ++ii) {
      mapping[ii] = atom->map(atom->tag[ii]);
    }
    lmp_list.set_mapping(mapping.data());
  }

  // === Step 2. Evaluate the model ===
  // LAMMPS resets ago to zero on every neighbor-list rebuild, which is when
  // the backend refreshes its cached topology.
  double dener = 0.0;
  std::vector<double> dforce(static_cast<size_t>(nall) * 3);
  std::vector<double> dforce_mag(static_cast<size_t>(nall) * 3);
  std::vector<double> dvirial(9, 0.0);
  std::vector<double> deatom, dvatom;
  try {
    if (eflag_atom || cvflag_atom) {
      deep_spin.compute(dener, dforce, dforce_mag, dvirial, deatom, dvatom,
                        dcoord, dspin, dtype, dbox, nghost, lmp_list,
                        neighbor->ago);
    } else {
      deep_spin.compute(dener, dforce, dforce_mag, dvirial, dcoord, dspin,
                        dtype, dbox, nghost, lmp_list, neighbor->ago);
    }
  } catch (deepmd_compat::deepmd_exception& e) {
    error->one(FLERR, e.what());
  }

  // === Step 3. Accumulate the model outputs ===
  // The model reports a force and a magnetic force for every atom the neighbor
  // list covers, ghosts included; the spin atom style folds the ghost rows
  // onto their owners in its reverse communication. Dividing the magnetic
  // force by hbar / |m| turns the energy gradient with respect to the moment
  // into the precession force LAMMPS stores. Multiplication by |m| preserves
  // that relation while making a zero moment well-defined.
  const double force_scale = scale[1][1] * force_unit_cvt_factor;
  const double magnetic_force_scale = force_scale / kHBar;
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      f[ii][dd] += dforce[3 * ii + dd] * force_scale;
      fm[ii][dd] += dforce_mag[3 * ii + dd] * sp[ii][3] * magnetic_force_scale;
    }
  }

  if (eflag) {
    eng_vdwl += scale[1][1] * dener * ener_unit_cvt_factor;
  }
  if (vflag) {
    for (int kk = 0; kk < 6; ++kk) {
      virial[kk] +=
          scale[1][1] * dvirial[kGlobalVirialMap[kk]] * ener_unit_cvt_factor;
    }
  }
  if (eflag_atom) {
    for (int ii = 0; ii < nlocal; ++ii) {
      eatom[ii] += scale[1][1] * deatom[ii] * ener_unit_cvt_factor;
    }
  }
  if (cvflag_atom) {
    for (int ii = 0; ii < nall; ++ii) {
      for (int kk = 0; kk < 9; ++kk) {
        cvatom[ii][kk] += scale[1][1] *
                          dvatom[9 * ii + kCentroidVirialMap[kk]] *
                          ener_unit_cvt_factor;
      }
    }
  }
}

void* PairDPA4Spin::extract(const char* str, int& dim) {
  if (strcmp(str, "scale") == 0) {
    dim = 2;
    return (void*)scale;
  }
  return nullptr;
}
