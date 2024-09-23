// SPDX-License-Identifier: LGPL-3.0-or-later
#include "common.h"

#include <fcntl.h>

#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "AtomMap.h"
#include "device.h"
#include <numeric>
#if defined(_WIN32)
#if defined(_WIN32_WINNT)
#undef _WIN32_WINNT
#endif

// target Windows version is windows 7 and later
#define _WIN32_WINNT _WIN32_WINNT_WIN7
#define PSAPI_VERSION 2
#include <io.h>
#include <windows.h>
#define O_RDONLY _O_RDONLY
#else
// not windows
#include <dlfcn.h>
#endif
#ifdef BUILD_TENSORFLOW
#include "commonTF.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

using namespace tensorflow;
#endif

#ifdef BUILD_PADDLE
#include "commonPD.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#endif

static std::vector<std::string> split(const std::string& input_,
                                      const std::string& delimiter) {
  std::string input = input_;
  size_t pos = 0;
  std::vector<std::string> res;
  while ((pos = input.find(delimiter)) != std::string::npos) {
    res.push_back(input.substr(0, pos));
    input.erase(0, pos + delimiter.length());
  }
  res.push_back(input);
  return res;
}

bool deepmd::model_compatable(std::string& model_version) {
  std::vector<std::string> words_mv = split(model_version, ".");
  std::vector<std::string> words_gmv = split(global_model_version, ".");
  if (words_mv.size() != 2) {
    throw deepmd::deepmd_exception("invalid graph model version string " +
                                   model_version);
  }
  if (words_gmv.size() != 2) {
    throw deepmd::deepmd_exception("invalid supported model version string " +
                                   global_model_version);
  }
  int model_version_major = atoi(words_mv[0].c_str());
  int model_version_minor = atoi(words_mv[1].c_str());
  int MODEL_VERSION_MAJOR = atoi(words_gmv[0].c_str());
  int MODEL_VERSION_MINOR = atoi(words_gmv[1].c_str());
  if (model_version_major != MODEL_VERSION_MAJOR ||
      model_version_minor > MODEL_VERSION_MINOR) {
    return false;
  } else {
    return true;
  }
}

template <typename VALUETYPE>
void deepmd::select_by_type(std::vector<int>& fwd_map,
                            std::vector<int>& bkw_map,
                            int& nghost_real,
                            const std::vector<VALUETYPE>& dcoord_,
                            const std::vector<int>& datype_,
                            const int& nghost,
                            const std::vector<int>& sel_type_) {
  std::vector<int> sel_type(sel_type_);
  sort(sel_type.begin(), sel_type.end());
  int nall = datype_.size();
  int nloc = nall - nghost;
  int nloc_real = 0;
  nghost_real = 0;
  fwd_map.resize(nall);
  bkw_map.clear();
  bkw_map.reserve(nall);
  int cc = 0;
  for (int ii = 0; ii < nall; ++ii) {
    // exclude virtual sites
    // select the type with id < ntypes
    if (binary_search(sel_type.begin(), sel_type.end(), datype_[ii])) {
      bkw_map.push_back(ii);
      if (ii < nloc) {
        nloc_real += 1;
      } else {
        nghost_real += 1;
      }
      fwd_map[ii] = cc;
      cc++;
    } else {
      fwd_map[ii] = -1;
    }
  }
  assert((nloc_real + nghost_real) == bkw_map.size());
}

template void deepmd::select_by_type<double>(std::vector<int>& fwd_map,
                                             std::vector<int>& bkw_map,
                                             int& nghost_real,
                                             const std::vector<double>& dcoord_,
                                             const std::vector<int>& datype_,
                                             const int& nghost,
                                             const std::vector<int>& sel_type_);

template void deepmd::select_by_type<float>(std::vector<int>& fwd_map,
                                            std::vector<int>& bkw_map,
                                            int& nghost_real,
                                            const std::vector<float>& dcoord_,
                                            const std::vector<int>& datype_,
                                            const int& nghost,
                                            const std::vector<int>& sel_type_);

template <typename VALUETYPE>
void deepmd::select_real_atoms(std::vector<int>& fwd_map,
                               std::vector<int>& bkw_map,
                               int& nghost_real,
                               const std::vector<VALUETYPE>& dcoord_,
                               const std::vector<int>& datype_,
                               const int& nghost,
                               const int& ntypes) {
  std::vector<int> sel_type;
  for (int ii = 0; ii < ntypes; ++ii) {
    sel_type.push_back(ii);
  }
  deepmd::select_by_type(fwd_map, bkw_map, nghost_real, dcoord_, datype_,
                         nghost, sel_type);
}

template void deepmd::select_real_atoms<double>(
    std::vector<int>& fwd_map,
    std::vector<int>& bkw_map,
    int& nghost_real,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const int& nghost,
    const int& ntypes);

template void deepmd::select_real_atoms<float>(
    std::vector<int>& fwd_map,
    std::vector<int>& bkw_map,
    int& nghost_real,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const int& nghost,
    const int& ntypes);

template <typename VALUETYPE>
void deepmd::select_real_atoms_coord(std::vector<VALUETYPE>& dcoord,
                                     std::vector<int>& datype,
                                     std::vector<VALUETYPE>& aparam,
                                     int& nghost_real,
                                     std::vector<int>& fwd_map,
                                     std::vector<int>& bkw_map,
                                     int& nall_real,
                                     int& nloc_real,
                                     const std::vector<VALUETYPE>& dcoord_,
                                     const std::vector<int>& datype_,
                                     const std::vector<VALUETYPE>& aparam_,
                                     const int& nghost,
                                     const int& ntypes,
                                     const int& nframes,
                                     const int& daparam,
                                     const int& nall,
                                     const bool aparam_nall) {
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost,
                    ntypes);
  // resize to nall_real
  nall_real = bkw_map.size();
  nloc_real = nall_real - nghost_real;
  dcoord.resize(static_cast<size_t>(nframes) * nall_real * 3);
  datype.resize(nall_real);
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3, nframes, nall_real, nall);
  select_map<int>(datype, datype_, fwd_map, 1);
  // aparam
  if (daparam > 0) {
    aparam.resize(static_cast<size_t>(nframes) *
                  (aparam_nall ? nall_real : nloc_real));
    select_map<VALUETYPE>(aparam, aparam_, fwd_map, daparam, nframes,
                          (aparam_nall ? nall_real : nloc_real),
                          (aparam_nall ? nall : (nall - nghost)));
  }
}

template void deepmd::select_real_atoms_coord<double>(
    std::vector<double>& dcoord,
    std::vector<int>& datype,
    std::vector<double>& aparam,
    int& nghost_real,
    std::vector<int>& fwd_map,
    std::vector<int>& bkw_map,
    int& nall_real,
    int& nloc_real,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& aparam_,
    const int& nghost,
    const int& ntypes,
    const int& nframes,
    const int& daparam,
    const int& nall,
    const bool aparam_nall);

template void deepmd::select_real_atoms_coord<float>(
    std::vector<float>& dcoord,
    std::vector<int>& datype,
    std::vector<float>& aparam,
    int& nghost_real,
    std::vector<int>& fwd_map,
    std::vector<int>& bkw_map,
    int& nall_real,
    int& nloc_real,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& aparam_,
    const int& nghost,
    const int& ntypes,
    const int& nframes,
    const int& daparam,
    const int& nall,
    const bool aparam_nall);

void deepmd::NeighborListData::copy_from_nlist(const InputNlist& inlist) {
  int inum = inlist.inum;
  ilist.resize(inum);
  jlist.resize(inum);
  memcpy(&ilist[0], inlist.ilist, inum * sizeof(int));
  for (int ii = 0; ii < inum; ++ii) {
    int jnum = inlist.numneigh[ii];
    jlist[ii].resize(jnum);
    memcpy(&jlist[ii][0], inlist.firstneigh[ii], jnum * sizeof(int));
  }
}

void deepmd::NeighborListData::shuffle(const AtomMap& map) {
  const std::vector<int>& fwd_map = map.get_fwd_map();
  shuffle(fwd_map);
}

void deepmd::NeighborListData::shuffle(const std::vector<int>& fwd_map) {
  int nloc = fwd_map.size();
  for (unsigned ii = 0; ii < ilist.size(); ++ii) {
    if (ilist[ii] < nloc) {
      ilist[ii] = fwd_map[ilist[ii]];
    }
  }
  for (unsigned ii = 0; ii < jlist.size(); ++ii) {
    for (unsigned jj = 0; jj < jlist[ii].size(); ++jj) {
      if (jlist[ii][jj] < nloc) {
        jlist[ii][jj] = fwd_map[jlist[ii][jj]];
      }
    }
  }
}

void deepmd::NeighborListData::shuffle_exclude_empty(
    const std::vector<int>& fwd_map) {
  shuffle(fwd_map);
  std::vector<int> new_ilist;
  std::vector<std::vector<int>> new_jlist;
  new_ilist.reserve(ilist.size());
  new_jlist.reserve(jlist.size());
  for (int ii = 0; ii < ilist.size(); ++ii) {
    if (ilist[ii] >= 0) {
      new_ilist.push_back(ilist[ii]);
    }
  }
  int new_inum = new_ilist.size();
  for (int ii = 0; ii < jlist.size(); ++ii) {
    if (ilist[ii] >= 0) {
      std::vector<int> tmp_jlist;
      tmp_jlist.reserve(jlist[ii].size());
      for (int jj = 0; jj < jlist[ii].size(); ++jj) {
        if (jlist[ii][jj] >= 0) {
          tmp_jlist.push_back(jlist[ii][jj]);
        }
      }
      new_jlist.push_back(tmp_jlist);
    }
  }
  ilist = new_ilist;
  jlist = new_jlist;
}
void deepmd::NeighborListData::padding() {
  size_t max_length = 0;
  for (const auto& row : jlist) {
    max_length = std::max(max_length, row.size());
  }

  for (int i = 0; i < jlist.size(); i++) {
    jlist[i].resize(max_length, -1);
  }
}

void deepmd::NeighborListData::make_inlist(InputNlist& inlist) {
  int nloc = ilist.size();
  numneigh.resize(nloc);
  firstneigh.resize(nloc);
  for (int ii = 0; ii < nloc; ++ii) {
    numneigh[ii] = jlist[ii].size();
    firstneigh[ii] = &jlist[ii][0];
  }
  inlist.inum = nloc;
  inlist.ilist = &ilist[0];
  inlist.numneigh = &numneigh[0];
  inlist.firstneigh = &firstneigh[0];
}

#ifdef BUILD_TENSORFLOW
void deepmd::check_status(const tensorflow::Status& status) {
  if (!status.ok()) {
    std::cout << status.ToString() << std::endl;
    throw deepmd::tf_exception(status.ToString());
  }
}
#endif

void throw_env_not_set_warning(std::string env_name) {
  std::cerr << "DeePMD-kit WARNING: Environmental variable " << env_name
            << " is not set. "
            << "Tune " << env_name << " for the best performance. "
            << "See https://deepmd.rtfd.io/parallelism/ for more information."
            << std::endl;
}

void deepmd::get_env_nthreads(int& num_intra_nthreads,
                              int& num_inter_nthreads) {
  num_intra_nthreads = 0;
  num_inter_nthreads = 0;
  const char* env_intra_nthreads =
      std::getenv("DP_INTRA_OP_PARALLELISM_THREADS");
  const char* env_inter_nthreads =
      std::getenv("DP_INTER_OP_PARALLELISM_THREADS");
  // backward compatibility
  const char* env_intra_nthreads_tf =
      std::getenv("TF_INTRA_OP_PARALLELISM_THREADS");
  const char* env_inter_nthreads_tf =
      std::getenv("TF_INTER_OP_PARALLELISM_THREADS");
  const char* env_omp_nthreads = std::getenv("OMP_NUM_THREADS");
  if (env_intra_nthreads &&
      std::string(env_intra_nthreads) != std::string("") &&
      atoi(env_intra_nthreads) >= 0) {
    num_intra_nthreads = atoi(env_intra_nthreads);
  } else if (env_intra_nthreads_tf &&
             std::string(env_intra_nthreads_tf) != std::string("") &&
             atoi(env_intra_nthreads_tf) >= 0) {
    num_intra_nthreads = atoi(env_intra_nthreads_tf);
  } else {
    throw_env_not_set_warning("DP_INTRA_OP_PARALLELISM_THREADS");
  }
  if (env_inter_nthreads &&
      std::string(env_inter_nthreads) != std::string("") &&
      atoi(env_inter_nthreads) >= 0) {
    num_inter_nthreads = atoi(env_inter_nthreads);
  } else if (env_inter_nthreads_tf &&
             std::string(env_inter_nthreads_tf) != std::string("") &&
             atoi(env_inter_nthreads_tf) >= 0) {
    num_inter_nthreads = atoi(env_inter_nthreads_tf);
  } else {
    throw_env_not_set_warning("DP_INTER_OP_PARALLELISM_THREADS");
  }
  if (!(env_omp_nthreads && std::string(env_omp_nthreads) != std::string("") &&
        atoi(env_omp_nthreads) >= 0)) {
    throw_env_not_set_warning("OMP_NUM_THREADS");
  }
}

static inline void _load_library_path(std::string dso_path) {
#if defined(_WIN32)
  void* dso_handle = LoadLibrary(dso_path.c_str());
#else
  void* dso_handle = dlopen(dso_path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
  if (!dso_handle) {
    throw deepmd::deepmd_exception(
        dso_path +
        " is not found! You can add the library directory to LD_LIBRARY_PATH");
  }
}

static inline void _load_single_op_library(std::string library_name) {
#if defined(_WIN32)
  std::string dso_path = library_name + ".dll";
#else
  std::string dso_path = "lib" + library_name + ".so";
#endif
  _load_library_path(dso_path);
}

void deepmd::load_op_library() {
#ifdef BUILD_TENSORFLOW
  _load_single_op_library("deepmd_op");
#endif
#ifdef BUILD_PYTORCH
  _load_single_op_library("deepmd_op_pt");
#endif
// #ifdef BUILD_PADDLE
//   _load_single_op_library("deepmd_op_pd");
// #endif
  // load customized plugins
  const char* env_customized_plugins = std::getenv("DP_PLUGIN_PATH");
  if (env_customized_plugins) {
#if !defined(_WIN32)
    // note: ":" is a string and ':' is a char
    std::string pathvarsep = ":";
#else
    std::string pathvarsep = ";";
#endif
    std::string plugin_path(env_customized_plugins);
    std::vector<std::string> plugin_paths = split(plugin_path, pathvarsep);
    for (const auto& plugin : plugin_paths) {
      std::cerr << "Loading customized plugin defined in DP_PLUGIN_PATH: "
                << plugin << std::endl;
      _load_library_path(plugin);
    }
  }
}

std::string deepmd::name_prefix(const std::string& scope) {
  std::string prefix = "";
  if (scope != "") {
    prefix = scope + "/";
  }
  return prefix;
}

#ifdef BUILD_TENSORFLOW
template <typename MODELTYPE, typename VALUETYPE>
int deepmd::session_input_tensors(
    std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const double& cell_size,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam__,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall) {
  // if datype.size is 0, not clear nframes; but 1 is just ok
  int nframes = datype_.size() > 0 ? (dcoord_.size() / 3 / datype_.size()) : 1;
  int nall = datype_.size();
  int nloc = nall;
  assert(nall * 3 * nframes == dcoord_.size());
  bool b_pbc = (dbox.size() == nframes * 9);

  std::vector<int> datype = atommap.get_type();
  std::vector<int> type_count(ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii) {
    type_count[datype[ii]]++;
  }
  datype.insert(datype.end(), datype_.begin() + nloc, datype_.end());

  TensorShape coord_shape;
  coord_shape.AddDim(nframes);
  coord_shape.AddDim(static_cast<int64_t>(nall) * 3);
  TensorShape type_shape;
  type_shape.AddDim(nframes);
  type_shape.AddDim(nall);
  TensorShape box_shape;
  box_shape.AddDim(nframes);
  box_shape.AddDim(9);
  TensorShape mesh_shape;
  if (b_pbc) {
    mesh_shape.AddDim(6);
  } else {
    mesh_shape.AddDim(0);
  }
  TensorShape natoms_shape;
  natoms_shape.AddDim(2 + ntypes);
  TensorShape fparam_shape;
  fparam_shape.AddDim(nframes);
  fparam_shape.AddDim(fparam_.size() / nframes);
  TensorShape aparam_shape;
  aparam_shape.AddDim(nframes);
  aparam_shape.AddDim(aparam__.size() / nframes);

  tensorflow::DataType model_type;
  if (std::is_same<MODELTYPE, double>::value) {
    model_type = tensorflow::DT_DOUBLE;
  } else if (std::is_same<MODELTYPE, float>::value) {
    model_type = tensorflow::DT_FLOAT;
  } else {
    throw deepmd::deepmd_exception("unsupported data type");
  }
  Tensor coord_tensor(model_type, coord_shape);
  Tensor box_tensor(model_type, box_shape);
  Tensor fparam_tensor(model_type, fparam_shape);
  Tensor aparam_tensor(model_type, aparam_shape);

  Tensor type_tensor(DT_INT32, type_shape);
  Tensor mesh_tensor(DT_INT32, mesh_shape);
  Tensor natoms_tensor(DT_INT32, natoms_shape);

  auto coord = coord_tensor.matrix<MODELTYPE>();
  auto type = type_tensor.matrix<int>();
  auto box = box_tensor.matrix<MODELTYPE>();
  auto mesh = mesh_tensor.flat<int>();
  auto natoms = natoms_tensor.flat<int>();
  auto fparam = fparam_tensor.matrix<MODELTYPE>();
  auto aparam = aparam_tensor.matrix<MODELTYPE>();

  std::vector<VALUETYPE> dcoord(dcoord_);
  atommap.forward<VALUETYPE>(dcoord.begin(), dcoord_.begin(), 3, nframes, nall);
  std::vector<VALUETYPE> aparam_(aparam__);
  if ((aparam_nall ? nall : nloc) > 0) {
    atommap.forward<VALUETYPE>(
        aparam_.begin(), aparam__.begin(),
        aparam__.size() / nframes / (aparam_nall ? nall : nloc), nframes,
        (aparam_nall ? nall : nloc));
  }
  // if == 0, aparam__.size should also be 0, so no need to forward

  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord(ii, jj) = dcoord[ii * nall * 3 + jj];
    }
    if (b_pbc) {
      for (int jj = 0; jj < 9; ++jj) {
        box(ii, jj) = dbox[ii * 9 + jj];
      }
    } else {
      for (int jj = 0; jj < 9; ++jj) {
        box(ii, jj) = 0.;
      }
    }
    for (int jj = 0; jj < nall; ++jj) {
      type(ii, jj) = datype[jj];
    }
    for (int jj = 0; jj < fparam_.size() / nframes; ++jj) {
      fparam(ii, jj) = fparam_[ii * fparam_.size() / nframes + jj];
    }
    for (int jj = 0; jj < aparam_.size() / nframes; ++jj) {
      aparam(ii, jj) = aparam_[ii * aparam_.size() / nframes + jj];
    }
  }
  if (b_pbc) {
    mesh(1 - 1) = 0;
    mesh(2 - 1) = 0;
    mesh(3 - 1) = 0;
    mesh(4 - 1) = 0;
    mesh(5 - 1) = 0;
    mesh(6 - 1) = 0;
  }
  natoms(0) = nloc;
  natoms(1) = nall;
  for (int ii = 0; ii < ntypes; ++ii) {
    natoms(ii + 2) = type_count[ii];
  }

  std::string prefix = "";
  if (scope != "") {
    prefix = scope + "/";
  }
  input_tensors = {
      {prefix + "t_coord", coord_tensor},   {prefix + "t_type", type_tensor},
      {prefix + "t_box", box_tensor},       {prefix + "t_mesh", mesh_tensor},
      {prefix + "t_natoms", natoms_tensor},
  };
  if (fparam_.size() > 0) {
    input_tensors.push_back({prefix + "t_fparam", fparam_tensor});
  }
  if (aparam_.size() > 0) {
    input_tensors.push_back({prefix + "t_aparam", aparam_tensor});
  }
  return nloc;
}

template <typename MODELTYPE, typename VALUETYPE>
int deepmd::session_input_tensors(
    std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    InputNlist& dlist,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam__,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const std::string scope,
    const bool aparam_nall) {
  // if datype.size is 0, not clear nframes; but 1 is just ok
  int nframes = datype_.size() > 0 ? (dcoord_.size() / 3 / datype_.size()) : 1;
  int nall = datype_.size();
  int nloc = nall - nghost;
  assert(nall * 3 * nframes == dcoord_.size());
  assert(dbox.size() == nframes * 9);

  std::vector<int> datype = atommap.get_type();
  std::vector<int> type_count(ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii) {
    type_count[datype[ii]]++;
  }
  datype.insert(datype.end(), datype_.begin() + nloc, datype_.end());

  TensorShape coord_shape;
  coord_shape.AddDim(nframes);
  coord_shape.AddDim(static_cast<int64_t>(nall) * 3);
  TensorShape type_shape;
  type_shape.AddDim(nframes);
  type_shape.AddDim(nall);
  TensorShape box_shape;
  box_shape.AddDim(nframes);
  box_shape.AddDim(9);
  TensorShape mesh_shape;
  mesh_shape.AddDim(16);
  TensorShape natoms_shape;
  natoms_shape.AddDim(2 + ntypes);
  TensorShape fparam_shape;
  fparam_shape.AddDim(nframes);
  fparam_shape.AddDim(fparam_.size() / nframes);
  TensorShape aparam_shape;
  aparam_shape.AddDim(nframes);
  aparam_shape.AddDim(aparam__.size() / nframes);

  tensorflow::DataType model_type;
  if (std::is_same<MODELTYPE, double>::value) {
    model_type = tensorflow::DT_DOUBLE;
  } else if (std::is_same<MODELTYPE, float>::value) {
    model_type = tensorflow::DT_FLOAT;
  } else {
    throw deepmd::deepmd_exception("unsupported data type");
  }
  Tensor coord_tensor(model_type, coord_shape);
  Tensor box_tensor(model_type, box_shape);
  Tensor fparam_tensor(model_type, fparam_shape);
  Tensor aparam_tensor(model_type, aparam_shape);

  Tensor type_tensor(DT_INT32, type_shape);
  Tensor mesh_tensor(DT_INT32, mesh_shape);
  Tensor natoms_tensor(DT_INT32, natoms_shape);

  auto coord = coord_tensor.matrix<MODELTYPE>();
  auto type = type_tensor.matrix<int>();
  auto box = box_tensor.matrix<MODELTYPE>();
  auto mesh = mesh_tensor.flat<int>();
  auto natoms = natoms_tensor.flat<int>();
  auto fparam = fparam_tensor.matrix<MODELTYPE>();
  auto aparam = aparam_tensor.matrix<MODELTYPE>();

  std::vector<VALUETYPE> dcoord(dcoord_);
  atommap.forward<VALUETYPE>(dcoord.begin(), dcoord_.begin(), 3, nframes, nall);
  std::vector<VALUETYPE> aparam_(aparam__);
  if ((aparam_nall ? nall : nloc) > 0) {
    atommap.forward<VALUETYPE>(
        aparam_.begin(), aparam__.begin(),
        aparam__.size() / nframes / (aparam_nall ? nall : nloc), nframes,
        (aparam_nall ? nall : nloc));
  }
  // if == 0, aparam__.size should also be 0, so no need to forward

  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord(ii, jj) = dcoord[ii * nall * 3 + jj];
    }
    for (int jj = 0; jj < 9; ++jj) {
      box(ii, jj) = dbox[ii * 9 + jj];
    }
    for (int jj = 0; jj < nall; ++jj) {
      type(ii, jj) = datype[jj];
    }
    for (int jj = 0; jj < fparam_.size() / nframes; ++jj) {
      fparam(ii, jj) = fparam_[ii * fparam_.size() / nframes + jj];
    }
    for (int jj = 0; jj < aparam_.size() / nframes; ++jj) {
      aparam(ii, jj) = aparam_[ii * aparam_.size() / nframes + jj];
    }
  }

  for (int ii = 0; ii < 16; ++ii) {
    mesh(ii) = 0;
  }

  const int stride = sizeof(int*) / sizeof(int);
  assert(stride * sizeof(int) == sizeof(int*));
  assert(stride <= 4);
  mesh(0) = ago;
  mesh(1) = dlist.inum;
  mesh(2) = 0;
  mesh(3) = 0;
  memcpy(&mesh(4), &(dlist.ilist), sizeof(int*));
  memcpy(&mesh(8), &(dlist.numneigh), sizeof(int*));
  memcpy(&mesh(12), &(dlist.firstneigh), sizeof(int**));

  natoms(0) = nloc;
  natoms(1) = nall;
  for (int ii = 0; ii < ntypes; ++ii) {
    natoms(ii + 2) = type_count[ii];
  }

  std::string prefix = "";
  if (scope != "") {
    prefix = scope + "/";
  }
  input_tensors = {
      {prefix + "t_coord", coord_tensor},   {prefix + "t_type", type_tensor},
      {prefix + "t_box", box_tensor},       {prefix + "t_mesh", mesh_tensor},
      {prefix + "t_natoms", natoms_tensor},
  };
  if (fparam_.size() > 0) {
    input_tensors.push_back({prefix + "t_fparam", fparam_tensor});
  }
  if (aparam_.size() > 0) {
    input_tensors.push_back({prefix + "t_aparam", aparam_tensor});
  }
  return nloc;
}

template <typename MODELTYPE, typename VALUETYPE>
int deepmd::session_input_tensors_mixed_type(
    std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const int& nframes,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const double& cell_size,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam__,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall) {
  int nall = datype_.size() / nframes;
  int nloc = nall;
  assert(nall * 3 * nframes == dcoord_.size());
  bool b_pbc = (dbox.size() == nframes * 9);

  std::vector<int> datype(datype_);
  atommap.forward<int>(datype.begin(), datype_.begin(), 1, nframes, nall);

  TensorShape coord_shape;
  coord_shape.AddDim(nframes);
  coord_shape.AddDim(static_cast<int64_t>(nall) * 3);
  TensorShape type_shape;
  type_shape.AddDim(nframes);
  type_shape.AddDim(nall);
  TensorShape box_shape;
  box_shape.AddDim(nframes);
  box_shape.AddDim(9);
  TensorShape mesh_shape;
  if (b_pbc) {
    mesh_shape.AddDim(7);
  } else {
    mesh_shape.AddDim(1);
  }
  TensorShape natoms_shape;
  natoms_shape.AddDim(2 + ntypes);
  TensorShape fparam_shape;
  fparam_shape.AddDim(nframes);
  fparam_shape.AddDim(fparam_.size() / nframes);
  TensorShape aparam_shape;
  aparam_shape.AddDim(nframes);
  aparam_shape.AddDim(aparam__.size() / nframes);

  tensorflow::DataType model_type;
  if (std::is_same<MODELTYPE, double>::value) {
    model_type = tensorflow::DT_DOUBLE;
  } else if (std::is_same<MODELTYPE, float>::value) {
    model_type = tensorflow::DT_FLOAT;
  } else {
    throw deepmd::deepmd_exception("unsupported data type");
  }
  Tensor coord_tensor(model_type, coord_shape);
  Tensor box_tensor(model_type, box_shape);
  Tensor fparam_tensor(model_type, fparam_shape);
  Tensor aparam_tensor(model_type, aparam_shape);

  Tensor type_tensor(DT_INT32, type_shape);
  Tensor mesh_tensor(DT_INT32, mesh_shape);
  Tensor natoms_tensor(DT_INT32, natoms_shape);

  auto coord = coord_tensor.matrix<MODELTYPE>();
  auto type = type_tensor.matrix<int>();
  auto box = box_tensor.matrix<MODELTYPE>();
  auto mesh = mesh_tensor.flat<int>();
  auto natoms = natoms_tensor.flat<int>();
  auto fparam = fparam_tensor.matrix<MODELTYPE>();
  auto aparam = aparam_tensor.matrix<MODELTYPE>();

  std::vector<VALUETYPE> dcoord(dcoord_);
  atommap.forward<VALUETYPE>(dcoord.begin(), dcoord_.begin(), 3, nframes, nall);
  std::vector<VALUETYPE> aparam_(aparam__);
  if ((aparam_nall ? nall : nloc) > 0) {
    atommap.forward<VALUETYPE>(
        aparam_.begin(), aparam__.begin(),
        aparam__.size() / nframes / (aparam_nall ? nall : nloc), nframes,
        (aparam_nall ? nall : nloc));
  }
  // if == 0, aparam__.size should also be 0, so no need to forward

  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall * 3; ++jj) {
      coord(ii, jj) = dcoord[ii * nall * 3 + jj];
    }
    if (b_pbc) {
      for (int jj = 0; jj < 9; ++jj) {
        box(ii, jj) = dbox[ii * 9 + jj];
      }
    } else {
      for (int jj = 0; jj < 9; ++jj) {
        box(ii, jj) = 0.;
      }
    }
    for (int jj = 0; jj < nall; ++jj) {
      type(ii, jj) = datype[ii * nall + jj];
    }
    for (int jj = 0; jj < fparam_.size() / nframes; ++jj) {
      fparam(ii, jj) = fparam_[ii * fparam_.size() / nframes + jj];
    }
    for (int jj = 0; jj < aparam_.size() / nframes; ++jj) {
      aparam(ii, jj) = aparam_[ii * aparam_.size() / nframes + jj];
    }
  }
  if (b_pbc) {
    mesh(1 - 1) = 0;
    mesh(2 - 1) = 0;
    mesh(3 - 1) = 0;
    mesh(4 - 1) = 0;
    mesh(5 - 1) = 0;
    mesh(6 - 1) = 0;
    mesh(7 - 1) = 0;
  } else {
    mesh(1 - 1) = 0;
  }
  natoms(0) = nloc;
  natoms(1) = nall;
  natoms(2) = nall;
  if (ntypes > 1) {
    for (int ii = 1; ii < ntypes; ++ii) {
      natoms(ii + 2) = 0;
    }
  }

  std::string prefix = "";
  if (scope != "") {
    prefix = scope + "/";
  }
  input_tensors = {
      {prefix + "t_coord", coord_tensor},   {prefix + "t_type", type_tensor},
      {prefix + "t_box", box_tensor},       {prefix + "t_mesh", mesh_tensor},
      {prefix + "t_natoms", natoms_tensor},
  };
  if (fparam_.size() > 0) {
    input_tensors.push_back({prefix + "t_fparam", fparam_tensor});
  }
  if (aparam_.size() > 0) {
    input_tensors.push_back({prefix + "t_aparam", aparam_tensor});
  }
  return nloc;
}

template <typename VT>
VT deepmd::session_get_scalar(Session* session,
                              const std::string name_,
                              const std::string scope) {
  std::string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  deepmd::check_status(
      session->Run(std::vector<std::pair<std::string, Tensor>>({}),
                   {name.c_str()}, {}, &output_tensors));
  Tensor output_rc = output_tensors[0];
  auto orc = output_rc.flat<VT>();
  return orc(0);
}

template <typename VT>
void deepmd::session_get_vector(std::vector<VT>& o_vec,
                                Session* session,
                                const std::string name_,
                                const std::string scope) {
  std::string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  deepmd::check_status(
      session->Run(std::vector<std::pair<std::string, Tensor>>({}),
                   {name.c_str()}, {}, &output_tensors));
  Tensor output_rc = output_tensors[0];
  assert(1 == output_rc.shape().dims());
  int dof = output_rc.shape().dim_size(0);
  o_vec.resize(dof);
  auto orc = output_rc.flat<VT>();
  for (int ii = 0; ii < dof; ++ii) {
    o_vec[ii] = orc(ii);
  }
}

int deepmd::session_get_dtype(tensorflow::Session* session,
                              const std::string name_,
                              const std::string scope) {
  std::string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  deepmd::check_status(
      session->Run(std::vector<std::pair<std::string, Tensor>>({}),
                   {name.c_str()}, {}, &output_tensors));
  Tensor output_rc = output_tensors[0];
  // cast enum to int
  return (int)output_rc.dtype();
}
#endif

#ifdef BUILD_PADDLE
template <typename MODELTYPE, typename VALUETYPE>
int deepmd::predictor_input_tensors(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const double& cell_size,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam__,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall) {
  // if datype.size is 0, not clear nframes; but 1 is just ok
  int nframes = datype_.size() > 0 ? (dcoord_.size() / 3 / datype_.size()) : 1;
  int nall = datype_.size();
  int nloc = nall;
  assert(nall * 3 * nframes == dcoord_.size());
  bool b_pbc = (dbox.size() == nframes * 9);

  std::vector<int> datype = atommap.get_type();
  std::vector<int> type_count(ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii) {
    type_count[datype[ii]]++;
  }
  datype.insert(datype.end(), datype_.begin() + nloc, datype_.end());

  // 准备输入Tensor句柄
  auto input_names = predictor->GetInputNames();
  auto coord_handle = predictor->GetInputHandle(input_names[0]);
  auto type_handle = predictor->GetInputHandle(input_names[1]);
  auto natoms_handle = predictor->GetInputHandle(input_names[2]);
  auto box_handle = predictor->GetInputHandle(input_names[3]);
  auto mesh_handle = predictor->GetInputHandle(input_names[4]);

  // 设置输入 Tensor 的维度信息
  std::vector<int> coord_shape = {nframes, nall * 3};
  std::vector<int> atype_shape = {nframes, nall};
  std::vector<int> box_shape = {nframes, 9};
  std::vector<int> mesh_shape;
  if (b_pbc) {
    mesh_shape = std::vector<int>({6});
  } else {
    mesh_shape = std::vector<int>({0});
  }

  std::vector<int> natoms_shape = {2 + ntypes};

  coord_handle->Reshape(coord_shape);
  type_handle->Reshape(atype_shape);
  natoms_handle->Reshape(natoms_shape);
  box_handle->Reshape(box_shape);
  mesh_handle->Reshape(mesh_shape);

  paddle_infer::DataType model_type;
  if (std::is_same<MODELTYPE, double>::value) {
    model_type = paddle_infer::DataType::FLOAT64;
  } else if (std::is_same<MODELTYPE, float>::value) {
    model_type = paddle_infer::DataType::FLOAT32;
  } else {
    throw deepmd::deepmd_exception("unsupported data type");
  }

  std::vector<VALUETYPE> dcoord(dcoord_);
  atommap.forward<VALUETYPE>(dcoord.begin(), dcoord_.begin(), 3, nframes, nall);
  std::vector<VALUETYPE> aparam_(aparam__);
  if ((aparam_nall ? nall : nloc) > 0) {
    atommap.forward<VALUETYPE>(
        aparam_.begin(), aparam__.begin(),
        aparam__.size() / nframes / (aparam_nall ? nall : nloc), nframes,
        (aparam_nall ? nall : nloc));
  }

  // 发送输入数据到Tensor句柄
  coord_handle->CopyFromCpu(dcoord.data());
  if (b_pbc) {
    box_handle->CopyFromCpu(dbox.data());
  } else {
    std::vector<VALUETYPE> zero = dbox;
    std::fill(zero.begin(), zero.end(), 0);
    box_handle->CopyFromCpu(zero.data());
  }
  std::vector<int> datype_rep(nframes * nall, 0);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall; ++jj) {
      datype_rep[ii * nall + jj] = datype[jj];
    }
  }
  type_handle->CopyFromCpu(datype_rep.data());
  std::vector<int> mesh;
  if (b_pbc) {
    mesh = std::vector<int>(6);
    mesh[1 - 1] = 0;
    mesh[2 - 1] = 0;
    mesh[3 - 1] = 0;
    mesh[4 - 1] = 0;
    mesh[5 - 1] = 0;
    mesh[6 - 1] = 0;
  } else {
    mesh = std::vector<int>(0);
  }
  mesh_handle->CopyFromCpu(mesh.data());
  std::vector<int> natoms = {nloc, nall};
  for (int ii = 0; ii < ntypes; ++ii) {
    natoms.push_back(type_count[ii]);
  }
  natoms_handle->CopyFromCpu(natoms.data());

  return nloc;
}

template <typename MODELTYPE, typename VALUETYPE>
int deepmd::predictor_input_tensors(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    InputNlist& dlist,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam__,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const bool aparam_nall) {
  // if datype.size is 0, not clear nframes; but 1 is just ok
  int nframes = datype_.size() > 0 ? (dcoord_.size() / 3 / datype_.size()) : 1;
  int nall = datype_.size();
  int nloc = nall - nghost;
  assert(nall * 3 * nframes == dcoord_.size());
  assert(dbox.size() == nframes * 9);

  std::vector<int> datype = atommap.get_type();
  // for (int i=0; i<datype.size(); ++i)
  //   printf("datype[%d] = %d\n", i, datype[i]);
  std::vector<int> type_count(ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii) {
    type_count[datype[ii]]++;
  }
  datype.insert(datype.end(), datype_.begin() + nloc, datype_.end());

  // 准备输入Tensor句柄
  auto input_names = predictor->GetInputNames();
  // for (auto &ss: input_names)
  // {
  //   std::cout << "input_name: " << " " << ss << std::endl;
  // }
  auto coord_handle = predictor->GetInputHandle(input_names[0]);
  auto type_handle = predictor->GetInputHandle(input_names[1]);
  // auto natoms_handle = predictor->GetInputHandle(input_names[2]);
  auto box_handle = predictor->GetInputHandle(input_names[2]);
  // auto mesh_handle = predictor->GetInputHandle(input_names[4]);

  // 设置输入 Tensor 的维度信息
  std::vector<int> coord_shape = {nframes, nall, 3};
  std::vector<int> coord_shape_flat = {nframes, nall * 3};

  std::vector<int> atype_shape = {nframes, nall};
  std::vector<int> atype_shape_flat = {nframes, nall};

  std::vector<int> box_shape = {nframes, 3, 3};
  std::vector<int> box_shape_flat = {nframes * 9};
  // std::vector<int> mesh_shape = std::vector<int>({16});
  // std::vector<int> natoms_shape = {2 + ntypes};

  paddle_infer::DataType model_type;
  if (std::is_same<MODELTYPE, double>::value) {
    model_type = paddle_infer::DataType::FLOAT64;
  } else if (std::is_same<MODELTYPE, float>::value) {
    model_type = paddle_infer::DataType::FLOAT32;
  } else {
    throw deepmd::deepmd_exception("unsupported data type");
  }

  coord_handle->Reshape(coord_shape_flat);
  box_handle->Reshape(box_shape_flat);
  type_handle->Reshape(atype_shape_flat);
  // printf("coord.shape = [");
  // for (auto &d: coord_shape)
  // {
  //   printf("%d, ", d);
  // }
  // printf("]\n");

  // printf("type.shape = [");
  // for (auto &d: atype_shape)
  // {
  //   printf("%d, ", d);
  // }
  // printf("]\n");

  // printf("box.shape = [");
  // for (auto &d: box_shape)
  // {
  //   printf("%d, ", d);
  // }
  // printf("]\n");
  // mesh_handle->Reshape(mesh_shape);
  // natoms_handle->Reshape(natoms_shape);

  std::vector<VALUETYPE> dcoord(dcoord_);
  atommap.forward<VALUETYPE>(dcoord.begin(), dcoord_.begin(), 3, nframes, nall); //012
  std::vector<VALUETYPE> aparam_(aparam__);
  if ((aparam_nall ? nall : nloc) > 0) {
    atommap.forward<VALUETYPE>(
        aparam_.begin(), aparam__.begin(),
        aparam__.size() / nframes / (aparam_nall ? nall : nloc), nframes,
        (aparam_nall ? nall : nloc));
  }

  // const std::string filename = "/workspace/hesensen/deepmd_backend/deepmd_paddle_new/examples/water/lmp/coord_torch.log";
  // std::ifstream inputFile(filename);
  // VALUETYPE number;
  // int iii = 0;
  // while (inputFile >> number) {
  //     dcoord[iii] = number;
  //     ++iii;
  // }
  // printf("dcoord finished, iii = %d\n", iii);
  // inputFile.close();

  // 发送输入数据到Tensor句柄
  coord_handle->CopyFromCpu(dcoord.data());
  coord_handle->Reshape(coord_shape);
  box_handle->CopyFromCpu(dbox.data());
  box_handle->Reshape(box_shape);
  // for (int i = 0; i < dcoord.size(); ++i)
  // {
  //   printf("dcoord[%d] = %.6lf\n", i, dcoord[i]);
  // }
  std::vector<int> datype_rep(nframes * nall, 0);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall; ++jj) {
      datype_rep[ii * nall + jj] = datype[jj];
    }
  }
  // const std::string filename1 = "/workspace/hesensen/deepmd_backend/deepmd_paddle_new/examples/water/lmp/type_torch.log";
  // std::ifstream inputFile1(filename1);
  // int number_int;
  // iii = 0;
  // while (inputFile1 >> number_int) {
  //     datype_rep[iii] = number_int;
  //     ++iii;
  // }
  // printf("atype finishied, iii = %d\n", iii);
  // inputFile1.close();

  type_handle->CopyFromCpu(datype_rep.data());
  // for (int i = 0; i < datype_rep.size(); ++i)
  // {
  //   printf("%d\n", datype_rep[i]);
  // }
  type_handle->Reshape(atype_shape);
  // std::vector<int> mesh(mesh_shape[0], 0);
  // for (int ii = 0; ii < 16; ++ii) {
  //   mesh[ii] = 0;
  // }
  // const int stride = sizeof(int*) / sizeof(int);
  // assert(stride * sizeof(int) == sizeof(int*));
  // assert(stride <= 4);
  // mesh[0] = ago;
  // mesh[1] = dlist.inum;
  // mesh[2] = 0;
  // mesh[3] = 0;
  // memcpy(&mesh[4], &(dlist.ilist), sizeof(int*));
  // memcpy(&mesh[8], &(dlist.numneigh), sizeof(int*));
  // memcpy(&mesh[12], &(dlist.firstneigh), sizeof(int**));
  // mesh_handle->CopyFromCpu(mesh.data());

  // std::vector<int> natoms = {nloc, nall};
  // for (int ii = 0; ii < ntypes; ++ii) {
  //   natoms.push_back(type_count[ii]);
  // }
  // natoms_handle->CopyFromCpu(natoms.data());
  // printf("finished predictor_input_tensors\n");
  // printf("nloc = %d\n", nloc);
  return nloc;
}

template <typename MODELTYPE, typename VALUETYPE>
int deepmd::predictor_input_tensors_mixed_type(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const int& nframes,
    const std::vector<VALUETYPE>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const double& cell_size,
    const std::vector<VALUETYPE>& fparam_,
    const std::vector<VALUETYPE>& aparam__,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall) {
  int nall = datype_.size() / nframes;
  int nloc = nall;
  assert(nall * 3 * nframes == dcoord_.size());
  bool b_pbc = (dbox.size() == nframes * 9);

  std::vector<int> datype(datype_);
  atommap.forward<int>(datype.begin(), datype_.begin(), 1, nframes, nall);

  auto input_names = predictor->GetInputNames();
  auto coord_handle = predictor->GetInputHandle(input_names[0]);
  auto type_handle = predictor->GetInputHandle(input_names[1]);
  auto box_handle = predictor->GetInputHandle(input_names[3]);
  auto mesh_handle = predictor->GetInputHandle(input_names[4]);
  auto natoms_handle = predictor->GetInputHandle(input_names[2]);

  // 设置输入 Tensor 的维度信息
  std::vector<int> coord_shape = {nframes, nall * 3};
  std::vector<int> atype_shape = {nframes, nall};
  std::vector<int> box_shape = {nframes, 9};
  std::vector<int> mesh_shape;
  if (b_pbc) {
    mesh_shape = std::vector<int>({7});
  } else {
    mesh_shape = std::vector<int>({1});
  }
  std::vector<int> natoms_shape = {2 + ntypes};

  coord_handle->Reshape(coord_shape);
  type_handle->Reshape(atype_shape);
  box_handle->Reshape(box_shape);
  mesh_handle->Reshape(mesh_shape);
  natoms_handle->Reshape(natoms_shape);

  paddle_infer::DataType model_type;
  if (std::is_same<MODELTYPE, double>::value) {
    model_type = paddle_infer::DataType::FLOAT64;
  } else if (std::is_same<MODELTYPE, float>::value) {
    model_type = paddle_infer::DataType::FLOAT32;
  } else {
    throw deepmd::deepmd_exception("unsupported data type");
  }

  std::vector<VALUETYPE> dcoord(dcoord_);
  atommap.forward<VALUETYPE>(dcoord.begin(), dcoord_.begin(), 3, nframes, nall);
  std::vector<VALUETYPE> aparam_(aparam__);
  if ((aparam_nall ? nall : nloc) > 0) {
    atommap.forward<VALUETYPE>(
        aparam_.begin(), aparam__.begin(),
        aparam__.size() / nframes / (aparam_nall ? nall : nloc), nframes,
        (aparam_nall ? nall : nloc));
  }
  // coord
  coord_handle->CopyFromCpu(dcoord.data());

  // box
  if (b_pbc) {
    box_handle->CopyFromCpu(dbox.data());
  } else {
    std::vector<VALUETYPE> zero = dbox;
    std::fill(zero.begin(), zero.end(), 0);
    box_handle->CopyFromCpu(zero.data());
  }

  // datype
  std::vector<int> datype_rep(nframes * nall, 0);
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nall; ++jj) {
      datype_rep[ii * nall + jj] = datype[jj];
    }
  }
  type_handle->CopyFromCpu(datype_rep.data());
  // mesh
  std::vector<int> mesh;
  if (b_pbc) {
    mesh = std::vector<int>(7, 0);
    mesh[1 - 1] = 0;
    mesh[2 - 1] = 0;
    mesh[3 - 1] = 0;
    mesh[4 - 1] = 0;
    mesh[5 - 1] = 0;
    mesh[6 - 1] = 0;
    mesh[7 - 1] = 0;
  } else {
    mesh = std::vector<int>(1, 0);
    mesh[1 - 1] = 0;
  }
  mesh_handle->CopyFromCpu(mesh.data());
  //natoms
  std::vector<int> natoms_pad = {nloc, nall, nall};
  if (ntypes > 1) {
    for (int ii = 0; ii < ntypes; ++ii) {
      natoms_pad.push_back(0);
    }
  }
  natoms_handle->CopyFromCpu(natoms_pad.data());

  // if (fparam_.size() > 0) {
  //   input_tensors.push_back({prefix + "t_fparam", fparam_tensor});
  // }
  // if (aparam_.size() > 0) {
  //   input_tensors.push_back({prefix + "t_aparam", aparam_tensor});
  // }
  return nloc;
}

#endif

#ifdef BUILD_PADDLE
template <typename VT>
VT deepmd::predictor_get_scalar(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::string& name_) {
  if (std::is_same<VT, std::string>::value) {
    /*
    NOTE: Convert from ascii code(int64) to std::string,
    A workaround for string data type is not supported in Paddle yet.
    */
    auto scalar_tensor = predictor->GetOutputHandle(name_);
    if (scalar_tensor->shape().size() == 0) {
      return VT();
    }
    const auto& shape = scalar_tensor->shape();
    const int& str_len = std::accumulate(std::begin(shape), std::end(shape), 1,
                                         std::multiplies<>{});
    if (str_len == 0) {
      return VT();
    }
    int32_t* scalar_ptr = (int32_t*)malloc(str_len * sizeof(int32_t));
    scalar_tensor->CopyToCpu(scalar_ptr);
    VT ret;
    for (int ii = 0; ii < str_len; ++ii) {
      ret += (char)scalar_ptr[ii];
    }
    free(scalar_ptr);
    return ret;
  } else {
    /* Vanillia process for other data type below*/
    auto scalar_tensor = predictor->GetOutputHandle(name_);
    // VT* scalar_ptr = (VT*)malloc(1 * sizeof(VT));
    std::unique_ptr<VT> scalar_ptr(new VT);
    scalar_tensor->CopyToCpu(scalar_ptr.get());
    return (*scalar_ptr);
  }
}


// template <typename VT>
// void deepmd::session_get_vector(std::vector<VT>& o_vec,
//                                 Session* session,
//                                 const std::string name_,
//                                 const std::string scope) {
//   std::string name = name_;
//   if (scope != "") {
//     name = scope + "/" + name;
//   }
//   std::vector<Tensor> output_tensors;
//   deepmd::check_status(
//       session->Run(std::vector<std::pair<std::string, Tensor>>({}),
//                    {name.c_str()}, {}, &output_tensors));
//   Tensor output_rc = output_tensors[0];
//   assert(1 == output_rc.shape().dims());
//   int dof = output_rc.shape().dim_size(0);
//   o_vec.resize(dof);
//   auto orc = output_rc.flat<VT>();
//   for (int ii = 0; ii < dof; ++ii) {
//     o_vec[ii] = orc(ii);
//   }
// }

paddle_infer::DataType deepmd::predictor_get_dtype(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::string& name_) {
  auto scalar_tensor = predictor->GetOutputHandle(name_);
  return scalar_tensor->type();
}

#endif

template <typename VT>
void deepmd::select_map(std::vector<VT>& out,
                        const std::vector<VT>& in,
                        const std::vector<int>& idx_map,
                        const int& stride,
                        const int& nframes,
                        const int& nall1,
                        const int& nall2) {
  for (int kk = 0; kk < nframes; ++kk) {
#ifdef DEBUG
    assert(in.size() / stride * stride == in.size() &&
           "in size should be multiples of stride")
#endif
        for (int ii = 0; ii < in.size() / stride / nframes; ++ii) {
#ifdef DEBUG
      assert(ii < idx_map.size() && "idx goes over the idx map size");
      assert(idx_map[ii] < out.size() && "mappped idx goes over the out size");
#endif
      if (idx_map[ii] >= 0) {
        int to_ii = idx_map[ii];
        for (int dd = 0; dd < stride; ++dd) {
          out[kk * nall1 * stride + to_ii * stride + dd] =
              in[kk * nall2 * stride + ii * stride + dd];
        }
      }
    }
  }
}

template <typename VT>
void deepmd::select_map(typename std::vector<VT>::iterator out,
                        const typename std::vector<VT>::const_iterator in,
                        const std::vector<int>& idx_map,
                        const int& stride,
                        const int& nframes,
                        const int& nall1,
                        const int& nall2) {
  for (int kk = 0; kk < nframes; ++kk) {
    for (int ii = 0; ii < idx_map.size(); ++ii) {
      if (idx_map[ii] >= 0) {
        int to_ii = idx_map[ii];
        for (int dd = 0; dd < stride; ++dd) {
          *(out + kk * nall1 * stride + to_ii * stride + dd) =
              *(in + kk * nall2 * stride + ii * stride + dd);
        }
      }
    }
  }
}

// sel_map(_,_,fwd_map,_) == sel_map_inv(_,_,bkw_map,_)
template <typename VT>
void deepmd::select_map_inv(std::vector<VT>& out,
                            const std::vector<VT>& in,
                            const std::vector<int>& idx_map,
                            const int& stride) {
#ifdef DEBUG
  assert(in.size() / stride * stride == in.size() &&
         "in size should be multiples of stride");
#endif
  for (int ii = 0; ii < out.size() / stride; ++ii) {
#ifdef DEBUG
    assert(ii < idx_map.size() && "idx goes over the idx map size");
    assert(idx_map[ii] < in.size() && "from idx goes over the in size");
#endif
    if (idx_map[ii] >= 0) {
      int from_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd) {
        out[ii * stride + dd] = in[from_ii * stride + dd];
      }
    }
  }
}

template <typename VT>
void deepmd::select_map_inv(typename std::vector<VT>::iterator out,
                            const typename std::vector<VT>::const_iterator in,
                            const std::vector<int>& idx_map,
                            const int& stride) {
  for (int ii = 0; ii < idx_map.size(); ++ii) {
    if (idx_map[ii] >= 0) {
      int from_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd) {
        *(out + ii * stride + dd) = *(in + from_ii * stride + dd);
      }
    }
  }
}

#ifdef BUILD_TENSORFLOW
template int deepmd::session_get_scalar<int>(Session*,
                                             const std::string,
                                             const std::string);

template bool deepmd::session_get_scalar<bool>(Session*,
                                               const std::string,
                                               const std::string);

template void deepmd::session_get_vector<int>(std::vector<int>&,
                                              Session*,
                                              const std::string,
                                              const std::string);
#endif

#ifdef BUILD_PADDLE
template int deepmd::predictor_get_scalar<int>(const std::shared_ptr<paddle_infer::Predictor>& predictor,
                                             const std::string &name_);

template bool deepmd::predictor_get_scalar<bool>(const std::shared_ptr<paddle_infer::Predictor>& predictor,
                                               const std::string &name_);

// template void deepmd::session_get_vector<int>(std::vector<int>&,
//                                               Session*,
//                                               const std::string,
//                                               const std::string);
#endif

template void deepmd::select_map<int>(std::vector<int>& out,
                                      const std::vector<int>& in,
                                      const std::vector<int>& idx_map,
                                      const int& stride,
                                      const int& nframes,
                                      const int& nall1,
                                      const int& nall2);

template void deepmd::select_map<int>(
    typename std::vector<int>::iterator out,
    const typename std::vector<int>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride,
    const int& nframes,
    const int& nall1,
    const int& nall2);

template void deepmd::select_map_inv<int>(std::vector<int>& out,
                                          const std::vector<int>& in,
                                          const std::vector<int>& idx_map,
                                          const int& stride);

template void deepmd::select_map_inv<int>(
    typename std::vector<int>::iterator out,
    const typename std::vector<int>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride);

#ifdef BUILD_TENSORFLOW
template float deepmd::session_get_scalar<float>(Session*,
                                                 const std::string,
                                                 const std::string);

template void deepmd::session_get_vector<float>(std::vector<float>&,
                                                Session*,
                                                const std::string,
                                                const std::string);
#endif

#ifdef BUILD_PADDLE
template float deepmd::predictor_get_scalar<float>(const std::shared_ptr<paddle_infer::Predictor>& predictor,
                                                 const std::string &name_);

#endif

template void deepmd::select_map<float>(std::vector<float>& out,
                                        const std::vector<float>& in,
                                        const std::vector<int>& idx_map,
                                        const int& stride,
                                        const int& nframes,
                                        const int& nall1,
                                        const int& nall2);

template void deepmd::select_map<float>(
    typename std::vector<float>::iterator out,
    const typename std::vector<float>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride,
    const int& nframes,
    const int& nall1,
    const int& nall2);

template void deepmd::select_map_inv<float>(std::vector<float>& out,
                                            const std::vector<float>& in,
                                            const std::vector<int>& idx_map,
                                            const int& stride);

template void deepmd::select_map_inv<float>(
    typename std::vector<float>::iterator out,
    const typename std::vector<float>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride);

#ifdef BUILD_TENSORFLOW
template double deepmd::session_get_scalar<double>(Session*,
                                                   const std::string,
                                                   const std::string);

template void deepmd::session_get_vector<double>(std::vector<double>&,
                                                 Session*,
                                                 const std::string,
                                                 const std::string);
#endif

#ifdef BUILD_PADDLE
template double deepmd::predictor_get_scalar<double>(const std::shared_ptr<paddle_infer::Predictor>& predictor,
                                                 const std::string& name_);
#endif

template void deepmd::select_map<double>(std::vector<double>& out,
                                         const std::vector<double>& in,
                                         const std::vector<int>& idx_map,
                                         const int& stride,
                                         const int& nframes,
                                         const int& nall1,
                                         const int& nall2);

template void deepmd::select_map<double>(
    typename std::vector<double>::iterator out,
    const typename std::vector<double>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride,
    const int& nframes,
    const int& nall1,
    const int& nall2);

template void deepmd::select_map_inv<double>(std::vector<double>& out,
                                             const std::vector<double>& in,
                                             const std::vector<int>& idx_map,
                                             const int& stride);

template void deepmd::select_map_inv<double>(
    typename std::vector<double>::iterator out,
    const typename std::vector<double>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride);

#ifdef BUILD_TENSORFLOW
template deepmd::STRINGTYPE deepmd::session_get_scalar<deepmd::STRINGTYPE>(
    Session*, const std::string, const std::string);

template void deepmd::session_get_vector<deepmd::STRINGTYPE>(
    std::vector<deepmd::STRINGTYPE>&,
    Session*,
    const std::string,
    const std::string);

template void deepmd::select_map<deepmd::STRINGTYPE>(
    std::vector<deepmd::STRINGTYPE>& out,
    const std::vector<deepmd::STRINGTYPE>& in,
    const std::vector<int>& idx_map,
    const int& stride,
    const int& nframes,
    const int& nall1,
    const int& nall2);

template void deepmd::select_map<deepmd::STRINGTYPE>(
    typename std::vector<deepmd::STRINGTYPE>::iterator out,
    const typename std::vector<deepmd::STRINGTYPE>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride,
    const int& nframes,
    const int& nall1,
    const int& nall2);

template void deepmd::select_map_inv<deepmd::STRINGTYPE>(
    std::vector<deepmd::STRINGTYPE>& out,
    const std::vector<deepmd::STRINGTYPE>& in,
    const std::vector<int>& idx_map,
    const int& stride);

template void deepmd::select_map_inv<deepmd::STRINGTYPE>(
    typename std::vector<deepmd::STRINGTYPE>::iterator out,
    const typename std::vector<deepmd::STRINGTYPE>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride);
#endif

#ifdef BUILD_PADDLE
template std::string deepmd::predictor_get_scalar<std::string>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor, const std::string &name_);

// template void deepmd::session_get_vector<std::string>(
//     std::vector<std::string>&,
//     const std::shared_ptr<paddle_infer::Predictor>& predictor,
//     const std::string);

template void deepmd::select_map<std::string>(
    std::vector<std::string>& out,
    const std::vector<std::string>& in,
    const std::vector<int>& idx_map,
    const int& stride,
    const int& nframes,
    const int& nall1,
    const int& nall2);

template void deepmd::select_map<std::string>(
    typename std::vector<std::string>::iterator out,
    const typename std::vector<std::string>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride,
    const int& nframes,
    const int& nall1,
    const int& nall2);

template void deepmd::select_map_inv<std::string>(
    std::vector<std::string>& out,
    const std::vector<std::string>& in,
    const std::vector<int>& idx_map,
    const int& stride);

template void deepmd::select_map_inv<std::string>(
    typename std::vector<std::string>::iterator out,
    const typename std::vector<std::string>::const_iterator in,
    const std::vector<int>& idx_map,
    const int& stride);
#endif

void deepmd::read_file_to_string(std::string model, std::string& file_content) {
  // generated by GitHub Copilot
  std::ifstream file(model);
  if (file.is_open()) {
    std::stringstream buffer;
    buffer << file.rdbuf();
    file_content = buffer.str();
    file.close();
  } else {
    throw deepmd::deepmd_exception("Failed to open file: " + model);
  }
}

void deepmd::convert_pbtxt_to_pb(std::string fn_pb_txt, std::string fn_pb) {
#ifdef BUILD_TENSORFLOW
  int fd = open(fn_pb_txt.c_str(), O_RDONLY);
  tensorflow::protobuf::io::ZeroCopyInputStream* input =
      new tensorflow::protobuf::io::FileInputStream(fd);
  tensorflow::GraphDef graph_def;
  tensorflow::protobuf::TextFormat::Parse(input, &graph_def);
  delete input;
  std::fstream output(fn_pb,
                      std::ios::out | std::ios::trunc | std::ios::binary);
  graph_def.SerializeToOstream(&output);
#else
  throw deepmd::deepmd_exception(
      "convert_pbtxt_to_pb: TensorFlow backend is not enabled.");
#endif
}

#ifdef BUILD_TENSORFLOW
template int deepmd::session_input_tensors<double, double>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);
template int deepmd::session_input_tensors<float, double>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);

template int deepmd::session_input_tensors<double, float>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);
template int deepmd::session_input_tensors<float, float>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);

template int deepmd::session_input_tensors<double, double>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    InputNlist& dlist,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const std::string scope,
    const bool aparam_nall);
template int deepmd::session_input_tensors<float, double>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    InputNlist& dlist,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const std::string scope,
    const bool aparam_nall);

template int deepmd::session_input_tensors<double, float>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    InputNlist& dlist,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const std::string scope,
    const bool aparam_nall);
template int deepmd::session_input_tensors<float, float>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    InputNlist& dlist,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const std::string scope,
    const bool aparam_nall);

template int deepmd::session_input_tensors_mixed_type<double, double>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);
template int deepmd::session_input_tensors_mixed_type<float, double>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);

template int deepmd::session_input_tensors_mixed_type<double, float>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);
template int deepmd::session_input_tensors_mixed_type<float, float>(
    std::vector<std::pair<std::string, tensorflow::Tensor>>& input_tensors,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const std::string scope,
    const bool aparam_nall);
#endif

#ifdef BUILD_PADDLE
template int deepmd::predictor_input_tensors<double, double>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);
template int deepmd::predictor_input_tensors<float, double>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);

template int deepmd::predictor_input_tensors<double, float>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);
template int deepmd::predictor_input_tensors<float, float>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);

template int deepmd::predictor_input_tensors<double, double>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    InputNlist& dlist,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const bool aparam_nall);
template int deepmd::predictor_input_tensors<float, double>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    InputNlist& dlist,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const bool aparam_nall);

template int deepmd::predictor_input_tensors<double, float>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    InputNlist& dlist,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const bool aparam_nall);
template int deepmd::predictor_input_tensors<float, float>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    InputNlist& dlist,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const int nghost,
    const int ago,
    const bool aparam_nall);

template int deepmd::predictor_input_tensors_mixed_type<double, double>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);
template int deepmd::predictor_input_tensors_mixed_type<float, double>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const double& cell_size,
    const std::vector<double>& fparam_,
    const std::vector<double>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);

template int deepmd::predictor_input_tensors_mixed_type<double, float>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);
template int deepmd::predictor_input_tensors_mixed_type<float, float>(
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const int& ntypes,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const double& cell_size,
    const std::vector<float>& fparam_,
    const std::vector<float>& aparam_,
    const deepmd::AtomMap& atommap,
    const bool aparam_nall);
#endif

void deepmd::print_summary(const std::string& pre) {
  int num_intra_nthreads, num_inter_nthreads;
  deepmd::get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  std::cout << pre << "installed to:       " + global_install_prefix << "\n";
  std::cout << pre << "source:             " + global_git_summ << "\n";
  std::cout << pre << "source branch:      " + global_git_branch << "\n";
  std::cout << pre << "source commit:      " + global_git_hash << "\n";
  std::cout << pre << "source commit at:   " + global_git_date << "\n";
  std::cout << pre << "support model ver.: " + global_model_version << "\n";
#if defined(GOOGLE_CUDA)
  std::cout << pre << "build variant:      cuda"
            << "\n";
#elif defined(TENSORFLOW_USE_ROCM)
  std::cout << pre << "build variant:      rocm"
            << "\n";
#else
  std::cout << pre << "build variant:      cpu"
            << "\n";
#endif
#ifdef BUILD_TENSORFLOW
  std::cout << pre << "build with tf inc:  " + global_tf_include_dir << "\n";
  std::cout << pre << "build with tf lib:  " + global_tf_lib << "\n";
#endif
#ifdef BUILD_PYTORCH
  std::cout << pre << "build with pt lib:  " + global_pt_lib << "\n";
#endif
#ifdef BUILD_PADDLE
  std::cout << pre << "build with pd lib:  " + global_pd_lib << "\n";
#endif
  std::cout << pre
            << "set tf intra_op_parallelism_threads: " << num_intra_nthreads
            << "\n";
  std::cout << pre
            << "set tf inter_op_parallelism_threads: " << num_inter_nthreads
            << std::endl;
}
