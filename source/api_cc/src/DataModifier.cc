// SPDX-License-Identifier: LGPL-3.0-or-later
#include "DataModifier.h"

using namespace deepmd;
using namespace tensorflow;

DipoleChargeModifier::DipoleChargeModifier()
    : inited(false), graph_def(new GraphDef()) {}

DipoleChargeModifier::DipoleChargeModifier(const std::string& model,
                                           const int& gpu_rank,
                                           const std::string& name_scope_)
    : inited(false), name_scope(name_scope_), graph_def(new GraphDef()) {
  init(model, gpu_rank, name_scope_);
}

DipoleChargeModifier::~DipoleChargeModifier() { delete graph_def; };

void DipoleChargeModifier::init(const std::string& model,
                                const int& gpu_rank,
                                const std::string& name_scope_) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  name_scope = name_scope_;
  SessionOptions options;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  deepmd::load_op_library();
  int gpu_num = -1;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  DPGetDeviceCount(gpu_num);  // check current device environment
  if (gpu_num > 0) {
    options.config.set_allow_soft_placement(true);
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
        0.9);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    DPErrcheck(DPSetDevice(gpu_rank % gpu_num));
    std::string str = "/gpu:";
    str += std::to_string(gpu_rank % gpu_num);
    graph::SetDefaultDevice(str, graph_def);
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  deepmd::check_status(NewSession(options, &session));
  deepmd::check_status(ReadBinaryProto(Env::Default(), model, graph_def));
  deepmd::check_status(session->Create(*graph_def));
  // int nnodes = graph_def.node_size();
  // for (int ii = 0; ii < nnodes; ++ii){
  //   cout << ii << " \t " << graph_def.node(ii).name() << endl;
  // }
  dtype = session_get_dtype(session, "descrpt_attr/rcut");
  if (dtype == tensorflow::DT_DOUBLE) {
    rcut = get_scalar<double>("descrpt_attr/rcut");
  } else {
    rcut = get_scalar<float>("descrpt_attr/rcut");
  }
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  get_vector<int>(sel_type, "model_attr/sel_type");
  sort(sel_type.begin(), sel_type.end());
  inited = true;
}

template <class VT>
VT DipoleChargeModifier::get_scalar(const std::string& name) const {
  return session_get_scalar<VT>(session, name, name_scope);
}

template <class VT>
void DipoleChargeModifier::get_vector(std::vector<VT>& vec,
                                      const std::string& name) const {
  session_get_vector<VT>(vec, session, name, name_scope);
}

template <typename MODELTYPE, typename VALUETYPE>
void DipoleChargeModifier::run_model(
    std::vector<VALUETYPE>& dforce,
    std::vector<VALUETYPE>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nghost) {
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  if (nloc == 0) {
    dforce.clear();
    dvirial.clear();
    return;
  }

  std::vector<Tensor> output_tensors;
  deepmd::check_status(session->Run(input_tensors,
                                    {"o_dm_force", "o_dm_virial", "o_dm_av"},
                                    {}, &output_tensors));
  int cc = 0;
  Tensor output_f = output_tensors[cc++];
  Tensor output_v = output_tensors[cc++];
  Tensor output_av = output_tensors[cc++];
  assert(output_f.dims() == 2 && "dim of output tensor should be 2");
  assert(output_v.dims() == 2 && "dim of output tensor should be 2");
  assert(output_av.dims() == 2 && "dim of output tensor should be 2");
  int nframes = output_f.dim_size(0);
  int natoms = output_f.dim_size(1) / 3;
  assert(output_f.dim_size(0) == 1 && "nframes should match");
  assert(natoms == nall && "natoms should be nall");
  assert(output_v.dim_size(0) == nframes && "nframes should match");
  assert(output_v.dim_size(1) == 9 && "dof of virial should be 9");
  assert(output_av.dim_size(0) == nframes && "nframes should match");
  assert(output_av.dim_size(1) == natoms * 9 &&
         "dof of atom virial should be 9 * natoms");

  auto of = output_f.flat<MODELTYPE>();
  auto ov = output_v.flat<MODELTYPE>();

  dforce.resize(nall * 3);
  dvirial.resize(9);
  for (int ii = 0; ii < nall * 3; ++ii) {
    dforce[ii] = of(ii);
  }
  for (int ii = 0; ii < 9; ++ii) {
    dvirial[ii] = ov(ii);
  }
}

template void DipoleChargeModifier::run_model<double, double>(
    std::vector<double>& dforce,
    std::vector<double>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nghost);

template void DipoleChargeModifier::run_model<float, double>(
    std::vector<double>& dforce,
    std::vector<double>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nghost);

template void DipoleChargeModifier::run_model<double, float>(
    std::vector<float>& dforce,
    std::vector<float>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nghost);

template void DipoleChargeModifier::run_model<float, float>(
    std::vector<float>& dforce,
    std::vector<float>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nghost);

template <typename VALUETYPE>
void DipoleChargeModifier::compute(
    std::vector<VALUETYPE>& dfcorr_,
    std::vector<VALUETYPE>& dvcorr_,
    const std::vector<VALUETYPE>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<VALUETYPE>& delef_,
    const int nghost,
    const InputNlist& lmp_list) {
  // firstly do selection
  int nall = datype_.size();
  int nloc = nall - nghost;
  int nghost_real;
  std::vector<int> real_fwd_map, real_bkw_map;
  select_real_atoms(real_fwd_map, real_bkw_map, nghost_real, dcoord_, datype_,
                    nghost, ntypes);
  int nall_real = real_bkw_map.size();
  int nloc_real = nall_real - nghost_real;
  if (nloc_real == 0) {
    dfcorr_.resize(nall * 3);
    dvcorr_.resize(9);
    fill(dfcorr_.begin(), dfcorr_.end(), (VALUETYPE)0.0);
    fill(dvcorr_.begin(), dvcorr_.end(), (VALUETYPE)0.0);
    return;
  }
  // resize to nall_real
  std::vector<VALUETYPE> dcoord_real;
  std::vector<VALUETYPE> delef_real;
  std::vector<int> datype_real;
  dcoord_real.resize(nall_real * 3);
  delef_real.resize(nall_real * 3);
  datype_real.resize(nall_real);
  // fwd map
  select_map<VALUETYPE>(dcoord_real, dcoord_, real_fwd_map, 3);
  select_map<VALUETYPE>(delef_real, delef_, real_fwd_map, 3);
  select_map<int>(datype_real, datype_, real_fwd_map, 1);
  // internal nlist
  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(lmp_list);
  nlist_data.shuffle_exclude_empty(real_fwd_map);
  // sort atoms
  AtomMap atommap(datype_real.begin(), datype_real.begin() + nloc_real);
  assert(nloc_real == atommap.get_type().size());
  const std::vector<int>& sort_fwd_map(atommap.get_fwd_map());
  const std::vector<int>& sort_bkw_map(atommap.get_bkw_map());
  // shuffle nlist
  nlist_data.shuffle(atommap);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);
  // make input tensors
  std::vector<std::pair<std::string, Tensor>> input_tensors;
  int ret;
  if (dtype == tensorflow::DT_DOUBLE) {
    ret = session_input_tensors<double>(
        input_tensors, dcoord_real, ntypes, datype_real, dbox, nlist,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap,
        nghost_real, 0, name_scope);
  } else {
    ret = session_input_tensors<float>(
        input_tensors, dcoord_real, ntypes, datype_real, dbox, nlist,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap,
        nghost_real, 0, name_scope);
  }
  assert(nloc_real == ret);
  // make bond idx map
  std::vector<int> bd_idx(nall, -1);
  for (int ii = 0; ii < pairs.size(); ++ii) {
    bd_idx[pairs[ii].first] = pairs[ii].second;
  }
  // make extf by bond idx map
  std::vector<int> dtype_sort_loc = atommap.get_type();
  std::vector<VALUETYPE> dextf;
  for (int ii = 0; ii < dtype_sort_loc.size(); ++ii) {
    if (binary_search(sel_type.begin(), sel_type.end(), dtype_sort_loc[ii])) {
      // selected atom
      int first_idx = real_bkw_map[sort_bkw_map[ii]];
      int second_idx = bd_idx[first_idx];
      assert(second_idx >= 0);
      dextf.push_back(delef_[second_idx * 3 + 0]);
      dextf.push_back(delef_[second_idx * 3 + 1]);
      dextf.push_back(delef_[second_idx * 3 + 2]);
    }
  }
  // dextf should be loc and virtual
  assert(dextf.size() == (nloc - nloc_real) * 3);
  // make tensor for extf
  int nframes = 1;
  TensorShape extf_shape;
  extf_shape.AddDim(nframes);
  extf_shape.AddDim(dextf.size());
  Tensor extf_tensor((tensorflow::DataType)dtype, extf_shape);
  if (dtype == tensorflow::DT_DOUBLE) {
    auto extf = extf_tensor.matrix<double>();
    for (int ii = 0; ii < nframes; ++ii) {
      for (int jj = 0; jj < extf.size(); ++jj) {
        extf(ii, jj) = dextf[jj];
      }
    }
  } else {
    auto extf = extf_tensor.matrix<float>();
    for (int ii = 0; ii < nframes; ++ii) {
      for (int jj = 0; jj < extf.size(); ++jj) {
        extf(ii, jj) = dextf[jj];
      }
    }
  }
  // append extf to input tensor
  input_tensors.push_back({"t_ef", extf_tensor});
  // run model
  std::vector<VALUETYPE> dfcorr, dvcorr;
  if (dtype == tensorflow::DT_DOUBLE) {
    run_model<double>(dfcorr, dvcorr, session, input_tensors, atommap,
                      nghost_real);
  } else {
    run_model<float>(dfcorr, dvcorr, session, input_tensors, atommap,
                     nghost_real);
  }
  assert(dfcorr.size() == nall_real * 3);
  // back map force
  std::vector<VALUETYPE> dfcorr_1 = dfcorr;
  atommap.backward<VALUETYPE>(dfcorr_1.begin(), dfcorr.begin(), 3);
  assert(dfcorr_1.size() == nall_real * 3);
  // resize to all and clear
  std::vector<VALUETYPE> dfcorr_2(nall * 3);
  fill(dfcorr_2.begin(), dfcorr_2.end(), (VALUETYPE)0.0);
  // back map to original position
  for (int ii = 0; ii < nall_real; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dfcorr_2[real_bkw_map[ii] * 3 + dd] += dfcorr_1[ii * 3 + dd];
    }
  }
  // self correction of bonded force
  for (int ii = 0; ii < pairs.size(); ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dfcorr_2[pairs[ii].first * 3 + dd] += delef_[pairs[ii].second * 3 + dd];
    }
  }
  // add ele contrinution
  dfcorr_ = dfcorr_2;
  // for (int ii = 0; ii < nloc; ++ii){
  //   for (int dd = 0; dd < 3; ++dd){
  //     dfcorr_[ii*3+dd] += delef_[ii*3+dd];
  //   }
  // }
  for (int ii = 0; ii < nloc_real; ++ii) {
    int oii = real_bkw_map[ii];
    for (int dd = 0; dd < 3; ++dd) {
      dfcorr_[oii * 3 + dd] += delef_[oii * 3 + dd];
    }
  }
  dvcorr_ = dvcorr;
}

template void DipoleChargeModifier::compute<double>(
    std::vector<double>& dfcorr_,
    std::vector<double>& dvcorr_,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<double>& delef_,
    const int nghost,
    const InputNlist& lmp_list);

template void DipoleChargeModifier::compute<float>(
    std::vector<float>& dfcorr_,
    std::vector<float>& dvcorr_,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<std::pair<int, int>>& pairs,
    const std::vector<float>& delef_,
    const int nghost,
    const InputNlist& lmp_list);

void DipoleChargeModifier::print_summary(const std::string& pre) const {
  deepmd::print_summary(pre);
}
