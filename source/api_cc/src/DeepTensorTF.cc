// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_TENSORFLOW
#include "DeepTensorTF.h"

using namespace deepmd;
using namespace tensorflow;

DeepTensorTF::DeepTensorTF() : inited(false), graph_def(new GraphDef()) {}

DeepTensorTF::DeepTensorTF(const std::string &model,
                           const int &gpu_rank,
                           const std::string &name_scope_)
    : inited(false), name_scope(name_scope_), graph_def(new GraphDef()) {
  try {
    init(model, gpu_rank, name_scope_);
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    delete graph_def;
    throw;
  }
}

DeepTensorTF::~DeepTensorTF() { delete graph_def; }

void DeepTensorTF::init(const std::string &model,
                        const int &gpu_rank,
                        const std::string &name_scope_) {
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
  try {
    model_version = get_scalar<STRINGTYPE>("model_attr/model_version");
  } catch (deepmd::tf_exception &e) {
    // no model version defined in old models
    model_version = "0.0";
  }
  if (!model_compatable(model_version)) {
    throw deepmd::deepmd_exception(
        "incompatable model: version " + model_version +
        " in graph, but version " + global_model_version +
        " supported "
        "See https://deepmd.rtfd.io/compatability/ for details.");
  }
  dtype = session_get_dtype(session, "descrpt_attr/rcut");
  if (dtype == tensorflow::DT_DOUBLE) {
    rcut = get_scalar<double>("descrpt_attr/rcut");
  } else {
    rcut = get_scalar<float>("descrpt_attr/rcut");
  }
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  odim = get_scalar<int>("model_attr/output_dim");
  get_vector<int>(sel_type, "model_attr/sel_type");
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  inited = true;
}

template <class VT>
VT DeepTensorTF::get_scalar(const std::string &name) const {
  return session_get_scalar<VT>(session, name, name_scope);
}

template <class VT>
void DeepTensorTF::get_vector(std::vector<VT> &vec,
                              const std::string &name) const {
  session_get_vector<VT>(vec, session, name, name_scope);
}

template <typename MODELTYPE, typename VALUETYPE>
void DeepTensorTF::run_model(
    std::vector<VALUETYPE> &d_tensor_,
    Session *session,
    const std::vector<std::pair<std::string, Tensor>> &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost) {
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  if (nloc == 0) {
    // return empty
    d_tensor_.clear();
    return;
  }

  std::vector<Tensor> output_tensors;
  deepmd::check_status(
      session->Run(input_tensors, {name_prefix(name_scope) + "o_" + model_type},
                   {}, &output_tensors));

  Tensor output_t = output_tensors[0];
  // Yixiao: newer model may output rank 2 tensor [nframes x (natoms x noutdim)]
  // assert (output_t.dims() == 1), "dim of output tensor should be 1";
  auto ot = output_t.flat<MODELTYPE>();
  // this is an Eigen Tensor
  int o_size = ot.size();

  std::vector<VALUETYPE> d_tensor(o_size);
  for (unsigned ii = 0; ii < o_size; ++ii) {
    d_tensor[ii] = ot(ii);
  }
  // now we map the type-sorted sel-atom tensor back to original order
  // first we have to get the type-sorted select map
  std::vector<int> sel_srt = sel_fwd;
  select_map<int>(sel_srt, sel_fwd, atommap.get_fwd_map(), 1);
  // remove those -1 that correspond to discarded atoms
  std::remove(sel_srt.begin(), sel_srt.end(), -1);
  // now map the tensor back
  d_tensor_.resize(o_size);
  select_map<VALUETYPE>(d_tensor_, d_tensor, sel_srt, odim);
}

template void DeepTensorTF::run_model<double, double>(
    std::vector<double> &d_tensor_,
    Session *session,
    const std::vector<std::pair<std::string, Tensor>> &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);
template void DeepTensorTF::run_model<float, double>(
    std::vector<double> &d_tensor_,
    Session *session,
    const std::vector<std::pair<std::string, Tensor>> &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);
template void DeepTensorTF::run_model<double, float>(
    std::vector<float> &d_tensor_,
    Session *session,
    const std::vector<std::pair<std::string, Tensor>> &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);
template void DeepTensorTF::run_model<float, float>(
    std::vector<float> &d_tensor_,
    Session *session,
    const std::vector<std::pair<std::string, Tensor>> &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);

template <typename MODELTYPE, typename VALUETYPE>
void DeepTensorTF::run_model(
    std::vector<VALUETYPE> &dglobal_tensor_,
    std::vector<VALUETYPE> &dforce_,
    std::vector<VALUETYPE> &dvirial_,
    std::vector<VALUETYPE> &datom_tensor_,
    std::vector<VALUETYPE> &datom_virial_,
    tensorflow::Session *session,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>
        &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost) {
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  unsigned nsel = nloc - std::count(sel_fwd.begin(), sel_fwd.end(), -1);
  if (nloc == 0) {
    // return empty
    dglobal_tensor_.clear();
    dforce_.clear();
    dvirial_.clear();
    return;
  }

  std::vector<Tensor> output_tensors;
  deepmd::check_status(
      session->Run(input_tensors,
                   {name_prefix(name_scope) + "o_global_" + model_type,
                    name_prefix(name_scope) + "o_force",
                    name_prefix(name_scope) + "o_virial",
                    name_prefix(name_scope) + "o_" + model_type,
                    name_prefix(name_scope) + "o_atom_virial"},
                   {}, &output_tensors));

  Tensor output_gt = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_v = output_tensors[2];
  Tensor output_at = output_tensors[3];
  Tensor output_av = output_tensors[4];
  // this is the new model, output has to be rank 2 tensor
  assert(output_gt.dims() == 2 && "dim of output tensor should be 2");
  assert(output_f.dims() == 2 && "dim of output tensor should be 2");
  assert(output_v.dims() == 2 && "dim of output tensor should be 2");
  assert(output_at.dims() == 2 && "dim of output tensor should be 2");
  assert(output_av.dims() == 2 && "dim of output tensor should be 2");
  // also check the tensor shapes
  assert(output_gt.dim_size(0) == 1 && "nframes should match");
  assert(output_gt.dim_size(1) == odim &&
         "dof of global tensor should be odim");
  assert(output_f.dim_size(0) == 1 && "nframes should match");
  assert(output_f.dim_size(1) == odim * nall * 3 &&
         "dof of force should be odim * nall * 3");
  assert(output_v.dim_size(0) == 1 && "nframes should match");
  assert(output_v.dim_size(1) == odim * 9 &&
         "dof of virial should be odim * 9");
  assert(output_at.dim_size(0) == 1 && "nframes should match");
  assert(output_at.dim_size(1) == nsel * odim &&
         "dof of atomic tensor should be nsel * odim");
  assert(output_av.dim_size(0) == 1 && "nframes should match");
  assert(output_av.dim_size(1) == odim * nall * 9 &&
         "dof of atomic virial should be odim * nall * 9");

  auto ogt = output_gt.flat<ENERGYTYPE>();
  auto of = output_f.flat<MODELTYPE>();
  auto ov = output_v.flat<MODELTYPE>();
  auto oat = output_at.flat<MODELTYPE>();
  auto oav = output_av.flat<MODELTYPE>();

  // global tensor
  dglobal_tensor_.resize(odim);
  for (unsigned ii = 0; ii < odim; ++ii) {
    dglobal_tensor_[ii] = ogt(ii);
  }

  // component-wise force
  std::vector<VALUETYPE> dforce(3 * static_cast<size_t>(nall) * odim);
  for (unsigned ii = 0; ii < odim * nall * 3; ++ii) {
    dforce[ii] = of(ii);
  }
  dforce_ = dforce;
  for (unsigned dd = 0; dd < odim; ++dd) {
    atommap.backward<VALUETYPE>(dforce_.begin() + (dd * nall * 3),
                                dforce.begin() + (dd * nall * 3), 3);
  }

  // component-wise virial
  dvirial_.resize(static_cast<size_t>(odim) * 9);
  for (unsigned ii = 0; ii < odim * 9; ++ii) {
    dvirial_[ii] = ov(ii);
  }

  // atomic tensor
  std::vector<VALUETYPE> datom_tensor(static_cast<size_t>(nsel) * odim);
  for (unsigned ii = 0; ii < nsel * odim; ++ii) {
    datom_tensor[ii] = oat(ii);
  }
  std::vector<int> sel_srt = sel_fwd;
  select_map<int>(sel_srt, sel_fwd, atommap.get_fwd_map(), 1);
  std::remove(sel_srt.begin(), sel_srt.end(), -1);
  datom_tensor_.resize(static_cast<size_t>(nsel) * odim);
  select_map<VALUETYPE>(datom_tensor_, datom_tensor, sel_srt, odim);

  // component-wise atomic virial
  std::vector<VALUETYPE> datom_virial(9 * static_cast<size_t>(nall) * odim);
  for (unsigned ii = 0; ii < odim * nall * 9; ++ii) {
    datom_virial[ii] = oav(ii);
  }
  datom_virial_ = datom_virial;
  for (unsigned dd = 0; dd < odim; ++dd) {
    atommap.backward<VALUETYPE>(datom_virial_.begin() + (dd * nall * 9),
                                datom_virial.begin() + (dd * nall * 9), 9);
  }
}

template void DeepTensorTF::run_model<double, double>(
    std::vector<double> &dglobal_tensor_,
    std::vector<double> &dforce_,
    std::vector<double> &dvirial_,
    std::vector<double> &datom_tensor_,
    std::vector<double> &datom_virial_,
    tensorflow::Session *session,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>
        &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);
template void DeepTensorTF::run_model<float, double>(
    std::vector<double> &dglobal_tensor_,
    std::vector<double> &dforce_,
    std::vector<double> &dvirial_,
    std::vector<double> &datom_tensor_,
    std::vector<double> &datom_virial_,
    tensorflow::Session *session,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>
        &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);

template void DeepTensorTF::run_model<double, float>(
    std::vector<float> &dglobal_tensor_,
    std::vector<float> &dforce_,
    std::vector<float> &dvirial_,
    std::vector<float> &datom_tensor_,
    std::vector<float> &datom_virial_,
    tensorflow::Session *session,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>
        &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);

template void DeepTensorTF::run_model<float, float>(
    std::vector<float> &dglobal_tensor_,
    std::vector<float> &dforce_,
    std::vector<float> &dvirial_,
    std::vector<float> &datom_tensor_,
    std::vector<float> &datom_virial_,
    tensorflow::Session *session,
    const std::vector<std::pair<std::string, tensorflow::Tensor>>
        &input_tensors,
    const AtomMap &atommap,
    const std::vector<int> &sel_fwd,
    const int nghost);

template <typename VALUETYPE>
void DeepTensorTF::compute(std::vector<VALUETYPE> &dtensor_,
                           const std::vector<VALUETYPE> &dcoord_,
                           const std::vector<int> &datype_,
                           const std::vector<VALUETYPE> &dbox) {
  int nall = datype_.size();
  std::vector<VALUETYPE> dcoord, aparam, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_, 0,
                          ntypes, 1, 0, nall);
  compute_inner(dtensor_, dcoord, datype, dbox);
}

template void DeepTensorTF::compute<double>(std::vector<double> &dtensor_,
                                            const std::vector<double> &dcoord_,
                                            const std::vector<int> &datype_,
                                            const std::vector<double> &dbox);

template void DeepTensorTF::compute<float>(std::vector<float> &dtensor_,
                                           const std::vector<float> &dcoord_,
                                           const std::vector<int> &datype_,
                                           const std::vector<float> &dbox);

template <typename VALUETYPE>
void DeepTensorTF::compute(std::vector<VALUETYPE> &dtensor_,
                           const std::vector<VALUETYPE> &dcoord_,
                           const std::vector<int> &datype_,
                           const std::vector<VALUETYPE> &dbox,
                           const int nghost,
                           const InputNlist &lmp_list) {
  int nall = datype_.size();
  std::vector<VALUETYPE> dcoord, dforce, datom_virial, aparam, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_,
                          nghost, ntypes, 1, 0, nall);
  // internal nlist
  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(lmp_list);
  nlist_data.shuffle_exclude_empty(fwd_map);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);
  compute_inner(dtensor_, dcoord, datype, dbox, nghost_real, nlist);
}

template void DeepTensorTF::compute<double>(std::vector<double> &dtensor_,
                                            const std::vector<double> &dcoord_,
                                            const std::vector<int> &datype_,
                                            const std::vector<double> &dbox,
                                            const int nghost,
                                            const InputNlist &lmp_list);

template void DeepTensorTF::compute<float>(std::vector<float> &dtensor_,
                                           const std::vector<float> &dcoord_,
                                           const std::vector<int> &datype_,
                                           const std::vector<float> &dbox,
                                           const int nghost,
                                           const InputNlist &lmp_list);

template <typename VALUETYPE>
void DeepTensorTF::compute(std::vector<VALUETYPE> &dglobal_tensor_,
                           std::vector<VALUETYPE> &dforce_,
                           std::vector<VALUETYPE> &dvirial_,
                           std::vector<VALUETYPE> &datom_tensor_,
                           std::vector<VALUETYPE> &datom_virial_,
                           const std::vector<VALUETYPE> &dcoord_,
                           const std::vector<int> &datype_,
                           const std::vector<VALUETYPE> &dbox) {
  int nall = datype_.size();
  std::vector<VALUETYPE> dcoord, dforce, datom_virial, aparam, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_, 0,
                          ntypes, 1, 0, nall);
  assert(nghost_real == 0);
  // resize to nall_real
  dcoord.resize(bkw_map.size() * 3);
  datype.resize(bkw_map.size());
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  compute_inner(dglobal_tensor_, dforce, dvirial_, datom_tensor_, datom_virial,
                dcoord, datype, dbox);
  // bkw map
  dforce_.resize(odim * fwd_map.size() * 3);
  for (int kk = 0; kk < odim; ++kk) {
    select_map<VALUETYPE>(dforce_.begin() + kk * fwd_map.size() * 3,
                          dforce.begin() + kk * bkw_map.size() * 3, bkw_map, 3);
  }
  datom_virial_.resize(odim * fwd_map.size() * 9);
  for (int kk = 0; kk < odim; ++kk) {
    select_map<VALUETYPE>(datom_virial_.begin() + kk * fwd_map.size() * 9,
                          datom_virial.begin() + kk * bkw_map.size() * 9,
                          bkw_map, 9);
  }
}

template void DeepTensorTF::compute<double>(
    std::vector<double> &dglobal_tensor_,
    std::vector<double> &dforce_,
    std::vector<double> &dvirial_,
    std::vector<double> &datom_tensor_,
    std::vector<double> &datom_virial_,
    const std::vector<double> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<double> &dbox);

template void DeepTensorTF::compute<float>(std::vector<float> &dglobal_tensor_,
                                           std::vector<float> &dforce_,
                                           std::vector<float> &dvirial_,
                                           std::vector<float> &datom_tensor_,
                                           std::vector<float> &datom_virial_,
                                           const std::vector<float> &dcoord_,
                                           const std::vector<int> &datype_,
                                           const std::vector<float> &dbox);

template <typename VALUETYPE>
void DeepTensorTF::compute(std::vector<VALUETYPE> &dglobal_tensor_,
                           std::vector<VALUETYPE> &dforce_,
                           std::vector<VALUETYPE> &dvirial_,
                           std::vector<VALUETYPE> &datom_tensor_,
                           std::vector<VALUETYPE> &datom_virial_,
                           const std::vector<VALUETYPE> &dcoord_,
                           const std::vector<int> &datype_,
                           const std::vector<VALUETYPE> &dbox,
                           const int nghost,
                           const InputNlist &lmp_list) {
  int nall = datype_.size();
  std::vector<VALUETYPE> dcoord, dforce, datom_virial, aparam, aparam_;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_,
                          nghost, ntypes, 1, 0, nall);
  // internal nlist
  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(lmp_list);
  nlist_data.shuffle_exclude_empty(fwd_map);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);
  compute_inner(dglobal_tensor_, dforce, dvirial_, datom_tensor_, datom_virial,
                dcoord, datype, dbox, nghost_real, nlist);
  // bkw map
  dforce_.resize(odim * fwd_map.size() * 3);
  for (int kk = 0; kk < odim; ++kk) {
    select_map<VALUETYPE>(dforce_.begin() + kk * fwd_map.size() * 3,
                          dforce.begin() + kk * bkw_map.size() * 3, bkw_map, 3);
  }
  datom_virial_.resize(odim * fwd_map.size() * 9);
  for (int kk = 0; kk < odim; ++kk) {
    select_map<VALUETYPE>(datom_virial_.begin() + kk * fwd_map.size() * 9,
                          datom_virial.begin() + kk * bkw_map.size() * 9,
                          bkw_map, 9);
  }
}

template void DeepTensorTF::compute<double>(
    std::vector<double> &dglobal_tensor_,
    std::vector<double> &dforce_,
    std::vector<double> &dvirial_,
    std::vector<double> &datom_tensor_,
    std::vector<double> &datom_virial_,
    const std::vector<double> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<double> &dbox,
    const int nghost,
    const InputNlist &lmp_list);

template void DeepTensorTF::compute<float>(std::vector<float> &dglobal_tensor_,
                                           std::vector<float> &dforce_,
                                           std::vector<float> &dvirial_,
                                           std::vector<float> &datom_tensor_,
                                           std::vector<float> &datom_virial_,
                                           const std::vector<float> &dcoord_,
                                           const std::vector<int> &datype_,
                                           const std::vector<float> &dbox,
                                           const int nghost,
                                           const InputNlist &lmp_list);

template <typename VALUETYPE>
void DeepTensorTF::compute_inner(std::vector<VALUETYPE> &dtensor_,
                                 const std::vector<VALUETYPE> &dcoord_,
                                 const std::vector<int> &datype_,
                                 const std::vector<VALUETYPE> &dbox) {
  int nall = dcoord_.size() / 3;
  int nloc = nall;
  AtomMap atommap(datype_.begin(), datype_.begin() + nloc);
  assert(nloc == atommap.get_type().size());

  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, 0, sel_type);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(
        input_tensors, dcoord_, ntypes, datype_, dbox, cell_size,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap,
        name_scope);
    assert(ret == nloc);
    run_model<double>(dtensor_, session, input_tensors, atommap, sel_fwd);
  } else {
    int ret = session_input_tensors<float>(
        input_tensors, dcoord_, ntypes, datype_, dbox, cell_size,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap,
        name_scope);
    assert(ret == nloc);
    run_model<float>(dtensor_, session, input_tensors, atommap, sel_fwd);
  }
}

template void DeepTensorTF::compute_inner<double>(
    std::vector<double> &dtensor_,
    const std::vector<double> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<double> &dbox);

template void DeepTensorTF::compute_inner<float>(
    std::vector<float> &dtensor_,
    const std::vector<float> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<float> &dbox);

template <typename VALUETYPE>
void DeepTensorTF::compute_inner(std::vector<VALUETYPE> &dtensor_,
                                 const std::vector<VALUETYPE> &dcoord_,
                                 const std::vector<int> &datype_,
                                 const std::vector<VALUETYPE> &dbox,
                                 const int nghost,
                                 const InputNlist &nlist_) {
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  AtomMap atommap(datype_.begin(), datype_.begin() + nloc);
  assert(nloc == atommap.get_type().size());

  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, nghost,
                 sel_type);
  sel_fwd.resize(nloc);

  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(nlist_);
  nlist_data.shuffle(atommap);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(
        input_tensors, dcoord_, ntypes, datype_, dbox, nlist,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, nghost, 0,
        name_scope);
    assert(nloc == ret);
    run_model<double>(dtensor_, session, input_tensors, atommap, sel_fwd,
                      nghost);
  } else {
    int ret = session_input_tensors<float>(
        input_tensors, dcoord_, ntypes, datype_, dbox, nlist,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, nghost, 0,
        name_scope);
    assert(nloc == ret);
    run_model<float>(dtensor_, session, input_tensors, atommap, sel_fwd,
                     nghost);
  }
}

template void DeepTensorTF::compute_inner<double>(
    std::vector<double> &dtensor_,
    const std::vector<double> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<double> &dbox,
    const int nghost,
    const InputNlist &nlist_);

template void DeepTensorTF::compute_inner<float>(
    std::vector<float> &dtensor_,
    const std::vector<float> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<float> &dbox,
    const int nghost,
    const InputNlist &nlist_);

template <typename VALUETYPE>
void DeepTensorTF::compute_inner(std::vector<VALUETYPE> &dglobal_tensor_,
                                 std::vector<VALUETYPE> &dforce_,
                                 std::vector<VALUETYPE> &dvirial_,
                                 std::vector<VALUETYPE> &datom_tensor_,
                                 std::vector<VALUETYPE> &datom_virial_,
                                 const std::vector<VALUETYPE> &dcoord_,
                                 const std::vector<int> &datype_,
                                 const std::vector<VALUETYPE> &dbox) {
  int nall = dcoord_.size() / 3;
  int nloc = nall;
  AtomMap atommap(datype_.begin(), datype_.begin() + nloc);
  assert(nloc == atommap.get_type().size());

  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, 0, sel_type);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(
        input_tensors, dcoord_, ntypes, datype_, dbox, cell_size,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap,
        name_scope);
    assert(ret == nloc);
    run_model<double>(dglobal_tensor_, dforce_, dvirial_, datom_tensor_,
                      datom_virial_, session, input_tensors, atommap, sel_fwd);
  } else {
    int ret = session_input_tensors<float>(
        input_tensors, dcoord_, ntypes, datype_, dbox, cell_size,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap,
        name_scope);
    assert(ret == nloc);
    run_model<float>(dglobal_tensor_, dforce_, dvirial_, datom_tensor_,
                     datom_virial_, session, input_tensors, atommap, sel_fwd);
  }
}

template void DeepTensorTF::compute_inner<double>(
    std::vector<double> &dglobal_tensor_,
    std::vector<double> &dforce_,
    std::vector<double> &dvirial_,
    std::vector<double> &datom_tensor_,
    std::vector<double> &datom_virial_,
    const std::vector<double> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<double> &dbox);

template void DeepTensorTF::compute_inner<float>(
    std::vector<float> &dglobal_tensor_,
    std::vector<float> &dforce_,
    std::vector<float> &dvirial_,
    std::vector<float> &datom_tensor_,
    std::vector<float> &datom_virial_,
    const std::vector<float> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<float> &dbox);

template <typename VALUETYPE>
void DeepTensorTF::compute_inner(std::vector<VALUETYPE> &dglobal_tensor_,
                                 std::vector<VALUETYPE> &dforce_,
                                 std::vector<VALUETYPE> &dvirial_,
                                 std::vector<VALUETYPE> &datom_tensor_,
                                 std::vector<VALUETYPE> &datom_virial_,
                                 const std::vector<VALUETYPE> &dcoord_,
                                 const std::vector<int> &datype_,
                                 const std::vector<VALUETYPE> &dbox,
                                 const int nghost,
                                 const InputNlist &nlist_) {
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  AtomMap atommap(datype_.begin(), datype_.begin() + nloc);
  assert(nloc == atommap.get_type().size());

  std::vector<int> sel_fwd, sel_bkw;
  int nghost_sel;
  // this gives the raw selection map, will pass to run model
  select_by_type(sel_fwd, sel_bkw, nghost_sel, dcoord_, datype_, nghost,
                 sel_type);
  sel_fwd.resize(nloc);

  NeighborListData nlist_data;
  nlist_data.copy_from_nlist(nlist_);
  nlist_data.shuffle(atommap);
  InputNlist nlist;
  nlist_data.make_inlist(nlist);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(
        input_tensors, dcoord_, ntypes, datype_, dbox, nlist,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, nghost, 0,
        name_scope);
    assert(nloc == ret);
    run_model<double>(dglobal_tensor_, dforce_, dvirial_, datom_tensor_,
                      datom_virial_, session, input_tensors, atommap, sel_fwd,
                      nghost);
  } else {
    int ret = session_input_tensors<float>(
        input_tensors, dcoord_, ntypes, datype_, dbox, nlist,
        std::vector<VALUETYPE>(), std::vector<VALUETYPE>(), atommap, nghost, 0,
        name_scope);
    assert(nloc == ret);
    run_model<float>(dglobal_tensor_, dforce_, dvirial_, datom_tensor_,
                     datom_virial_, session, input_tensors, atommap, sel_fwd,
                     nghost);
  }
}

template void DeepTensorTF::compute_inner<double>(
    std::vector<double> &dglobal_tensor_,
    std::vector<double> &dforce_,
    std::vector<double> &dvirial_,
    std::vector<double> &datom_tensor_,
    std::vector<double> &datom_virial_,
    const std::vector<double> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<double> &dbox,
    const int nghost,
    const InputNlist &nlist_);

template void DeepTensorTF::compute_inner<float>(
    std::vector<float> &dglobal_tensor_,
    std::vector<float> &dforce_,
    std::vector<float> &dvirial_,
    std::vector<float> &datom_tensor_,
    std::vector<float> &datom_virial_,
    const std::vector<float> &dcoord_,
    const std::vector<int> &datype_,
    const std::vector<float> &dbox,
    const int nghost,
    const InputNlist &nlist_);

void DeepTensorTF::get_type_map(std::string &type_map) {
  type_map = get_scalar<STRINGTYPE>("model_attr/tmap");
}

void DeepTensorTF::computew(std::vector<double> &global_tensor,
                            std::vector<double> &force,
                            std::vector<double> &virial,
                            std::vector<double> &atom_tensor,
                            std::vector<double> &atom_virial,
                            const std::vector<double> &coord,
                            const std::vector<int> &atype,
                            const std::vector<double> &box,
                            const bool request_deriv) {
  if (request_deriv) {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box);
  } else {
    compute(global_tensor, coord, atype, box);
    force.clear();
    virial.clear();
    atom_tensor.clear();
    atom_virial.clear();
  }
}
void DeepTensorTF::computew(std::vector<float> &global_tensor,
                            std::vector<float> &force,
                            std::vector<float> &virial,
                            std::vector<float> &atom_tensor,
                            std::vector<float> &atom_virial,
                            const std::vector<float> &coord,
                            const std::vector<int> &atype,
                            const std::vector<float> &box,
                            const bool request_deriv) {
  if (request_deriv) {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box);
  } else {
    compute(global_tensor, coord, atype, box);
    force.clear();
    virial.clear();
    atom_tensor.clear();
    atom_virial.clear();
  }
}

void DeepTensorTF::computew(std::vector<double> &global_tensor,
                            std::vector<double> &force,
                            std::vector<double> &virial,
                            std::vector<double> &atom_tensor,
                            std::vector<double> &atom_virial,
                            const std::vector<double> &coord,
                            const std::vector<int> &atype,
                            const std::vector<double> &box,
                            const int nghost,
                            const InputNlist &inlist,
                            const bool request_deriv) {
  if (request_deriv) {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box, nghost, inlist);
  } else {
    compute(global_tensor, coord, atype, box, nghost, inlist);
    force.clear();
    virial.clear();
    atom_tensor.clear();
    atom_virial.clear();
  }
}
void DeepTensorTF::computew(std::vector<float> &global_tensor,
                            std::vector<float> &force,
                            std::vector<float> &virial,
                            std::vector<float> &atom_tensor,
                            std::vector<float> &atom_virial,
                            const std::vector<float> &coord,
                            const std::vector<int> &atype,
                            const std::vector<float> &box,
                            const int nghost,
                            const InputNlist &inlist,
                            const bool request_deriv) {
  if (request_deriv) {
    compute(global_tensor, force, virial, atom_tensor, atom_virial, coord,
            atype, box, nghost, inlist);
  } else {
    compute(global_tensor, coord, atype, box, nghost, inlist);
    force.clear();
    virial.clear();
    atom_tensor.clear();
    atom_virial.clear();
  }
}
#endif
