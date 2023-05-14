#include "DeepPot.h"

#include <stdexcept>

#include "AtomMap.h"
#include "device.h"

using namespace tensorflow;
using namespace deepmd;

// start multiple frames

template <typename MODELTYPE, typename VALUETYPE>
static void run_model(
    std::vector<ENERGYTYPE>& dener,
    std::vector<VALUETYPE>& dforce_,
    std::vector<VALUETYPE>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost = 0) {
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  dener.resize(nframes);
  if (nloc == 0) {
    // no backward map needed
    // dforce of size nall * 3
    dforce_.resize(nframes * nall * 3);
    fill(dforce_.begin(), dforce_.end(), (VALUETYPE)0.0);
    // dvirial of size 9
    dvirial.resize(nframes * 9);
    fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.0);
    return;
  }

  std::vector<Tensor> output_tensors;
  check_status(session->Run(
      input_tensors, {"o_energy", "o_force", "o_atom_energy", "o_atom_virial"},
      {}, &output_tensors));

  Tensor output_e = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_av = output_tensors[3];

  auto oe = output_e.flat<ENERGYTYPE>();
  auto of = output_f.flat<MODELTYPE>();
  auto oav = output_av.flat<MODELTYPE>();

  std::vector<VALUETYPE> dforce(nframes * 3 * nall);
  dvirial.resize(nframes * 9);
  for (int ii = 0; ii < nframes; ++ii) {
    dener[ii] = oe(ii);
  }
  for (unsigned ii = 0; ii < nframes * nall * 3; ++ii) {
    dforce[ii] = of(ii);
  }
  // set dvirial to zero, prevent input vector is not zero (#1123)
  std::fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.);
  for (int kk = 0; kk < nframes; ++kk) {
    for (int ii = 0; ii < nall; ++ii) {
      dvirial[kk * 9 + 0] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 0);
      dvirial[kk * 9 + 1] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 1);
      dvirial[kk * 9 + 2] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 2);
      dvirial[kk * 9 + 3] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 3);
      dvirial[kk * 9 + 4] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 4);
      dvirial[kk * 9 + 5] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 5);
      dvirial[kk * 9 + 6] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 6);
      dvirial[kk * 9 + 7] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 7);
      dvirial[kk * 9 + 8] += (VALUETYPE)1.0 * oav(kk * nall * 9 + 9 * ii + 8);
    }
  }
  dforce_ = dforce;
  atommap.backward<VALUETYPE>(dforce_.begin(), dforce.begin(), 3, nframes,
                              nall);
}

template void run_model<double, double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<double, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template <typename MODELTYPE, typename VALUETYPE>
static void run_model(
    std::vector<ENERGYTYPE>& dener,
    std::vector<VALUETYPE>& dforce_,
    std::vector<VALUETYPE>& dvirial,
    std::vector<VALUETYPE>& datom_energy_,
    std::vector<VALUETYPE>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost = 0) {
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  dener.resize(nframes);
  if (nloc == 0) {
    // no backward map needed
    // dforce of size nall * 3
    dforce_.resize(nframes * nall * 3);
    fill(dforce_.begin(), dforce_.end(), (VALUETYPE)0.0);
    // dvirial of size 9
    dvirial.resize(nframes * 9);
    fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.0);
    // datom_energy_ of size nall
    datom_energy_.resize(nframes * nall);
    fill(datom_energy_.begin(), datom_energy_.end(), (VALUETYPE)0.0);
    // datom_virial_ of size nall * 9
    datom_virial_.resize(nframes * nall * 9);
    fill(datom_virial_.begin(), datom_virial_.end(), (VALUETYPE)0.0);
    return;
  }
  std::vector<Tensor> output_tensors;

  check_status(session->Run(
      input_tensors, {"o_energy", "o_force", "o_atom_energy", "o_atom_virial"},
      {}, &output_tensors));

  Tensor output_e = output_tensors[0];
  Tensor output_f = output_tensors[1];
  Tensor output_ae = output_tensors[2];
  Tensor output_av = output_tensors[3];

  auto oe = output_e.flat<ENERGYTYPE>();
  auto of = output_f.flat<MODELTYPE>();
  auto oae = output_ae.flat<MODELTYPE>();
  auto oav = output_av.flat<MODELTYPE>();

  std::vector<VALUETYPE> dforce(nframes * 3 * nall);
  std::vector<VALUETYPE> datom_energy(nframes * nall, 0);
  std::vector<VALUETYPE> datom_virial(nframes * 9 * nall);
  dvirial.resize(nframes * 9);
  for (int ii = 0; ii < nframes; ++ii) {
    dener[ii] = oe(ii);
  }
  for (int ii = 0; ii < nframes * nall * 3; ++ii) {
    dforce[ii] = of(ii);
  }
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nloc; ++jj) {
      datom_energy[ii * nall + jj] = oae(ii * nloc + jj);
    }
  }
  for (int ii = 0; ii < nframes * nall * 9; ++ii) {
    datom_virial[ii] = oav(ii);
  }
  // set dvirial to zero, prevent input vector is not zero (#1123)
  std::fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.);
  for (int kk = 0; kk < nframes; ++kk) {
    for (int ii = 0; ii < nall; ++ii) {
      dvirial[kk * 9 + 0] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 0];
      dvirial[kk * 9 + 1] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 1];
      dvirial[kk * 9 + 2] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 2];
      dvirial[kk * 9 + 3] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 3];
      dvirial[kk * 9 + 4] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 4];
      dvirial[kk * 9 + 5] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 5];
      dvirial[kk * 9 + 6] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 6];
      dvirial[kk * 9 + 7] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 7];
      dvirial[kk * 9 + 8] +=
          (VALUETYPE)1.0 * datom_virial[kk * nall * 9 + 9 * ii + 8];
    }
  }
  dforce_ = dforce;
  datom_energy_ = datom_energy;
  datom_virial_ = datom_virial;
  atommap.backward<VALUETYPE>(dforce_.begin(), dforce.begin(), 3, nframes,
                              nall);
  atommap.backward<VALUETYPE>(datom_energy_.begin(), datom_energy.begin(), 1,
                              nframes, nall);
  atommap.backward<VALUETYPE>(datom_virial_.begin(), datom_virial.begin(), 9,
                              nframes, nall);
}

template void run_model<double, double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

template void run_model<double, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

template void run_model<float, double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

template void run_model<float, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

// end multiple frames

// start single frame

template <typename MODELTYPE, typename VALUETYPE>
static void run_model(
    ENERGYTYPE& dener,
    std::vector<VALUETYPE>& dforce_,
    std::vector<VALUETYPE>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes = 1,
    const int nghost = 0) {
  assert(nframes == 1);
  std::vector<ENERGYTYPE> dener_(1);
  // call multi-frame version
  run_model<MODELTYPE, VALUETYPE>(dener_, dforce_, dvirial, session,
                                  input_tensors, atommap, nframes, nghost);
  dener = dener_[0];
}

template void run_model<double, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<double, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template <typename MODELTYPE, typename VALUETYPE>
static void run_model(
    ENERGYTYPE& dener,
    std::vector<VALUETYPE>& dforce_,
    std::vector<VALUETYPE>& dvirial,
    std::vector<VALUETYPE>& datom_energy_,
    std::vector<VALUETYPE>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes = 1,
    const int& nghost = 0) {
  assert(nframes == 1);
  std::vector<ENERGYTYPE> dener_(1);
  // call multi-frame version
  run_model<MODELTYPE, VALUETYPE>(dener_, dforce_, dvirial, datom_energy_,
                                  datom_virial_, session, input_tensors,
                                  atommap, nframes, nghost);
  dener = dener_[0];
}

template void run_model<double, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

template void run_model<double, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

template void run_model<float, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

template void run_model<float, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    Session* session,
    const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int& nframes,
    const int& nghost);

// end single frame

DeepPot::DeepPot()
    : inited(false), init_nbor(false), graph_def(new GraphDef()) {}

DeepPot::DeepPot(const std::string& model,
                 const int& gpu_rank,
                 const std::string& file_content)
    : inited(false), init_nbor(false), graph_def(new GraphDef()) {
  init(model, gpu_rank, file_content);
}

DeepPot::~DeepPot() { delete graph_def; }

void DeepPot::init(const std::string& model,
                   const int& gpu_rank,
                   const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  SessionOptions options;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  deepmd::load_op_library();

  if (file_content.size() == 0)
    check_status(ReadBinaryProto(Env::Default(), model, graph_def));
  else
    (*graph_def).ParseFromString(file_content);
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
  check_status(NewSession(options, &session));
  check_status(session->Create(*graph_def));
  try {
    model_version = get_scalar<STRINGTYPE>("model_attr/model_version");
  } catch (deepmd::tf_exception& e) {
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
  try {
    ntypes_spin = get_scalar<int>("spin_attr/ntypes_spin");
  } catch (deepmd::deepmd_exception) {
    ntypes_spin = 0;
  }
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  daparam = get_scalar<int>("fitting_attr/daparam");
  if (dfparam < 0) dfparam = 0;
  if (daparam < 0) daparam = 0;
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  inited = true;

  init_nbor = false;
}

void DeepPot::print_summary(const std::string& pre) const {
  deepmd::print_summary(pre);
}

template <class VT>
VT DeepPot::get_scalar(const std::string& name) const {
  return session_get_scalar<VT>(session, name);
}

template <typename VALUETYPE>
void DeepPot::validate_fparam_aparam(
    const int& nframes,
    const int& nloc,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam) const {
  if (fparam.size() != dfparam && fparam.size() != nframes * dfparam) {
    throw deepmd::deepmd_exception(
        "the dim of frame parameter provided is not consistent with what the "
        "model uses");
  }

  if (aparam.size() != daparam * nloc &&
      aparam.size() != nframes * daparam * nloc) {
    throw deepmd::deepmd_exception(
        "the dim of atom parameter provided is not consistent with what the "
        "model uses");
  }
}

template void DeepPot::validate_fparam_aparam<double>(
    const int& nframes,
    const int& nloc,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam) const;

template void DeepPot::validate_fparam_aparam<float>(
    const int& nframes,
    const int& nloc,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam) const;

template <typename VALUETYPE>
void DeepPot::tile_fparam_aparam(std::vector<VALUETYPE>& out_param,
                                 const int& nframes,
                                 const int& dparam,
                                 const std::vector<VALUETYPE>& param) const {
  if (param.size() == dparam) {
    out_param.resize(nframes * dparam);
    for (int ii = 0; ii < nframes; ++ii) {
      std::copy(param.begin(), param.end(), out_param.begin() + ii * dparam);
    }
  } else if (param.size() == nframes * dparam) {
    out_param = param;
  }
}

template void DeepPot::tile_fparam_aparam<double>(
    std::vector<double>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<double>& param) const;

template void DeepPot::tile_fparam_aparam<float>(
    std::vector<float>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<float>& param) const;

// ENERGYVTYPE: std::vector<ENERGYTYPE> or ENERGYTYPE

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute(ENERGYVTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam_) {
  int nall = datype_.size();
  int nframes = dcoord_.size() / nall / 3;
  int nloc = nall;
  atommap = deepmd::AtomMap(datype_.begin(), datype_.begin() + nloc);
  assert(nloc == atommap.get_type().size());
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret =
        session_input_tensors<double>(input_tensors, dcoord_, ntypes, datype_,
                                      dbox, cell_size, fparam, aparam, atommap);
    assert(ret == nloc);
    run_model<double>(dener, dforce_, dvirial, session, input_tensors, atommap,
                      nframes);
  } else {
    int ret =
        session_input_tensors<float>(input_tensors, dcoord_, ntypes, datype_,
                                     dbox, cell_size, fparam, aparam, atommap);
    assert(ret == nloc);
    run_model<float>(dener, dforce_, dvirial, session, input_tensors, atommap,
                     nframes);
  }
}

template void DeepPot::compute<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template void DeepPot::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute(ENERGYVTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const int nghost,
                      const InputNlist& lmp_list,
                      const int& ago,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam__) {
  int nall = datype_.size();
  int nframes = dcoord_.size() / nall / 3;
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam_;
  validate_fparam_aparam(nframes, nall - nghost, fparam_, aparam__);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam_, nframes, (nall - nghost) * daparam, aparam__);

  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_,
                          nghost, ntypes, nframes, daparam, nall);

  // internal nlist
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
  }
  compute_inner(dener, dforce, dvirial, dcoord, datype, dbox, nghost_real, ago,
                fparam, aparam);
  // bkw map
  dforce_.resize(nframes * fwd_map.size() * 3);
  select_map<VALUETYPE>(dforce_, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        bkw_map.size());
}

template void DeepPot::compute<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam_);

template void DeepPot::compute<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam_);

template void DeepPot::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam_);

template void DeepPot::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam_);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute_inner(ENERGYVTYPE& dener,
                            std::vector<VALUETYPE>& dforce_,
                            std::vector<VALUETYPE>& dvirial,
                            const std::vector<VALUETYPE>& dcoord_,
                            const std::vector<int>& datype_,
                            const std::vector<VALUETYPE>& dbox,
                            const int nghost,
                            const int& ago,
                            const std::vector<VALUETYPE>& fparam,
                            const std::vector<VALUETYPE>& aparam) {
  int nall = datype_.size();
  int nframes = dcoord_.size() / nall / 3;
  int nloc = nall - nghost;

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  // agp == 0 means that the LAMMPS nbor list has been updated
  if (ago == 0) {
    atommap = deepmd::AtomMap(datype_.begin(), datype_.begin() + nloc);
    assert(nloc == atommap.get_type().size());
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }
  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(input_tensors, dcoord_, ntypes,
                                            datype_, dbox, nlist, fparam,
                                            aparam, atommap, nghost, ago);
    assert(nloc == ret);
    run_model<double>(dener, dforce_, dvirial, session, input_tensors, atommap,
                      nframes, nghost);
  } else {
    int ret = session_input_tensors<float>(input_tensors, dcoord_, ntypes,
                                           datype_, dbox, nlist, fparam, aparam,
                                           atommap, nghost, ago);
    assert(nloc == ret);
    run_model<float>(dener, dforce_, dvirial, session, input_tensors, atommap,
                     nframes, nghost);
  }
}

template void DeepPot::compute_inner<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute_inner<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template void DeepPot::compute_inner<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute_inner<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute(ENERGYVTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      std::vector<VALUETYPE>& datom_energy_,
                      std::vector<VALUETYPE>& datom_virial_,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam_) {
  int nframes = dcoord_.size() / 3 / datype_.size();
  atommap = deepmd::AtomMap(datype_.begin(), datype_.end());
  int nloc = datype_.size();
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int nloc =
        session_input_tensors<double>(input_tensors, dcoord_, ntypes, datype_,
                                      dbox, cell_size, fparam, aparam, atommap);
    run_model<double>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                      session, input_tensors, atommap, nframes);
  } else {
    int nloc =
        session_input_tensors<float>(input_tensors, dcoord_, ntypes, datype_,
                                     dbox, cell_size, fparam, aparam, atommap);
    run_model<float>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                     session, input_tensors, atommap, nframes);
  }
}

template void DeepPot::compute<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template void DeepPot::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute(ENERGYVTYPE& dener,
                      std::vector<VALUETYPE>& dforce_,
                      std::vector<VALUETYPE>& dvirial,
                      std::vector<VALUETYPE>& datom_energy_,
                      std::vector<VALUETYPE>& datom_virial_,
                      const std::vector<VALUETYPE>& dcoord_,
                      const std::vector<int>& datype_,
                      const std::vector<VALUETYPE>& dbox,
                      const int nghost,
                      const InputNlist& lmp_list,
                      const int& ago,
                      const std::vector<VALUETYPE>& fparam_,
                      const std::vector<VALUETYPE>& aparam__) {
  int nall = datype_.size();
  int nframes = dcoord_.size() / 3 / nall;
  int nloc = nall - nghost;
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam_;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam__);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam_, nframes, nloc * daparam, aparam__);
  std::vector<std::pair<std::string, Tensor>> input_tensors;
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_,
                          nghost, ntypes, nframes, daparam, nall);

  if (ago == 0) {
    atommap = deepmd::AtomMap(datype.begin(), datype.begin() + nloc_real);
    assert(nloc_real == atommap.get_type().size());

    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(input_tensors, dcoord, ntypes,
                                            datype, dbox, nlist, fparam, aparam,
                                            atommap, nghost_real, ago);
    assert(nloc_real == ret);
    run_model<double>(dener, dforce, dvirial, datom_energy, datom_virial,
                      session, input_tensors, atommap, nframes, nghost_real);
  } else {
    int ret = session_input_tensors<float>(input_tensors, dcoord, ntypes,
                                           datype, dbox, nlist, fparam, aparam,
                                           atommap, nghost_real, ago);
    assert(nloc_real == ret);
    run_model<float>(dener, dforce, dvirial, datom_energy, datom_virial,
                     session, input_tensors, atommap, nframes, nghost_real);
  }

  // bkw map
  dforce_.resize(nframes * fwd_map.size() * 3);
  datom_energy_.resize(nframes * fwd_map.size());
  datom_virial_.resize(nframes * fwd_map.size() * 9);
  select_map<VALUETYPE>(dforce_, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(datom_energy_, datom_energy, bkw_map, 1, nframes,
                        fwd_map.size(), nall_real);
  select_map<VALUETYPE>(datom_virial_, datom_virial, bkw_map, 9, nframes,
                        fwd_map.size(), nall_real);
}

template void DeepPot::compute<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam_);

template void DeepPot::compute<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam_);

template void DeepPot::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam_);

template void DeepPot::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam_);

// mixed type
template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute_mixed_type(ENERGYVTYPE& dener,
                                 std::vector<VALUETYPE>& dforce_,
                                 std::vector<VALUETYPE>& dvirial,
                                 const int& nframes,
                                 const std::vector<VALUETYPE>& dcoord_,
                                 const std::vector<int>& datype_,
                                 const std::vector<VALUETYPE>& dbox,
                                 const std::vector<VALUETYPE>& fparam_,
                                 const std::vector<VALUETYPE>& aparam_) {
  int nloc = datype_.size() / nframes;
  // here atommap only used to get nloc
  atommap = deepmd::AtomMap(datype_.begin(), datype_.begin() + nloc);
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors_mixed_type<double>(
        input_tensors, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap);
    assert(ret == nloc);
    run_model<double>(dener, dforce_, dvirial, session, input_tensors, atommap,
                      nframes);
  } else {
    int ret = session_input_tensors_mixed_type<float>(
        input_tensors, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap);
    assert(ret == nloc);
    run_model<float>(dener, dforce_, dvirial, session, input_tensors, atommap,
                     nframes);
  }
}

template void DeepPot::compute_mixed_type<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute_mixed_type<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template void DeepPot::compute_mixed_type<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute_mixed_type<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPot::compute_mixed_type(ENERGYVTYPE& dener,
                                 std::vector<VALUETYPE>& dforce_,
                                 std::vector<VALUETYPE>& dvirial,
                                 std::vector<VALUETYPE>& datom_energy_,
                                 std::vector<VALUETYPE>& datom_virial_,
                                 const int& nframes,
                                 const std::vector<VALUETYPE>& dcoord_,
                                 const std::vector<int>& datype_,
                                 const std::vector<VALUETYPE>& dbox,
                                 const std::vector<VALUETYPE>& fparam_,
                                 const std::vector<VALUETYPE>& aparam_) {
  int nloc = datype_.size() / nframes;
  // here atommap only used to get nloc
  atommap = deepmd::AtomMap(datype_.begin(), datype_.begin() + nloc);
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int nloc = session_input_tensors_mixed_type<double>(
        input_tensors, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap);
    run_model<double>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                      session, input_tensors, atommap, nframes);
  } else {
    int nloc = session_input_tensors_mixed_type<float>(
        input_tensors, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap);
    run_model<float>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                     session, input_tensors, atommap, nframes);
  }
}

template void DeepPot::compute_mixed_type<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute_mixed_type<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template void DeepPot::compute_mixed_type<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPot::compute_mixed_type<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

void DeepPot::get_type_map(std::string& type_map) {
  type_map = get_scalar<STRINGTYPE>("model_attr/tmap");
}

DeepPotModelDevi::DeepPotModelDevi()
    : inited(false), init_nbor(false), numb_models(0) {}

DeepPotModelDevi::DeepPotModelDevi(
    const std::vector<std::string>& models,
    const int& gpu_rank,
    const std::vector<std::string>& file_contents)
    : inited(false), init_nbor(false), numb_models(0) {
  init(models, gpu_rank, file_contents);
}

DeepPotModelDevi::~DeepPotModelDevi() {
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    delete graph_defs[ii];
  }
}

void DeepPotModelDevi::init(const std::vector<std::string>& models,
                            const int& gpu_rank,
                            const std::vector<std::string>& file_contents) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  numb_models = models.size();
  sessions.resize(numb_models);
  graph_defs.resize(numb_models);

  int gpu_num = -1;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  DPGetDeviceCount(gpu_num);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  SessionOptions options;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    graph_defs[ii] = new GraphDef();
    if (file_contents.size() == 0)
      check_status(ReadBinaryProto(Env::Default(), models[ii], graph_defs[ii]));
    else
      (*graph_defs[ii]).ParseFromString(file_contents[ii]);
  }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (gpu_num > 0) {
    options.config.set_allow_soft_placement(true);
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
        0.9);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    DPErrcheck(DPSetDevice(gpu_rank % gpu_num));
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  for (unsigned ii = 0; ii < numb_models; ++ii) {
    if (gpu_num > 0) {
      std::string str = "/gpu:";
      str += std::to_string(gpu_rank % gpu_num);
      graph::SetDefaultDevice(str, &(*graph_defs[ii]));
    }
    check_status(NewSession(options, &(sessions[ii])));
    check_status(sessions[ii]->Create(*graph_defs[ii]));
  }
  try {
    model_version = get_scalar<STRINGTYPE>("model_attr/model_version");
  } catch (deepmd::tf_exception& e) {
    // no model version defined in old models
    model_version = "0.0";
  }
  if (!model_compatable(model_version)) {
    throw deepmd::deepmd_exception(
        "incompatable model: version " + model_version +
        " in graph, but version " + global_model_version +
        " supported. "
        "See https://deepmd.rtfd.io/compatability/ for details.");
  }
  dtype = session_get_dtype(sessions[0], "descrpt_attr/rcut");
  if (dtype == tensorflow::DT_DOUBLE) {
    rcut = get_scalar<double>("descrpt_attr/rcut");
  } else {
    rcut = get_scalar<float>("descrpt_attr/rcut");
  }
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  try {
    ntypes_spin = get_scalar<int>("spin_attr/ntypes_spin");
  } catch (deepmd::deepmd_exception) {
    ntypes_spin = 0;
  }
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  daparam = get_scalar<int>("fitting_attr/daparam");
  if (dfparam < 0) dfparam = 0;
  if (daparam < 0) daparam = 0;
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  // rcut = get_rcut();
  // cell_size = rcut;
  // ntypes = get_ntypes();
  inited = true;

  init_nbor = false;
}

template <class VT>
VT DeepPotModelDevi::get_scalar(const std::string name) const {
  VT myrcut;
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    VT ret = session_get_scalar<VT>(sessions[ii], name);
    if (ii == 0) {
      myrcut = ret;
    } else {
      assert(myrcut == ret);
    }
  }
  return myrcut;
}

template <typename VALUETYPE>
void DeepPotModelDevi::validate_fparam_aparam(
    const int& nloc,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam) const {
  if (fparam.size() != dfparam) {
    throw deepmd::deepmd_exception(
        "the dim of frame parameter provided is not consistent with what the "
        "model uses");
  }
  if (aparam.size() != daparam * nloc) {
    throw deepmd::deepmd_exception(
        "the dim of atom parameter provided is not consistent with what the "
        "model uses");
  }
}

template void DeepPotModelDevi::validate_fparam_aparam<double>(
    const int& nloc,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam) const;

template void DeepPotModelDevi::validate_fparam_aparam<float>(
    const int& nloc,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam) const;

// void
// DeepPotModelDevi::
// compute (ENERGYTYPE &			dener,
// 	 std::vector<VALUETYPE> &	dforce_,
// 	 std::vector<VALUETYPE> &	dvirial,
// 	 std::vector<VALUETYPE> &	model_devi,
// 	 const std::vector<VALUETYPE> &	dcoord_,
// 	 const std::vector<int> &	datype_,
// 	 const std::vector<VALUETYPE> &	dbox,
// 	 const std::vector<VALUETYPE> &	fparam,
// 	 const std::vector<VALUETYPE> &	aparam)
// {
//   if (numb_models == 0) return;

//   atommap = AtomMap<VALUETYPE> (datype_.begin(), datype_.end());
//   validate_fparam_aparam(atommap.get_type().size(), fparam, aparam);

//   std::vector<std::pair<std::string, Tensor>> input_tensors;
//   int nloc = session_input_tensors (input_tensors, dcoord_, ntypes, datype_,
//   dbox, cell_size, fparam, aparam, atommap);

//   std::vector<ENERGYTYPE > all_energy (numb_models);
//   std::vector<std::vector<VALUETYPE > > all_force (numb_models);
//   std::vector<std::vector<VALUETYPE > > all_virial (numb_models);

//   for (unsigned ii = 0; ii < numb_models; ++ii){
//     run_model (all_energy[ii], all_force[ii], all_virial[ii], sessions[ii],
//     input_tensors, atommap);
//   }

//   dener = 0;
//   for (unsigned ii = 0; ii < numb_models; ++ii){
//     dener += all_energy[ii];
//   }
//   dener /= VALUETYPE(numb_models);
//   compute_avg (dvirial, all_virial);
//   compute_avg (dforce_, all_force);

//   compute_std_f (model_devi, dforce_, all_force);

//   // for (unsigned ii = 0; ii < numb_models; ++ii){
//   //   cout << all_force[ii][573] << " " << all_force[ii][574] << " " <<
//   all_force[ii][575] << endl;
//   // }
//   // cout << dforce_[573] << " "
//   //      << dforce_[574] << " "
//   //      << dforce_[575] << " "
//   //      << model_devi[191] << endl;
// }

template <typename VALUETYPE>
void DeepPotModelDevi::compute(std::vector<ENERGYTYPE>& all_energy,
                               std::vector<std::vector<VALUETYPE>>& all_force,
                               std::vector<std::vector<VALUETYPE>>& all_virial,
                               const std::vector<VALUETYPE>& dcoord_,
                               const std::vector<int>& datype_,
                               const std::vector<VALUETYPE>& dbox,
                               const int nghost,
                               const InputNlist& lmp_list,
                               const int& ago,
                               const std::vector<VALUETYPE>& fparam,
                               const std::vector<VALUETYPE>& aparam_) {
  if (numb_models == 0) return;
  int nall = dcoord_.size() / 3;
  int nframes = 1;
  int nloc = nall - nghost;
  validate_fparam_aparam(nloc, fparam, aparam_);
  std::vector<std::pair<std::string, Tensor>> input_tensors;

  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_,
                          nghost, ntypes, nframes, daparam, nall);

  // agp == 0 means that the LAMMPS nbor list has been updated
  if (ago == 0) {
    atommap = AtomMap(datype.begin(), datype.begin() + nloc_real);
    assert(nloc == atommap.get_type().size());

    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }
  int ret;
  if (dtype == tensorflow::DT_DOUBLE) {
    ret = session_input_tensors<double>(input_tensors, dcoord, ntypes, datype,
                                        dbox, nlist, fparam, aparam, atommap,
                                        nghost_real, ago);
  } else {
    ret = session_input_tensors<float>(input_tensors, dcoord, ntypes, datype,
                                       dbox, nlist, fparam, aparam, atommap,
                                       nghost_real, ago);
  }
  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_virial.resize(numb_models);
  assert(nloc == ret);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    std::vector<VALUETYPE> dforce;
    if (dtype == tensorflow::DT_DOUBLE) {
      run_model<double>(all_energy[ii], dforce, all_virial[ii], sessions[ii],
                        input_tensors, atommap, 1, nghost_real);
    } else {
      run_model<float>(all_energy[ii], dforce, all_virial[ii], sessions[ii],
                       input_tensors, atommap, 1, nghost_real);
    }
    // bkw map
    all_force[ii].resize(nframes * fwd_map.size() * 3);
    select_map<VALUETYPE>(all_force[ii], dforce, bkw_map, 3, nframes,
                          fwd_map.size(), nall_real);
  }
}

template void DeepPotModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_virial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPotModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_virial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template <typename VALUETYPE>
void DeepPotModelDevi::compute(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<VALUETYPE>>& all_force,
    std::vector<std::vector<VALUETYPE>>& all_virial,
    std::vector<std::vector<VALUETYPE>>& all_atom_energy,
    std::vector<std::vector<VALUETYPE>>& all_atom_virial,
    const std::vector<VALUETYPE>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<VALUETYPE>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam_) {
  if (numb_models == 0) return;
  int nframes = 1;
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  validate_fparam_aparam(nloc, fparam, aparam_);
  std::vector<std::pair<std::string, Tensor>> input_tensors;

  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord_, datype_, aparam_,
                          nghost, ntypes, nframes, daparam, nall);
  // agp == 0 means that the LAMMPS nbor list has been updated

  if (ago == 0) {
    atommap = AtomMap(datype.begin(), datype.begin() + nloc_real);
    assert(nloc == atommap.get_type().size());

    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }
  int ret;
  if (dtype == tensorflow::DT_DOUBLE) {
    ret = session_input_tensors<double>(input_tensors, dcoord, ntypes, datype,
                                        dbox, nlist, fparam, aparam, atommap,
                                        nghost_real, ago);
  } else {
    ret = session_input_tensors<float>(input_tensors, dcoord, ntypes, datype,
                                       dbox, nlist, fparam, aparam, atommap,
                                       nghost_real, ago);
  }

  all_energy.resize(numb_models);
  all_force.resize(numb_models);
  all_virial.resize(numb_models);
  all_atom_energy.resize(numb_models);
  all_atom_virial.resize(numb_models);
  assert(nloc == ret);
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    std::vector<VALUETYPE> dforce, datom_energy, datom_virial;
    if (dtype == tensorflow::DT_DOUBLE) {
      run_model<double>(all_energy[ii], dforce, all_virial[ii], datom_energy,
                        datom_virial, sessions[ii], input_tensors, atommap, 1,
                        nghost_real);
    } else {
      run_model<float>(all_energy[ii], dforce, all_virial[ii], datom_energy,
                       datom_virial, sessions[ii], input_tensors, atommap, 1,
                       nghost_real);
    }
    // bkw map
    all_force[ii].resize(nframes * fwd_map.size() * 3);
    all_atom_energy[ii].resize(nframes * fwd_map.size());
    all_atom_virial[ii].resize(nframes * fwd_map.size() * 9);
    select_map<VALUETYPE>(all_force[ii], dforce, bkw_map, 3, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(all_atom_energy[ii], datom_energy, bkw_map, 1,
                          nframes, fwd_map.size(), nall_real);
    select_map<VALUETYPE>(all_atom_virial[ii], datom_virial, bkw_map, 9,
                          nframes, fwd_map.size(), nall_real);
  }
}

template void DeepPotModelDevi::compute<double>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<double>>& all_force,
    std::vector<std::vector<double>>& all_virial,
    std::vector<std::vector<double>>& all_atom_energy,
    std::vector<std::vector<double>>& all_atom_virial,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPotModelDevi::compute<float>(
    std::vector<ENERGYTYPE>& all_energy,
    std::vector<std::vector<float>>& all_force,
    std::vector<std::vector<float>>& all_virial,
    std::vector<std::vector<float>>& all_atom_energy,
    std::vector<std::vector<float>>& all_atom_virial,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_avg(VALUETYPE& dener,
                                   const std::vector<VALUETYPE>& all_energy) {
  assert(all_energy.size() == numb_models);
  if (numb_models == 0) return;

  dener = 0;
  for (unsigned ii = 0; ii < numb_models; ++ii) {
    dener += all_energy[ii];
  }
  dener /= (VALUETYPE)(numb_models);
}

template void DeepPotModelDevi::compute_avg<double>(
    double& dener, const std::vector<double>& all_energy);

template void DeepPotModelDevi::compute_avg<float>(
    float& dener, const std::vector<float>& all_energy);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_avg(
    std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  assert(xx.size() == numb_models);
  if (numb_models == 0) return;

  avg.resize(xx[0].size());
  fill(avg.begin(), avg.end(), VALUETYPE(0.));

  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0; jj < avg.size(); ++jj) {
      avg[jj] += xx[ii][jj];
    }
  }

  for (unsigned jj = 0; jj < avg.size(); ++jj) {
    avg[jj] /= VALUETYPE(numb_models);
  }
}

template void DeepPotModelDevi::compute_avg<double>(
    std::vector<double>& avg, const std::vector<std::vector<double>>& xx);

template void DeepPotModelDevi::compute_avg<float>(
    std::vector<float>& avg, const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_std(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx,
    const int& stride) {
  assert(xx.size() == numb_models);
  if (numb_models == 0) return;

  unsigned ndof = avg.size();
  unsigned nloc = ndof / stride;
  assert(nloc * stride == ndof);

  std.resize(nloc);
  fill(std.begin(), std.end(), VALUETYPE(0.));

  for (unsigned ii = 0; ii < numb_models; ++ii) {
    for (unsigned jj = 0; jj < nloc; ++jj) {
      const VALUETYPE* tmp_f = &(xx[ii][jj * stride]);
      const VALUETYPE* tmp_avg = &(avg[jj * stride]);
      for (unsigned dd = 0; dd < stride; ++dd) {
        VALUETYPE vdiff = tmp_f[dd] - tmp_avg[dd];
        std[jj] += vdiff * vdiff;
      }
    }
  }

  for (unsigned jj = 0; jj < nloc; ++jj) {
    std[jj] = sqrt(std[jj] / VALUETYPE(numb_models));
  }
}

template void DeepPotModelDevi::compute_std<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx,
    const int& stride);

template void DeepPotModelDevi::compute_std<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx,
    const int& stride);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_std_e(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  compute_std(std, avg, xx, 1);
}

template void DeepPotModelDevi::compute_std_e<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx);

template void DeepPotModelDevi::compute_std_e<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_std_f(
    std::vector<VALUETYPE>& std,
    const std::vector<VALUETYPE>& avg,
    const std::vector<std::vector<VALUETYPE>>& xx) {
  compute_std(std, avg, xx, 3);
}

template void DeepPotModelDevi::compute_std_f<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const std::vector<std::vector<double>>& xx);

template void DeepPotModelDevi::compute_std_f<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const std::vector<std::vector<float>>& xx);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_relative_std(std::vector<VALUETYPE>& std,
                                            const std::vector<VALUETYPE>& avg,
                                            const VALUETYPE eps,
                                            const int& stride) {
  unsigned ndof = avg.size();
  unsigned nloc = std.size();
  assert(nloc * stride == ndof);

  for (unsigned ii = 0; ii < nloc; ++ii) {
    const VALUETYPE* tmp_avg = &(avg[ii * stride]);
    VALUETYPE f_norm = 0.0;
    for (unsigned dd = 0; dd < stride; ++dd) {
      f_norm += tmp_avg[dd] * tmp_avg[dd];
    }
    f_norm = sqrt(f_norm);
    std[ii] /= f_norm + eps;
  }
}

template void DeepPotModelDevi::compute_relative_std<double>(
    std::vector<double>& std,
    const std::vector<double>& avg,
    const double eps,
    const int& stride);

template void DeepPotModelDevi::compute_relative_std<float>(
    std::vector<float>& std,
    const std::vector<float>& avg,
    const float eps,
    const int& stride);

template <typename VALUETYPE>
void DeepPotModelDevi::compute_relative_std_f(std::vector<VALUETYPE>& std,
                                              const std::vector<VALUETYPE>& avg,
                                              const VALUETYPE eps) {
  compute_relative_std(std, avg, eps, 3);
}

template void DeepPotModelDevi::compute_relative_std_f<double>(
    std::vector<double>& std, const std::vector<double>& avg, const double eps);

template void DeepPotModelDevi::compute_relative_std_f<float>(
    std::vector<float>& std, const std::vector<float>& avg, const float eps);
