// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_TENSORFLOW
#include "DeepSpinTF.h"

#include <cstdint>
#include <stdexcept>

#include "AtomMap.h"
#include "common.h"
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
    dforce_.resize(static_cast<size_t>(nframes) * nall * 3);
    fill(dforce_.begin(), dforce_.end(), (VALUETYPE)0.0);
    // dvirial of size 9
    dvirial.resize(static_cast<size_t>(nframes) * 9);
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

  std::vector<VALUETYPE> dforce(static_cast<size_t>(nframes) * 3 * nall);
  dvirial.resize(static_cast<size_t>(nframes) * 9);
  for (int ii = 0; ii < nframes; ++ii) {
    dener[ii] = oe(ii);
  }
  for (size_t ii = 0; ii < static_cast<size_t>(nframes) * nall * 3; ++ii) {
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
    dforce_.resize(static_cast<size_t>(nframes) * nall * 3);
    fill(dforce_.begin(), dforce_.end(), (VALUETYPE)0.0);
    // dvirial of size 9
    dvirial.resize(static_cast<size_t>(nframes) * 9);
    fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.0);
    // datom_energy_ of size nall
    datom_energy_.resize(static_cast<size_t>(nframes) * nall);
    fill(datom_energy_.begin(), datom_energy_.end(), (VALUETYPE)0.0);
    // datom_virial_ of size nall * 9
    datom_virial_.resize(static_cast<size_t>(nframes) * nall * 9);
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

  std::vector<VALUETYPE> dforce(static_cast<size_t>(nframes) * 3 * nall);
  std::vector<VALUETYPE> datom_energy(static_cast<size_t>(nframes) * nall, 0);
  std::vector<VALUETYPE> datom_virial(static_cast<size_t>(nframes) * 9 * nall);
  dvirial.resize(static_cast<size_t>(nframes) * 9);
  for (int ii = 0; ii < nframes; ++ii) {
    dener[ii] = oe(ii);
  }
  for (size_t ii = 0; ii < static_cast<size_t>(nframes) * nall * 3; ++ii) {
    dforce[ii] = of(ii);
  }
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nloc; ++jj) {
      datom_energy[ii * nall + jj] = oae(ii * nloc + jj);
    }
  }
  for (size_t ii = 0; ii < static_cast<size_t>(nframes) * nall * 9; ++ii) {
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

DeepSpinTF::DeepSpinTF()
    : inited(false), init_nbor(false), graph_def(new GraphDef()) {}

DeepSpinTF::DeepSpinTF(const std::string& model,
                       const int& gpu_rank,
                       const std::string& file_content)
    : inited(false), init_nbor(false), graph_def(new GraphDef()) {
  try {
    init(model, gpu_rank, file_content);
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    delete graph_def;
    throw;
  }
}

DeepSpinTF::~DeepSpinTF() { delete graph_def; }

void DeepSpinTF::init(const std::string& model,
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

  if (file_content.size() == 0) {
    check_status(ReadBinaryProto(Env::Default(), model, graph_def));
  } else {
    (*graph_def).ParseFromString(file_content);
  }
  int gpu_num = -1;
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  DPGetDeviceCount(gpu_num);  // check current device environment
  if (gpu_num > 0) {
    options.config.set_allow_soft_placement(true);
    options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(
        0.9);
    options.config.mutable_gpu_options()->set_allow_growth(true);
    DPErrcheck(DPSetDevice(gpu_rank % gpu_num));
    std::string str = "/gpu:0";
    // See
    // https://github.com/tensorflow/tensorflow/blame/8fac27b486939f40bc8e362b94a16a4a8bb51869/tensorflow/core/protobuf/config.proto#L80
    options.config.mutable_gpu_options()->set_visible_device_list(
        std::to_string(gpu_rank % gpu_num));
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
        "incompatible model: version " + model_version +
        " in graph, but version " + global_model_version +
        " supported "
        "See https://deepmd.rtfd.io/compatibility/ for details.");
  }
  dtype = session_get_dtype(session, "descrpt_attr/rcut");
  if (dtype == tensorflow::DT_DOUBLE) {
    rcut = get_scalar<double>("descrpt_attr/rcut");
  } else {
    rcut = get_scalar<float>("descrpt_attr/rcut");
  }
  cell_size = rcut;
  ntypes = get_scalar<int>("descrpt_attr/ntypes");
  ntypes_spin = get_scalar<int>("spin_attr/ntypes_spin");
  dfparam = get_scalar<int>("fitting_attr/dfparam");
  daparam = get_scalar<int>("fitting_attr/daparam");
  if (dfparam < 0) {
    dfparam = 0;
  }
  if (daparam < 0) {
    daparam = 0;
  }
  if (daparam > 0) {
    try {
      aparam_nall = get_scalar<bool>("fitting_attr/aparam_nall");
    } catch (const deepmd::deepmd_exception&) {
      aparam_nall = false;
    }
  } else {
    aparam_nall = false;
  }
  model_type = get_scalar<STRINGTYPE>("model_attr/model_type");
  inited = true;

  init_nbor = false;
}

template <class VT>
VT DeepSpinTF::get_scalar(const std::string& name) const {
  return session_get_scalar<VT>(session, name);
}

template <class VT>
void DeepSpinTF::get_vector(std::vector<VT>& vec,
                            const std::string& name) const {
  session_get_vector<VT>(vec, session, name);
}

template <typename VALUETYPE>
void DeepSpinTF::validate_fparam_aparam(
    const int& nframes,
    const int& nloc,
    const std::vector<VALUETYPE>& fparam,
    const std::vector<VALUETYPE>& aparam) const {
  if (fparam.size() != dfparam &&
      fparam.size() != static_cast<size_t>(nframes) * dfparam) {
    throw deepmd::deepmd_exception(
        "the dim of frame parameter provided is not consistent with what the "
        "model uses");
  }

  if (aparam.size() != static_cast<size_t>(daparam) * nloc &&
      aparam.size() != static_cast<size_t>(nframes) * daparam * nloc) {
    throw deepmd::deepmd_exception(
        "the dim of atom parameter provided is not consistent with what the "
        "model uses");
  }
}

template void DeepSpinTF::validate_fparam_aparam<double>(
    const int& nframes,
    const int& nloc,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam) const;

template void DeepSpinTF::validate_fparam_aparam<float>(
    const int& nframes,
    const int& nloc,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam) const;

template <typename VALUETYPE>
void DeepSpinTF::tile_fparam_aparam(std::vector<VALUETYPE>& out_param,
                                    const int& nframes,
                                    const int& dparam,
                                    const std::vector<VALUETYPE>& param) const {
  if (param.size() == dparam) {
    out_param.resize(static_cast<size_t>(nframes) * dparam);
    for (int ii = 0; ii < nframes; ++ii) {
      std::copy(param.begin(), param.end(),
                out_param.begin() + static_cast<std::uint64_t>(ii) * dparam);
    }
  } else if (param.size() == static_cast<size_t>(nframes) * dparam) {
    out_param = param;
  }
}

template void DeepSpinTF::tile_fparam_aparam<double>(
    std::vector<double>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<double>& param) const;

template void DeepSpinTF::tile_fparam_aparam<float>(
    std::vector<float>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<float>& param) const;

// ENERGYVTYPE: std::vector<ENERGYTYPE> or ENERGYTYPE

// support spin
template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepSpinTF::compute(ENERGYVTYPE& dener,
                         std::vector<VALUETYPE>& dforce_,
                         std::vector<VALUETYPE>& dforce_mag_,
                         std::vector<VALUETYPE>& dvirial,
                         std::vector<VALUETYPE>& datom_energy_,
                         std::vector<VALUETYPE>& datom_virial_,
                         const std::vector<VALUETYPE>& dcoord_,
                         const std::vector<VALUETYPE>& dspin_,
                         const std::vector<int>& datype_,
                         const std::vector<VALUETYPE>& dbox,
                         const std::vector<VALUETYPE>& fparam_,
                         const std::vector<VALUETYPE>& aparam_,
                         const bool atomic) {
  // if datype.size is 0, not clear nframes; but 1 is just ok
  int nframes = datype_.size() > 0 ? (dcoord_.size() / 3 / datype_.size()) : 1;
  int nloc = datype_.size();
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  std::vector<VALUETYPE> extend_dcoord;
  std::vector<int> extend_atype;
  extend_nlist(extend_dcoord, extend_atype, dcoord_, dspin_, datype_);

  atommap = deepmd::AtomMap(extend_atype.begin(), extend_atype.end());

  std::vector<std::pair<std::string, Tensor>> input_tensors;
  std::vector<VALUETYPE> dforce_tmp;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(
        input_tensors, extend_dcoord, ntypes, extend_atype, dbox, cell_size,
        fparam, aparam, atommap, "", aparam_nall);
    if (atomic) {
      run_model<double>(dener, dforce_tmp, dvirial, datom_energy_,
                        datom_virial_, session, input_tensors, atommap,
                        nframes);
    } else {
      run_model<double>(dener, dforce_tmp, dvirial, session, input_tensors,
                        atommap, nframes);
    }
  } else {
    int ret = session_input_tensors<float>(
        input_tensors, extend_dcoord, ntypes, extend_atype, dbox, cell_size,
        fparam, aparam, atommap, "", aparam_nall);
    if (atomic) {
      run_model<float>(dener, dforce_tmp, dvirial, datom_energy_, datom_virial_,
                       session, input_tensors, atommap, nframes);
    } else {
      run_model<float>(dener, dforce_tmp, dvirial, session, input_tensors,
                       atommap, nframes);
    }
  }
  // backward force and mag.
  dforce_.resize(static_cast<size_t>(nframes) * nloc * 3);
  dforce_mag_.resize(static_cast<size_t>(nframes) * nloc * 3);
  for (int ii = 0; ii < nloc; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      dforce_[3 * ii + dd] = dforce_tmp[3 * ii + dd];
      if (datype_[ii] < ntypes_spin) {
        dforce_mag_[3 * ii + dd] = dforce_tmp[3 * (ii + nloc) + dd];
      } else {
        dforce_mag_[3 * ii + dd] = 0.0;
      }
    }
  }
}

template void DeepSpinTF::compute<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dforce_mag_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepSpinTF::compute<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dforce_mag_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

template void DeepSpinTF::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dforce_mag_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepSpinTF::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dforce_mag_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

// support spin
template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepSpinTF::compute(ENERGYVTYPE& dener,
                         std::vector<VALUETYPE>& dforce_,
                         std::vector<VALUETYPE>& dforce_mag_,
                         std::vector<VALUETYPE>& dvirial,
                         std::vector<VALUETYPE>& datom_energy_,
                         std::vector<VALUETYPE>& datom_virial_,
                         const std::vector<VALUETYPE>& dcoord_,
                         const std::vector<VALUETYPE>& dspin_,
                         const std::vector<int>& datype_,
                         const std::vector<VALUETYPE>& dbox,
                         const int nghost,
                         const InputNlist& lmp_list,
                         const int& ago,
                         const std::vector<VALUETYPE>& fparam_,
                         const std::vector<VALUETYPE>& aparam__,
                         const bool atomic) {
  int nall = datype_.size();
  // if nall==0, unclear nframes, but 1 is ok
  int nframes = nall > 0 ? (dcoord_.size() / nall / 3) : 1;
  int nloc = nall - nghost;

  std::vector<VALUETYPE> extend_dcoord;
  extend(extend_inum, extend_ilist, extend_numneigh, extend_neigh,
         extend_firstneigh, extend_dcoord, extend_dtype, extend_nghost,
         new_idx_map, old_idx_map, lmp_list, dcoord_, datype_, nghost, dspin_,
         ntypes, ntypes_spin);
  InputNlist extend_lmp_list(extend_inum, &extend_ilist[0], &extend_numneigh[0],
                             &extend_firstneigh[0]);
  extend_lmp_list.set_mask(lmp_list.mask);
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam_;
  validate_fparam_aparam(nframes, (aparam_nall ? nall : nloc), fparam_,
                         aparam__);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam_, nframes, (aparam_nall ? nall : nloc) * daparam,
                     aparam__);
  std::vector<std::pair<std::string, Tensor>> input_tensors;
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  select_real_atoms_coord(dcoord, datype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, extend_dcoord, extend_dtype,
                          aparam_, extend_nghost, ntypes, nframes, daparam,
                          nall, aparam_nall);

  if (ago == 0) {
    atommap = deepmd::AtomMap(datype.begin(), datype.begin() + nloc_real);
    assert(nloc_real == atommap.get_type().size());

    nlist_data.copy_from_nlist(extend_lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(
        input_tensors, dcoord, ntypes, datype, dbox, nlist, fparam, aparam,
        atommap, nghost_real, ago, "", aparam_nall);
    assert(nloc_real == ret);
    if (atomic) {
      run_model<double>(dener, dforce, dvirial, datom_energy, datom_virial,
                        session, input_tensors, atommap, nframes, nghost_real);
    } else {
      run_model<double>(dener, dforce, dvirial, session, input_tensors, atommap,
                        nframes, nghost_real);
    }
  } else {
    int ret = session_input_tensors<float>(
        input_tensors, dcoord, ntypes, datype, dbox, nlist, fparam, aparam,
        atommap, nghost_real, ago, "", aparam_nall);
    assert(nloc_real == ret);
    if (atomic) {
      run_model<float>(dener, dforce, dvirial, datom_energy, datom_virial,
                       session, input_tensors, atommap, nframes, nghost_real);
    } else {
      run_model<float>(dener, dforce, dvirial, session, input_tensors, atommap,
                       nframes, nghost_real);
    }
  }

  // bkw map
  std::vector<VALUETYPE> dforce_tmp, datom_energy_tmp, datom_virial_tmp;
  dforce_tmp.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  datom_energy_tmp.resize(static_cast<size_t>(nframes) * fwd_map.size());
  datom_virial_tmp.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
  select_map<VALUETYPE>(dforce_tmp, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(datom_energy_tmp, datom_energy, bkw_map, 1, nframes,
                        fwd_map.size(), nall_real);
  select_map<VALUETYPE>(datom_virial_tmp, datom_virial, bkw_map, 9, nframes,
                        fwd_map.size(), nall_real);
  // backward force and mag.
  dforce_.resize(static_cast<size_t>(nframes) * nall * 3);
  dforce_mag_.resize(static_cast<size_t>(nframes) * nall * 3);
  datom_energy_.resize(static_cast<size_t>(nframes) * nall);
  datom_virial_.resize(static_cast<size_t>(nframes) * nall * 9);
  for (int ii = 0; ii < nall; ++ii) {
    for (int dd = 0; dd < 3; ++dd) {
      int new_idx = new_idx_map[ii];
      dforce_[3 * ii + dd] = dforce_tmp[3 * new_idx + dd];
      datom_energy_[ii] = datom_energy_tmp[new_idx];
      datom_virial_[ii] = datom_virial_tmp[new_idx];
      if (datype_[ii] < ntypes_spin && ii < nloc) {
        dforce_mag_[3 * ii + dd] = dforce_tmp[3 * (new_idx + nloc) + dd];
      } else if (datype_[ii] < ntypes_spin) {
        dforce_mag_[3 * ii + dd] = dforce_tmp[3 * (new_idx + nghost) + dd];
      } else {
        dforce_mag_[3 * ii + dd] = 0.0;
      }
    }
  }
}

template void DeepSpinTF::compute<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dforce_mag_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam_,
    const bool atomic);

template void DeepSpinTF::compute<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dforce_mag_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam_,
    const bool atomic);

template void DeepSpinTF::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dforce_mag_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam_,
    const bool atomic);

template void DeepSpinTF::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dforce_mag_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<float>& dspin_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam_,
    const bool atomic);

// end support spin

void DeepSpinTF::get_type_map(std::string& type_map) {
  type_map = get_scalar<STRINGTYPE>("model_attr/tmap");
}

// forward to template method
// support spin
void DeepSpinTF::computew(std::vector<double>& ener,
                          std::vector<double>& force,
                          std::vector<double>& force_mag,
                          std::vector<double>& virial,
                          std::vector<double>& atom_energy,
                          std::vector<double>& atom_virial,
                          const std::vector<double>& coord,
                          const std::vector<double>& spin,
                          const std::vector<int>& atype,
                          const std::vector<double>& box,
                          const std::vector<double>& fparam,
                          const std::vector<double>& aparam,
                          const bool atomic) {
  compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord, spin,
          atype, box, fparam, aparam, atomic);
}
void DeepSpinTF::computew(std::vector<double>& ener,
                          std::vector<float>& force,
                          std::vector<float>& force_mag,
                          std::vector<float>& virial,
                          std::vector<float>& atom_energy,
                          std::vector<float>& atom_virial,
                          const std::vector<float>& coord,
                          const std::vector<float>& spin,
                          const std::vector<int>& atype,
                          const std::vector<float>& box,
                          const std::vector<float>& fparam,
                          const std::vector<float>& aparam,
                          const bool atomic) {
  compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord, spin,
          atype, box, fparam, aparam, atomic);
}
// support spin
void DeepSpinTF::computew(std::vector<double>& ener,
                          std::vector<double>& force,
                          std::vector<double>& force_mag,
                          std::vector<double>& virial,
                          std::vector<double>& atom_energy,
                          std::vector<double>& atom_virial,
                          const std::vector<double>& coord,
                          const std::vector<double>& spin,
                          const std::vector<int>& atype,
                          const std::vector<double>& box,
                          const int nghost,
                          const InputNlist& inlist,
                          const int& ago,
                          const std::vector<double>& fparam,
                          const std::vector<double>& aparam,
                          const bool atomic) {
  compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord, spin,
          atype, box, nghost, inlist, ago, fparam, aparam, atomic);
}
void DeepSpinTF::computew(std::vector<double>& ener,
                          std::vector<float>& force,
                          std::vector<float>& force_mag,
                          std::vector<float>& virial,
                          std::vector<float>& atom_energy,
                          std::vector<float>& atom_virial,
                          const std::vector<float>& coord,
                          const std::vector<float>& spin,
                          const std::vector<int>& atype,
                          const std::vector<float>& box,
                          const int nghost,
                          const InputNlist& inlist,
                          const int& ago,
                          const std::vector<float>& fparam,
                          const std::vector<float>& aparam,
                          const bool atomic) {
  compute(ener, force, force_mag, virial, atom_energy, atom_virial, coord, spin,
          atype, box, nghost, inlist, ago, fparam, aparam, atomic);
}

void DeepSpinTF::cum_sum(std::map<int, int>& sum, std::map<int, int>& vec) {
  sum[0] = 0;
  for (int ii = 1; ii < vec.size(); ++ii) {
    sum[ii] = sum[ii - 1] + vec[ii - 1];
  }
}

template <typename VALUETYPE>
void DeepSpinTF::extend(int& extend_inum,
                        std::vector<int>& extend_ilist,
                        std::vector<int>& extend_numneigh,
                        std::vector<std::vector<int>>& extend_neigh,
                        std::vector<int*>& extend_firstneigh,
                        std::vector<VALUETYPE>& extend_dcoord,
                        std::vector<int>& extend_atype,
                        int& extend_nghost,
                        std::map<int, int>& new_idx_map,
                        std::map<int, int>& old_idx_map,
                        const InputNlist& lmp_list,
                        const std::vector<VALUETYPE>& dcoord,
                        const std::vector<int>& atype,
                        const int nghost,
                        const std::vector<VALUETYPE>& spin,
                        const int numb_types,
                        const int numb_types_spin) {
  extend_ilist.clear();
  extend_numneigh.clear();
  extend_neigh.clear();
  extend_firstneigh.clear();
  extend_dcoord.clear();
  extend_atype.clear();
  if (dtype == tensorflow::DT_DOUBLE) {
    get_vector<double>(virtual_len, "spin_attr/virtual_len");
    get_vector<double>(spin_norm, "spin_attr/spin_norm");
  } else {
    std::vector<float> virtual_len;
    std::vector<float> spin_norm;
    get_vector<float>(virtual_len, "spin_attr/virtual_len");
    get_vector<float>(spin_norm, "spin_attr/spin_norm");
  }

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
      loc_type_count.insert(std::pair<int, int>(atype[i], 1));
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
      ghost_type_count.insert(std::pair<int, int>(atype[i], 1));
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
    const int* jlist = lmp_list.firstneigh[old_idx_map[ii]];
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
        double temp_dcoord = dcoord[ii * 3 + jj] + spin[ii * 3 + jj] /
                                                       spin_norm[atype[ii]] *
                                                       virtual_len[atype[ii]];
        extend_dcoord[(new_idx_map[ii] + nloc) * 3 + jj] = temp_dcoord;
      }
    }
  }
  for (int ii = nloc; ii < nall; ii++) {
    for (int jj = 0; jj < 3; jj++) {
      extend_dcoord[new_idx_map[ii] * 3 + jj] = dcoord[ii * 3 + jj];
      if (atype[ii] < numb_types_spin) {
        double temp_dcoord = dcoord[ii * 3 + jj] + spin[ii * 3 + jj] /
                                                       spin_norm[atype[ii]] *
                                                       virtual_len[atype[ii]];
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

template void DeepSpinTF::extend<double>(
    int& extend_inum,
    std::vector<int>& extend_ilist,
    std::vector<int>& extend_numneigh,
    std::vector<std::vector<int>>& extend_neigh,
    std::vector<int*>& extend_firstneigh,
    std::vector<double>& extend_dcoord,
    std::vector<int>& extend_atype,
    int& extend_nghost,
    std::map<int, int>& new_idx_map,
    std::map<int, int>& old_idx_map,
    const InputNlist& lmp_list,
    const std::vector<double>& dcoord,
    const std::vector<int>& atype,
    const int nghost,
    const std::vector<double>& spin,
    const int numb_types,
    const int numb_types_spin);

template void DeepSpinTF::extend<float>(
    int& extend_inum,
    std::vector<int>& extend_ilist,
    std::vector<int>& extend_numneigh,
    std::vector<std::vector<int>>& extend_neigh,
    std::vector<int*>& extend_firstneigh,
    std::vector<float>& extend_dcoord,
    std::vector<int>& extend_atype,
    int& extend_nghost,
    std::map<int, int>& new_idx_map,
    std::map<int, int>& old_idx_map,
    const InputNlist& lmp_list,
    const std::vector<float>& dcoord,
    const std::vector<int>& atype,
    const int nghost,
    const std::vector<float>& spin,
    const int numb_types,
    const int numb_types_spin);

template <typename VALUETYPE>
void DeepSpinTF::extend_nlist(std::vector<VALUETYPE>& extend_dcoord,
                              std::vector<int>& extend_atype,
                              const std::vector<VALUETYPE>& dcoord_,
                              const std::vector<VALUETYPE>& dspin_,
                              const std::vector<int>& datype_) {
  if (dtype == tensorflow::DT_DOUBLE) {
    get_vector<double>(virtual_len, "spin_attr/virtual_len");
    get_vector<double>(spin_norm, "spin_attr/spin_norm");
  } else {
    std::vector<float> virtual_len;
    std::vector<float> spin_norm;
    get_vector<float>(virtual_len, "spin_attr/virtual_len");
    get_vector<float>(spin_norm, "spin_attr/spin_norm");
  }
  // extend coord and atype
  int nloc = datype_.size();
  int nloc_spin = 0;
  for (int ii = 0; ii < nloc; ii++) {
    if (datype_[ii] < ntypes_spin) {
      nloc_spin += 1;
    }
  }
  int extend_nall = nloc + nloc_spin;
  extend_dcoord.resize(static_cast<size_t>(extend_nall) * 3);
  extend_atype.resize(extend_nall);
  for (int ii = 0; ii < nloc; ii++) {
    extend_atype[ii] = datype_[ii];
    if (datype_[ii] < ntypes_spin) {
      extend_atype[ii + nloc] = datype_[ii] + ntypes - ntypes_spin;
    }
    for (int jj = 0; jj < 3; jj++) {
      extend_dcoord[ii * 3 + jj] = dcoord_[ii * 3 + jj];
      if (datype_[ii] < ntypes_spin) {
        extend_dcoord[(ii + nloc) * 3 + jj] =
            dcoord_[ii * 3 + jj] + dspin_[ii * 3 + jj] /
                                       spin_norm[datype_[ii]] *
                                       virtual_len[datype_[ii]];
      }
    }
  }
}

template void DeepSpinTF::extend_nlist<double>(
    std::vector<double>& extend_dcoord,
    std::vector<int>& extend_atype,
    const std::vector<double>& dcoord_,
    const std::vector<double>& dspin_,
    const std::vector<int>& datype_);

template void DeepSpinTF::extend_nlist<float>(std::vector<float>& extend_dcoord,
                                              std::vector<int>& extend_atype,
                                              const std::vector<float>& dcoord_,
                                              const std::vector<float>& dspin_,
                                              const std::vector<int>& datype_);
#endif