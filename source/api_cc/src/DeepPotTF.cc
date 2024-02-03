// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_TENSORFLOW
#include "DeepPotTF.h"

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

DeepPotTF::DeepPotTF()
    : inited(false), init_nbor(false), graph_def(new GraphDef()) {}

DeepPotTF::DeepPotTF(const std::string& model,
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

DeepPotTF::~DeepPotTF() { delete graph_def; }

void DeepPotTF::init(const std::string& model,
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
  } catch (const deepmd::deepmd_exception&) {
    ntypes_spin = 0;
  }
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
VT DeepPotTF::get_scalar(const std::string& name) const {
  return session_get_scalar<VT>(session, name);
}

template <typename VALUETYPE>
void DeepPotTF::validate_fparam_aparam(
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

template void DeepPotTF::validate_fparam_aparam<double>(
    const int& nframes,
    const int& nloc,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam) const;

template void DeepPotTF::validate_fparam_aparam<float>(
    const int& nframes,
    const int& nloc,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam) const;

template <typename VALUETYPE>
void DeepPotTF::tile_fparam_aparam(std::vector<VALUETYPE>& out_param,
                                   const int& nframes,
                                   const int& dparam,
                                   const std::vector<VALUETYPE>& param) const {
  if (param.size() == dparam) {
    out_param.resize(static_cast<size_t>(nframes) * dparam);
    for (int ii = 0; ii < nframes; ++ii) {
      std::copy(param.begin(), param.end(),
                out_param.begin() + static_cast<unsigned long>(ii) * dparam);
    }
  } else if (param.size() == static_cast<size_t>(nframes) * dparam) {
    out_param = param;
  }
}

template void DeepPotTF::tile_fparam_aparam<double>(
    std::vector<double>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<double>& param) const;

template void DeepPotTF::tile_fparam_aparam<float>(
    std::vector<float>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<float>& param) const;

// ENERGYVTYPE: std::vector<ENERGYTYPE> or ENERGYTYPE

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotTF::compute(ENERGYVTYPE& dener,
                        std::vector<VALUETYPE>& dforce_,
                        std::vector<VALUETYPE>& dvirial,
                        std::vector<VALUETYPE>& datom_energy_,
                        std::vector<VALUETYPE>& datom_virial_,
                        const std::vector<VALUETYPE>& dcoord_,
                        const std::vector<int>& datype_,
                        const std::vector<VALUETYPE>& dbox,
                        const std::vector<VALUETYPE>& fparam_,
                        const std::vector<VALUETYPE>& aparam_) {
  // if datype.size is 0, not clear nframes; but 1 is just ok
  int nframes = datype_.size() > 0 ? (dcoord_.size() / 3 / datype_.size()) : 1;
  atommap = deepmd::AtomMap(datype_.begin(), datype_.end());
  int nloc = datype_.size();
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  std::vector<std::pair<std::string, Tensor>> input_tensors;

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(input_tensors, dcoord_, ntypes,
                                            datype_, dbox, cell_size, fparam,
                                            aparam, atommap, "", aparam_nall);
    run_model<double>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                      session, input_tensors, atommap, nframes);
  } else {
    int ret = session_input_tensors<float>(input_tensors, dcoord_, ntypes,
                                           datype_, dbox, cell_size, fparam,
                                           aparam, atommap, "", aparam_nall);
    run_model<float>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                     session, input_tensors, atommap, nframes);
  }
}

template void DeepPotTF::compute<double, ENERGYTYPE>(
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

template void DeepPotTF::compute<float, ENERGYTYPE>(
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

template void DeepPotTF::compute<double, std::vector<ENERGYTYPE>>(
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

template void DeepPotTF::compute<float, std::vector<ENERGYTYPE>>(
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
void DeepPotTF::compute(ENERGYVTYPE& dener,
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
  // if nall==0, unclear nframes, but 1 is ok
  int nframes = nall > 0 ? (dcoord_.size() / nall / 3) : 1;
  int nloc = nall - nghost;
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
                          nall_real, nloc_real, dcoord_, datype_, aparam_,
                          nghost, ntypes, nframes, daparam, nall, aparam_nall);

  if (ago == 0) {
    atommap = deepmd::AtomMap(datype.begin(), datype.begin() + nloc_real);
    assert(nloc_real == atommap.get_type().size());

    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.shuffle(atommap);
    nlist_data.make_inlist(nlist);
  }

  if (dtype == tensorflow::DT_DOUBLE) {
    int ret = session_input_tensors<double>(
        input_tensors, dcoord, ntypes, datype, dbox, nlist, fparam, aparam,
        atommap, nghost_real, ago, "", aparam_nall);
    assert(nloc_real == ret);
    run_model<double>(dener, dforce, dvirial, datom_energy, datom_virial,
                      session, input_tensors, atommap, nframes, nghost_real);
  } else {
    int ret = session_input_tensors<float>(
        input_tensors, dcoord, ntypes, datype, dbox, nlist, fparam, aparam,
        atommap, nghost_real, ago, "", aparam_nall);
    assert(nloc_real == ret);
    run_model<float>(dener, dforce, dvirial, datom_energy, datom_virial,
                     session, input_tensors, atommap, nframes, nghost_real);
  }

  // bkw map
  dforce_.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  datom_energy_.resize(static_cast<size_t>(nframes) * fwd_map.size());
  datom_virial_.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
  select_map<VALUETYPE>(dforce_, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(datom_energy_, datom_energy, bkw_map, 1, nframes,
                        fwd_map.size(), nall_real);
  select_map<VALUETYPE>(datom_virial_, datom_virial, bkw_map, 9, nframes,
                        fwd_map.size(), nall_real);
}

template void DeepPotTF::compute<double, ENERGYTYPE>(
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

template void DeepPotTF::compute<float, ENERGYTYPE>(
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

template void DeepPotTF::compute<double, std::vector<ENERGYTYPE>>(
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

template void DeepPotTF::compute<float, std::vector<ENERGYTYPE>>(
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
void DeepPotTF::compute_mixed_type(ENERGYVTYPE& dener,
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
        fparam, aparam, atommap, "", aparam_nall);
    assert(ret == nloc);
    run_model<double>(dener, dforce_, dvirial, session, input_tensors, atommap,
                      nframes);
  } else {
    int ret = session_input_tensors_mixed_type<float>(
        input_tensors, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap, "", aparam_nall);
    assert(ret == nloc);
    run_model<float>(dener, dforce_, dvirial, session, input_tensors, atommap,
                     nframes);
  }
}

template void DeepPotTF::compute_mixed_type<double, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPotTF::compute_mixed_type<float, ENERGYTYPE>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const int& nframes,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam);

template void DeepPotTF::compute_mixed_type<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const int& nframes,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam);

template void DeepPotTF::compute_mixed_type<float, std::vector<ENERGYTYPE>>(
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
void DeepPotTF::compute_mixed_type(ENERGYVTYPE& dener,
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
        fparam, aparam, atommap, "", aparam_nall);
    run_model<double>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                      session, input_tensors, atommap, nframes);
  } else {
    int nloc = session_input_tensors_mixed_type<float>(
        input_tensors, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap, "", aparam_nall);
    run_model<float>(dener, dforce_, dvirial, datom_energy_, datom_virial_,
                     session, input_tensors, atommap, nframes);
  }
}

template void DeepPotTF::compute_mixed_type<double, ENERGYTYPE>(
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

template void DeepPotTF::compute_mixed_type<float, ENERGYTYPE>(
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

template void DeepPotTF::compute_mixed_type<double, std::vector<ENERGYTYPE>>(
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

template void DeepPotTF::compute_mixed_type<float, std::vector<ENERGYTYPE>>(
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

void DeepPotTF::get_type_map(std::string& type_map) {
  type_map = get_scalar<STRINGTYPE>("model_attr/tmap");
}

// forward to template method
void DeepPotTF::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam);
}
void DeepPotTF::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam);
}
void DeepPotTF::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const int nghost,
                         const InputNlist& inlist,
                         const int& ago,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam);
}
void DeepPotTF::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const int nghost,
                         const InputNlist& inlist,
                         const int& ago,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam);
}
void DeepPotTF::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<double>& force,
                                    std::vector<double>& virial,
                                    std::vector<double>& atom_energy,
                                    std::vector<double>& atom_virial,
                                    const int& nframes,
                                    const std::vector<double>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<double>& box,
                                    const std::vector<double>& fparam,
                                    const std::vector<double>& aparam) {
  compute_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                     coord, atype, box, fparam, aparam);
}
void DeepPotTF::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<float>& force,
                                    std::vector<float>& virial,
                                    std::vector<float>& atom_energy,
                                    std::vector<float>& atom_virial,
                                    const int& nframes,
                                    const std::vector<float>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<float>& box,
                                    const std::vector<float>& fparam,
                                    const std::vector<float>& aparam) {
  compute_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                     coord, atype, box, fparam, aparam);
}
#endif
