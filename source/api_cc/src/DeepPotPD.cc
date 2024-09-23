// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PADDLE
#include "DeepPotPD.h"

#include <cstdint>
#include <stdexcept>
#include <numeric>

#include "AtomMap.h"
#include "device.h"
#include "common.h"
#include "paddle/include/paddle_inference_api.h"
// #include "glog/logging.h"

using namespace deepmd;

template <typename MODELTYPE, typename VALUETYPE>
static void run_model(
    std::vector<ENERGYTYPE>& dener,
    std::vector<VALUETYPE>& dforce_,
    std::vector<VALUETYPE>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const AtomMap& atommap,
    const int nframes,
    const int nghost = 0) {
  // printf("run_model 1 st\n");
  unsigned nloc = atommap.get_type().size();
  unsigned nall = nloc + nghost;
  // printf("nloc = %d, nall = %d\n", nloc, nall);
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

  /* Running inference */
  // printf("Running inference st\n");
  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference failed");
  }
  // printf("Running inference ed\n");

  auto output_names = predictor->GetOutputNames();
  // for (auto &name: output_names)
  // {
  //   printf("output name: %s, shape: [", name.c_str());
  //   auto shape = predictor->GetOutputHandle(name)->shape();
  //   for (auto &dd: shape)
  //   {
  //     printf("%d, ", dd);
  //   }
  //   printf("]\n");
  // }
  auto output_e = predictor->GetOutputHandle(output_names[1]);
  auto output_f = predictor->GetOutputHandle(output_names[2]);
  auto output_virial_tensor = predictor->GetOutputHandle(output_names[4]);

  // 获取 Output paddle::Tensor 的维度信息
  std::vector<int> output_energy_shape = output_e->shape();
  int output_energy_size =
      std::accumulate(output_energy_shape.begin(), output_energy_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_force_shape = output_f->shape();
  int output_force_size =
      std::accumulate(output_force_shape.begin(), output_force_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_virial_shape = output_virial_tensor->shape();
  int output_virial_size =
      std::accumulate(output_virial_shape.begin(), output_virial_shape.end(), 1,
                      std::multiplies<int>());
  // for (int i=0; i<output_energy_shape.size(); ++i)
  //   printf("output_energy_shape.shape[%d] = %d\n", i, output_energy_shape[i]);
  // for (int i=0; i<output_force_shape.size(); ++i)
  //   printf("output_force_shape.shape[%d] = %d\n", i, output_force_shape[i]);
  // for (int i=0; i<output_virial_shape.size(); ++i)
  //   printf("output_virial_shape.shape[%d] = %d\n", i, output_virial_shape[i]);

  // get data of output_energy
  // printf("Starting copy back to CPU\n");
  std::vector<ENERGYTYPE> oe;
  // printf("Resize st\n");
  oe.resize(output_energy_size);
  // printf("Resize ed\n");
  // printf("CopytoCpu st\n");
  output_e->CopyToCpu(oe.data());
  // printf("Resize st\n");
  // printf("CopytoCpu ed\n");
  // get data of output_force
  // printf("of\n");
  std::vector<MODELTYPE> of;
  of.resize(output_force_size);
  output_f->CopyToCpu(of.data());
  // get data of output_virial
  // printf("oav\n");
  std::vector<VALUETYPE> oav;
  oav.resize(output_virial_size);
  // printf("oav 2\n");
  output_virial_tensor->CopyToCpu(oav.data());
  // printf("oav 22\n");

  // printf("dvirial\n");
  std::vector<VALUETYPE> dforce(nframes * 3 * nall);
  dvirial.resize(nframes * 9);
  for (int ii = 0; ii < nframes; ++ii) {
    // printf("oe[%d] = %.5lf\n", ii, oe[ii]);
    dener[ii] = oe[ii];
  }
  for (int ii = 0; ii < nframes * nall * 3; ++ii) {
    dforce[ii] = of[ii];
  }
  // set dvirial to zero, prevent input vector is not zero (#1123)
  // printf("fill\n");
  std::fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.);
  for (int kk = 0; kk < nframes; ++kk) {
    for (int ii = 0; ii < nall; ++ii) {
      dvirial[kk * 9 + 0] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 0];
      dvirial[kk * 9 + 1] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 1];
      dvirial[kk * 9 + 2] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 2];
      dvirial[kk * 9 + 3] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 3];
      dvirial[kk * 9 + 4] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 4];
      dvirial[kk * 9 + 5] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 5];
      dvirial[kk * 9 + 6] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 6];
      dvirial[kk * 9 + 7] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 7];
      dvirial[kk * 9 + 8] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 8];
    }
  }
  dforce_ = dforce;
  // printf("atommap.backward\n");
  atommap.backward<VALUETYPE>(dforce_.begin(), dforce.begin(), 3, nframes,
                              nall);
  // printf("run_model 1 ed\n");
}

template void run_model<double, double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<double, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
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
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost = 0) {
  // printf("run_model 2\n");
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

  /* Running inference */
  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference failed");
  }

  /* Get output handles*/
  auto output_names = predictor->GetOutputNames();
  auto output_ae = predictor->GetOutputHandle(output_names[0]);
  auto output_av = predictor->GetOutputHandle(output_names[1]);
  auto output_e = predictor->GetOutputHandle(output_names[4]);
  auto output_f = predictor->GetOutputHandle(output_names[5]);

  // 获取 Output paddle::Tensor 的维度信息
  std::vector<int> output_atom_ener_shape = output_ae->shape();
  int output_atom_ener_size =
      std::accumulate(output_atom_ener_shape.begin(),
                      output_atom_ener_shape.end(), 1, std::multiplies<int>());
  std::vector<int> output_atom_virial_shape = output_av->shape();
  int output_atom_virial_size =
      std::accumulate(output_atom_virial_shape.begin(), output_atom_virial_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_energy_shape = output_e->shape();
  int output_energy_size =
      std::accumulate(output_energy_shape.begin(), output_energy_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_force_shape = output_f->shape();
  int output_force_size =
      std::accumulate(output_force_shape.begin(), output_force_shape.end(), 1,
                      std::multiplies<int>());

  // get data of output_atom_ener
  std::vector<VALUETYPE> oae;
  oae.resize(output_atom_ener_size);
  output_ae->CopyToCpu(oae.data());
  // get data of output_atom_virial
  std::vector<VALUETYPE> oav;
  oav.resize(output_atom_virial_size);
  output_av->CopyToCpu(oav.data());
  // get data of output_energy
  std::vector<VALUETYPE> oe;
  oe.resize(output_energy_size);
  output_e->CopyToCpu(oe.data());
  // get data of output_force
  std::vector<VALUETYPE> of;
  of.resize(output_force_size);
  output_f->CopyToCpu(of.data());

  std::vector<VALUETYPE> dforce(nframes * 3 * nall);
  std::vector<VALUETYPE> datom_energy(nframes * nall, 0);
  std::vector<VALUETYPE> datom_virial(nframes * 9 * nall);
  dvirial.resize(nframes * 9);
  for (int ii = 0; ii < nframes; ++ii) {
    dener[ii] = oe[ii];
  }
  for (int ii = 0; ii < nframes * nall * 3; ++ii) {
    dforce[ii] = of[ii];
  }
  for (int ii = 0; ii < nframes; ++ii) {
    for (int jj = 0; jj < nloc; ++jj) {
      datom_energy[ii * nall + jj] = oae[ii * nloc + jj];
    }
  }
  for (int ii = 0; ii < nframes * nall * 9; ++ii) {
    datom_virial[ii] = oav[ii];
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
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<double, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, double>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, float>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

// end multiple frames

// start single frame

template <typename MODELTYPE, typename VALUETYPE>
static void run_model(
    ENERGYTYPE& dener,
    std::vector<VALUETYPE>& dforce_,
    std::vector<VALUETYPE>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost = 0) {
  assert(nframes == 1);
  std::vector<ENERGYTYPE> dener_(1);
  // call multi-frame version
  run_model<MODELTYPE, VALUETYPE>(dener_, dforce_, dvirial, predictor,
                                  atommap, nframes, nghost);
  dener = dener_[0];
}

template void run_model<double, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<double, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, Tensor>>& input_tensors,
    const AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, Tensor>>& input_tensors,
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
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    const deepmd::AtomMap& atommap,
    const int nframes = 1,
    const int nghost = 0) {
  assert(nframes == 1);
  std::vector<ENERGYTYPE> dener_(1);
  // call multi-frame version
  run_model<MODELTYPE, VALUETYPE>(dener_, dforce_, dvirial, datom_energy_,
                                  datom_virial_, predictor,//, input_tensors,
                                  atommap, nframes, nghost);
  dener = dener_[0];
}

template void run_model<double, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<double, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, double>(
    ENERGYTYPE& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

template void run_model<float, float>(
    ENERGYTYPE& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::shared_ptr<paddle_infer::Predictor>& predictor,
    // const std::vector<std::pair<std::string, paddle::Tensor>>& input_tensors,
    const deepmd::AtomMap& atommap,
    const int nframes,
    const int nghost);

// end single frame

DeepPotPD::DeepPotPD() : inited(false) {}

DeepPotPD::DeepPotPD(const std::string& model,
                 const int& gpu_rank,
                 const std::string& file_content)
    : inited(false) {
  init(model, gpu_rank, file_content);
}

void DeepPotPD::init(const std::string& model,
                   const int& gpu_rank,
                   const std::string& file_content) {
  // std::cout << ("** Access here.") << std::endl;
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  // deepmd::load_op_library();
  int gpu_num = 1; // hard code here
  if (gpu_num > 0) {
    gpu_id = gpu_rank % gpu_num;
  } else {
    gpu_id = 0;
  }
  std::string pdmodel_path = "";
  std::string pdiparams_path = "";
  bool use_paddle_inference = false;
  bool use_pir = false;
  if (model.find(".json") != std::string::npos) {
    use_pir = true;
    pdmodel_path = model;
    std::string tmp = model;
    pdiparams_path = tmp.replace(model.find(".json"), 5, std::string(".pdiparams"));
    use_paddle_inference = true;
  } else if (model.find(".pdmodel") != std::string::npos){
    pdmodel_path = model;
    std::string tmp = model;
    pdiparams_path = tmp.replace(model.find(".pdmodel"), 8, std::string(".pdiparams"));
    use_paddle_inference = true;
  } else {
    throw "[Error] Not found any inference model in";
  }
  int math_lib_num_threads = 1;

  if (use_paddle_inference) {
    // printf("***** creating paddle predictor\n");
    config = std::make_shared<paddle_infer::Config>();
    config->DisableGlogInfo();
    // config->SwitchIrDebug(true);
    if (use_pir) {
      config->EnableNewExecutor(true);
      config->EnableNewIR(true);
    }
    config->SetModel(pdmodel_path, pdiparams_path);
    // config->SwitchIrOptim(true);
    config->EnableUseGpu(8192, 0);
    // std::cout << "IR Optim is: " << config->ir_optim() << std::endl;
    // config->EnableMKLDNN();
    // config->EnableMemoryOptim();
    // config->EnableProfile();
    predictor = paddle_infer::CreatePredictor(*config);
    // printf("***** created paddle predictor\n");
  }
  /* water se_e2_a
  tensorflow::DT_DOUBLE = 2
  tensorflow::DT_FLOAT = 1
  paddle_infer::DataType::FLOAT64 = 7
  paddle_infer::DataType::FLOAT32 = 0
  * st_model.descrpt.buffer_rcut.name = generated_tensor_0
  * st_model.descrpt.buffer_ntypes.name = generated_tensor_2
  * st_model.fitting.buffer_dfparam.name = generated_tensor_9
  * st_model.fitting.buffer_daparam.name = generated_tensor_10
  [buffer_t_type, [3]] generated name in static_model is: generated_tensor_12
  [buffer_t_mt, [4]] generated name in static_model is: generated_tensor_13
  [buffer_t_ver, [1]] generated name in static_model is: generated_tensor_14
  [descrpt.buffer_rcut, []] generated name in static_model is:
  generated_tensor_0 [descrpt.buffer_ntypes_spin, []] generated name in
  static_model is: generated_tensor_1 [descrpt.buffer_ntypes, []] generated
  name in static_model is: generated_tensor_2 [descrpt.avg_zero, [2, 552]]
  generated name in static_model is: eager_tmp_0 [descrpt.std_ones, [2, 552]]
  generated name in static_model is: eager_tmp_1 [descrpt.t_rcut, []]
  generated name in static_model is: generated_tensor_3 [descrpt.t_rcut, []]
  generated name in static_model is: generated_tensor_3 [descrpt.t_rcut, []]
  generated name in static_model is: generated_tensor_3 [descrpt.t_ntypes, []]
  generated name in static_model is: generated_tensor_4 [descrpt.t_ntypes, []]
  generated name in static_model is: generated_tensor_4 [descrpt.t_ntypes, []]
  generated name in static_model is: generated_tensor_4 [descrpt.t_ndescrpt,
  []] generated name in static_model is: generated_tensor_5 [descrpt.t_sel,
  [2]] generated name in static_model is: generated_tensor_6 [descrpt.t_avg,
  [2, 552]] generated name in static_model is: generated_tensor_7
  [descrpt.t_std, [2, 552]] generated name in static_model is:
  generated_tensor_8 [fitting.buffer_dfparam, []] generated name in
  static_model is: generated_tensor_9 [fitting.buffer_daparam, []] generated
  name in static_model is: generated_tensor_10
  **/
  /* spin se_e2_a
  [buffer_tmap, [4]] generated name in static_model is: generated_tensor_14
  [buffer_model_type, [4]] generated name in static_model is:
  generated_tensor_15 [buffer_model_version, [1]] generated name in
  static_model is: generated_tensor_16 [descrpt.buffer_rcut, []] generated
  name in static_model is: generated_tensor_3 [descrpt.buffer_ntypes, []]
  generated name in static_model is: generated_tensor_4 [descrpt.avg_zero, [3,
  720]] generated name in static_model is: eager_tmp_0 [descrpt.std_ones, [3,
  720]] generated name in static_model is: eager_tmp_1 [descrpt.t_rcut, []]
  generated name in static_model is: generated_tensor_5 [descrpt.buffer_sel,
  [3]] generated name in static_model is: generated_tensor_6
  [descrpt.buffer_ndescrpt, []] generated name in static_model is:
  generated_tensor_7 [descrpt.buffer_original_sel, [3]] generated name in
  static_model is: generated_tensor_8 [descrpt.t_avg, [3, 720]] generated name
  in static_model is: generated_tensor_9 [descrpt.t_std, [3, 720]] generated
  name in static_model is: generated_tensor_10
  [descrpt.spin.buffer_ntypes_spin, [1]] generated name in static_model is:
  generated_tensor_0 [descrpt.spin.buffer_virtual_len, [1, 1]] generated name
  in static_model is: generated_tensor_1 [descrpt.spin.buffer_spin_norm, [1,
  1]] generated name in static_model is: generated_tensor_2
  [fitting.buffer_dfparam, []] generated name in static_model is:
  generated_tensor_11 [fitting.buffer_daparam, []] generated name in
  static_model is: generated_tensor_12 [fitting.t_bias_atom_e, [2]] generated
  name in static_model is: generated_tensor_13
  */
  // dtype = predictor_get_dtype(predictor, "generated_tensor_0"); // hard code
  // auto dtype = paddle_infer::DataType::FLOAT64;
  // if (dtype == paddle_infer::DataType::FLOAT64) {
  //   rcut = paddle_get_scalar<double>("generated_tensor_0");
  // } else {
  //   rcut = 3.18;
  // }
  rcut = double(6.0);
  ntypes = 2;
  ntypes_spin = 0;
  dfparam = 0;
  daparam = 0;
  aparam_nall = false;

  inited = true;
  // if (!model_compatable(model_version)) {
  //   throw deepmd::deepmd_exception(
  //       "incompatable model: version " + model_version +
  //       " in graph, but version " + global_model_version +
  //       " supported "
  //       "See https://deepmd.rtfd.io/compatability/ for details.");
  // }
  // printf("***** initialized finished\n");
}

DeepPotPD::~DeepPotPD() {}

// void DeepPotPD::print_summary(const std::string& pre) const {
//   deepmd::print_summary(pre);
// }

// template <class VT>
// VT DeepPotPD::get_scalar(const std::string& name) const {
//   return session_get_scalar<VT>(session, name);
// }

// template <class VT>
// VT DeepPotPD::paddle_get_scalar(const std::string& name) const {
//   return predictor_get_scalar<VT>(predictor, name);
// }

template <typename VALUETYPE>
void DeepPotPD::validate_fparam_aparam(
    const int nframes,
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

template void DeepPotPD::validate_fparam_aparam<double>(
    const int nframes,
    const int& nloc,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam) const;

template void DeepPotPD::validate_fparam_aparam<float>(
    const int nframes,
    const int& nloc,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam) const;

template <typename VALUETYPE>
void DeepPotPD::tile_fparam_aparam(std::vector<VALUETYPE>& out_param,
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

template void DeepPotPD::tile_fparam_aparam<double>(
    std::vector<double>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<double>& param) const;

template void DeepPotPD::tile_fparam_aparam<float>(
    std::vector<float>& out_param,
    const int& nframes,
    const int& dparam,
    const std::vector<float>& param) const;

// ENERGYVTYPE: std::vector<ENERGYTYPE> or ENERGYTYPE

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute(ENERGYVTYPE& dener,
                        std::vector<VALUETYPE>& dforce_,
                        std::vector<VALUETYPE>& dvirial,
                        std::vector<VALUETYPE>& datom_energy_,
                        std::vector<VALUETYPE>& datom_virial_,
                        const std::vector<VALUETYPE>& dcoord_,
                        const std::vector<int>& datype_,
                        const std::vector<VALUETYPE>& dbox,
                        const std::vector<VALUETYPE>& fparam_,
                        const std::vector<VALUETYPE>& aparam_,
                        const bool atomic) {
  // printf("compute 1\n");
  // if datype.size is 0, not clear nframes; but 1 is just ok
  int nframes = datype_.size() > 0 ? (dcoord_.size() / 3 / datype_.size()) : 1;
  atommap = deepmd::AtomMap(datype_.begin(), datype_.end());
  int nloc = datype_.size();
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  // std::vector<std::pair<std::string, paddle::Tensor>> input_tensors;

  if (dtype == paddle_infer::DataType::FLOAT64) {
    int ret = predictor_input_tensors<double>(predictor, dcoord_, ntypes,
                                              datype_, dbox, cell_size, fparam,
                                              aparam, atommap, aparam_nall);
    if (atomic) {
      run_model<double>(dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor,
                               atommap, nframes);
    } else {
      run_model<double>(dener, dforce_, dvirial, predictor,
                               atommap, nframes);
    }
  } else {
    int ret = predictor_input_tensors<float>(predictor, dcoord_, ntypes, datype_, dbox, cell_size, fparam, aparam,
                                              atommap, aparam_nall);
    if (atomic) {
      run_model<float>(dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor,
                              atommap, nframes);
    } else {
      run_model<float>(dener, dforce_, dvirial, predictor,
                              atommap, nframes);
    }
  }
}

// template void DeepPotPD::compute<double, ENERGYTYPE>(
//     ENERGYTYPE& dener,
//     std::vector<double>& dforce_,
//     std::vector<double>& dvirial,
//     std::vector<double>& datom_energy_,
//     std::vector<double>& datom_virial_,
//     const std::vector<double>& dcoord_,
//     const std::vector<int>& datype_,
//     const std::vector<double>& dbox,
//     const std::vector<double>& fparam,
//     const std::vector<double>& aparam,
//     const bool atomic);

// template void DeepPotPD::compute<float, ENERGYTYPE>(
//     ENERGYTYPE& dener,
//     std::vector<float>& dforce_,
//     std::vector<float>& dvirial,
//     std::vector<float>& datom_energy_,
//     std::vector<float>& datom_virial_,
//     const std::vector<float>& dcoord_,
//     const std::vector<int>& datype_,
//     const std::vector<float>& dbox,
//     const std::vector<float>& fparam,
//     const std::vector<float>& aparam,
//     const bool atomic);

template void DeepPotPD::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<double>& dforce_,
    std::vector<double>& dvirial,
    std::vector<double>& datom_energy_,
    std::vector<double>& datom_virial_,
    const std::vector<double>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<double>& dbox,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepPotPD::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& dener,
    std::vector<float>& dforce_,
    std::vector<float>& dvirial,
    std::vector<float>& datom_energy_,
    std::vector<float>& datom_virial_,
    const std::vector<float>& dcoord_,
    const std::vector<int>& datype_,
    const std::vector<float>& dbox,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

std::vector<int> createNlistTensor(const std::vector<std::vector<int>>& data) {
  std::vector<int> ret;

  for (const auto& row : data) {
    ret.insert(ret.end(), row.begin(), row.end());
  }

  return ret;
}

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute(ENERGYVTYPE& dener,
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
                        const std::vector<VALUETYPE>& aparam__,
                        const bool atomic) {
  /*参考pytorch的推理代码如下*/
  int natoms = datype_.size();
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam_, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, dcoord_, datype_, aparam__,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  int nframes = 1;
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  auto coord_wrapped_Tensor = predictor->GetInputHandle("coord");
  coord_wrapped_Tensor->Reshape({1, nall_real, 3});
  coord_wrapped_Tensor->CopyFromCpu(coord_wrapped.data());

  auto atype_Tensor = predictor->GetInputHandle("atype");
  atype_Tensor->Reshape({1, nall_real});
  atype_Tensor->CopyFromCpu(datype.data());

  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
  }
  std::vector<int> firstneigh = createNlistTensor(nlist_data.jlist);
  auto firstneigh_tensor = predictor->GetInputHandle("nlist");
  firstneigh_tensor->Reshape({1, nloc, firstneigh.size() / nloc});
  firstneigh_tensor->CopyFromCpu(firstneigh.data());


  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference failed");
  }
  auto output_names = predictor->GetOutputNames();

  auto print_shape = [](const std::vector<int> &shape, const std::string &name=""){
    printf("shape of %s: [", name.c_str());
    for (int i=0; i<shape.size(); ++i)
    {
      printf("%d%c", shape[i], ",]"[i==shape.size()-1]);
    }
    printf("\n");
  };
  auto output_e = predictor->GetOutputHandle(output_names[1]);
  auto output_f = predictor->GetOutputHandle(output_names[2]);
  auto output_virial_tensor = predictor->GetOutputHandle(output_names[3]);
  // print_shape(output_e->shape(), "ener");
  // print_shape(output_f->shape(), "force");
  // print_shape(output_virial_tensor->shape(), "virial");
  std::vector<int> output_energy_shape = output_e->shape();
  int output_energy_size =
      std::accumulate(output_energy_shape.begin(), output_energy_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_force_shape = output_f->shape();
  int output_force_size =
      std::accumulate(output_force_shape.begin(), output_force_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<int> output_virial_shape = output_virial_tensor->shape();
  int output_virial_size =
      std::accumulate(output_virial_shape.begin(), output_virial_shape.end(), 1,
                      std::multiplies<int>());
  std::vector<ENERGYTYPE> oe;
  oe.resize(output_energy_size);
  output_e->CopyToCpu(oe.data());

  std::vector<VALUETYPE> of;
  of.resize(output_force_size);
  output_f->CopyToCpu(of.data());

  std::vector<VALUETYPE> oav;
  oav.resize(output_virial_size);
  output_virial_tensor->CopyToCpu(oav.data());

  dvirial.resize(nframes * 9);
  dener.assign(oe.begin(), oe.end());
  dforce.resize(nframes * 3 * nall);
  for (int ii = 0; ii < nframes * nall * 3; ++ii) {
    dforce[ii] = of[ii];
  }
  std::fill(dvirial.begin(), dvirial.end(), (VALUETYPE)0.);
  dvirial.assign(oav.begin(), oav.end());
  // for (int kk = 0; kk < nframes; ++kk) {
  //   for (int ii = 0; ii < nall; ++ii) {
  //     dvirial[kk * 9 + 0] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 0];
  //     dvirial[kk * 9 + 1] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 1];
  //     dvirial[kk * 9 + 2] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 2];
  //     dvirial[kk * 9 + 3] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 3];
  //     dvirial[kk * 9 + 4] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 4];
  //     dvirial[kk * 9 + 5] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 5];
  //     dvirial[kk * 9 + 6] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 6];
  //     dvirial[kk * 9 + 7] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 7];
  //     dvirial[kk * 9 + 8] += (VALUETYPE)1.0 * oav[kk * nall * 9 + 9 * ii + 8];
  //   }
  // }
  // bkw map
  dforce_.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(dforce_, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
}

// template void DeepPotPD::compute<double, ENERGYTYPE>(
//     ENERGYTYPE& dener,
//     std::vector<double>& dforce_,
//     std::vector<double>& dvirial,
//     std::vector<double>& datom_energy_,
//     std::vector<double>& datom_virial_,
//     const std::vector<double>& dcoord_,
//     const std::vector<int>& datype_,
//     const std::vector<double>& dbox,
//     const int nghost,
//     const InputNlist& lmp_list,
//     const int& ago,
//     const std::vector<double>& fparam,
//     const std::vector<double>& aparam_,
//     const bool atomic);

// template void DeepPotPD::compute<float, ENERGYTYPE>(
//     ENERGYTYPE& dener,
//     std::vector<float>& dforce_,
//     std::vector<float>& dvirial,
//     std::vector<float>& datom_energy_,
//     std::vector<float>& datom_virial_,
//     const std::vector<float>& dcoord_,
//     const std::vector<int>& datype_,
//     const std::vector<float>& dbox,
//     const int nghost,
//     const InputNlist& lmp_list,
//     const int& ago,
//     const std::vector<float>& fparam,
//     const std::vector<float>& aparam_,
//     const bool atomic);

template void DeepPotPD::compute<double, std::vector<ENERGYTYPE>>(
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
    const std::vector<double>& aparam_,
    const bool atomic);

template void DeepPotPD::compute<float, std::vector<ENERGYTYPE>>(
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
    const std::vector<float>& aparam_,
    const bool atomic);

// mixed type

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute_mixed_type(ENERGYVTYPE& dener,
                                   std::vector<VALUETYPE>& dforce_,
                                   std::vector<VALUETYPE>& dvirial,
                                   std::vector<VALUETYPE>& datom_energy_,
                                   std::vector<VALUETYPE>& datom_virial_,
                                   const int& nframes,
                                   const std::vector<VALUETYPE>& dcoord_,
                                   const std::vector<int>& datype_,
                                   const std::vector<VALUETYPE>& dbox,
                                   const std::vector<VALUETYPE>& fparam_,
                                   const std::vector<VALUETYPE>& aparam_,
                                   const bool atomic) {
  int nloc = datype_.size() / nframes;
  // here atommap only used to get nloc
  atommap = deepmd::AtomMap(datype_.begin(), datype_.begin() + nloc);
  std::vector<VALUETYPE> fparam;
  std::vector<VALUETYPE> aparam;
  validate_fparam_aparam(nframes, nloc, fparam_, aparam_);
  tile_fparam_aparam(fparam, nframes, dfparam, fparam_);
  tile_fparam_aparam(aparam, nframes, nloc * daparam, aparam_);

  if (dtype == paddle_infer::DataType::FLOAT64) {
    int nloc = predictor_input_tensors_mixed_type<double>(
        predictor, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap, aparam_nall);
    if (atomic) {
      run_model<double>(dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor,
                        atommap, nframes);
    } else {
      run_model<double>(dener, dforce_, dvirial, predictor,
                        atommap, nframes);
    }
  } else {
    int nloc = predictor_input_tensors_mixed_type<double>(
        predictor, nframes, dcoord_, ntypes, datype_, dbox, cell_size,
        fparam, aparam, atommap, aparam_nall);
    if (atomic) {
      run_model<float>(dener, dforce_, dvirial, datom_energy_, datom_virial_, predictor,
                       atommap, nframes);
    } else {
      run_model<float>(dener, dforce_, dvirial, predictor, atommap,
                       nframes);
    }
  }
}

template void DeepPotPD::compute_mixed_type<double, ENERGYTYPE>(
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
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepPotPD::compute_mixed_type<float, ENERGYTYPE>(
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
    const std::vector<float>& aparam,
    const bool atomic);

template void DeepPotPD::compute_mixed_type<double, std::vector<ENERGYTYPE>>(
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
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepPotPD::compute_mixed_type<float, std::vector<ENERGYTYPE>>(
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
    const std::vector<float>& aparam,
    const bool atomic);


template <class VT>
VT DeepPotPD::get_scalar(const std::string& name) const {
  return predictor_get_scalar<VT>(predictor, name);
}

void DeepPotPD::get_type_map(std::string& type_map) {
  type_map = "O H ";
  // type_map = predictor_get_scalar<std::string>(predictor, "type_map");
}

// forward to template method
void DeepPotPD::computew(std::vector<double>& ener,
                         std::vector<double>& force,
                         std::vector<double>& virial,
                         std::vector<double>& atom_energy,
                         std::vector<double>& atom_virial,
                         const std::vector<double>& coord,
                         const std::vector<int>& atype,
                         const std::vector<double>& box,
                         const std::vector<double>& fparam,
                         const std::vector<double>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam, atomic);
}
void DeepPotPD::computew(std::vector<double>& ener,
                         std::vector<float>& force,
                         std::vector<float>& virial,
                         std::vector<float>& atom_energy,
                         std::vector<float>& atom_virial,
                         const std::vector<float>& coord,
                         const std::vector<int>& atype,
                         const std::vector<float>& box,
                         const std::vector<float>& fparam,
                         const std::vector<float>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          fparam, aparam, atomic);
}
void DeepPotPD::computew(std::vector<double>& ener,
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
                         const std::vector<double>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam, atomic);
}
void DeepPotPD::computew(std::vector<double>& ener,
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
                         const std::vector<float>& aparam,
                         const bool atomic) {
  compute(ener, force, virial, atom_energy, atom_virial, coord, atype, box,
          nghost, inlist, ago, fparam, aparam, atomic);
}
void DeepPotPD::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<double>& force,
                                    std::vector<double>& virial,
                                    std::vector<double>& atom_energy,
                                    std::vector<double>& atom_virial,
                                    const int& nframes,
                                    const std::vector<double>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<double>& box,
                                    const std::vector<double>& fparam,
                                    const std::vector<double>& aparam,
                                    const bool atomic) {
  compute_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                     coord, atype, box, fparam, aparam, atomic);
}
void DeepPotPD::computew_mixed_type(std::vector<double>& ener,
                                    std::vector<float>& force,
                                    std::vector<float>& virial,
                                    std::vector<float>& atom_energy,
                                    std::vector<float>& atom_virial,
                                    const int& nframes,
                                    const std::vector<float>& coord,
                                    const std::vector<int>& atype,
                                    const std::vector<float>& box,
                                    const std::vector<float>& fparam,
                                    const std::vector<float>& aparam,
                                    const bool atomic) {
  compute_mixed_type(ener, force, virial, atom_energy, atom_virial, nframes,
                     coord, atype, box, fparam, aparam, atomic);
}
#endif
