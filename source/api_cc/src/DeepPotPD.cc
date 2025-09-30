// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_PADDLE
#include "DeepPotPD.h"

#include <cstdint>
#include <numeric>

#include "common.h"
#include "device.h"
#include "errors.h"

using namespace deepmd;

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

class Logger {
 public:
  enum Level { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3 };

 private:
  static Level minLevel;
  static bool colorEnabled;
  static bool showTimestamp;

  static const char* getColorCode(Level level) {
    if (!colorEnabled) {
      return "";
    }
    switch (level) {
      case DEBUG:
        return "\033[1;36m";
      case INFO:
        return "\033[1;32m";
      case WARNING:
        return "\033[1;33m";
      case ERROR:
        return "\033[1;31m";
      default:
        return "";
    }
  }

  static const char* getResetCode() { return colorEnabled ? "\033[0m" : ""; }

  static const char* getLevelName(Level level) {
    switch (level) {
      case DEBUG:
        return "DEBUG";
      case INFO:
        return "INFO";
      case WARNING:
        return "WARNING";
      case ERROR:
        return "ERROR";
      default:
        return "UNKNOWN";
    }
  }

  static std::string getCurrentTime() {
    if (!showTimestamp) {
      return "";
    }

    std::time_t now = std::time(0);
    std::tm* ltm = std::localtime(&now);

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(4) << (1900 + ltm->tm_year) << "-"
        << std::setw(2) << (1 + ltm->tm_mon) << "-" << std::setw(2)
        << ltm->tm_mday << " " << std::setw(2) << ltm->tm_hour << ":"
        << std::setw(2) << ltm->tm_min << ":" << std::setw(2) << ltm->tm_sec;
    return oss.str();
  }

 public:
  class LogStream {
   private:
    std::ostringstream oss;
    Level level;
    bool shouldLog;

   public:
    LogStream(Level lvl) : level(lvl), shouldLog(lvl >= minLevel) {
      if (shouldLog) {
        std::string timestamp = getCurrentTime();
        if (!timestamp.empty()) {
          oss << "[" << timestamp << "] ";
        }
        oss << getColorCode(level) << "[" << getLevelName(level) << "]"
            << getResetCode() << " ";
      }
    }

    ~LogStream() {
      if (shouldLog) {
        std::cout << oss.str() << std::flush;
      }
    }

    template <typename T>
    LogStream& operator<<(const T& value) {
      if (shouldLog) {
        oss << value;
      }
      return *this;
    }

    LogStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
      if (shouldLog) {
        oss << manip;
      }
      return *this;
    }

    LogStream(const LogStream&) = delete;
    LogStream& operator=(const LogStream&) = delete;
    LogStream(LogStream&& other) noexcept
        : oss(std::move(other.oss)),
          level(other.level),
          shouldLog(other.shouldLog) {}

    LogStream& operator=(LogStream&& other) noexcept {
      if (this != &other) {
        oss = std::move(other.oss);
        level = other.level;
        shouldLog = other.shouldLog;
      }
      return *this;
    }
  };

  static void setLevel(Level level) { minLevel = level; }
  static void enableColor(bool enable = true) { colorEnabled = enable; }
  static void enableTimestamp(bool enable = true) { showTimestamp = enable; }
  static Level getLevel() { return minLevel; }
  static bool isColorEnabled() { return colorEnabled; }
  static bool isTimestampEnabled() { return showTimestamp; }

  static LogStream debug() { return LogStream(DEBUG); }
  static LogStream info() { return LogStream(INFO); }
  static LogStream warning() { return LogStream(WARNING); }
  static LogStream error() { return LogStream(ERROR); }
};

Logger::Level Logger::minLevel = Logger::INFO;
bool Logger::colorEnabled = true;
bool Logger::showTimestamp = true;

namespace logg {
inline Logger::LogStream debug() { return Logger::debug(); }
inline Logger::LogStream info() { return Logger::info(); }
inline Logger::LogStream warning() { return Logger::warning(); }
inline Logger::LogStream error() { return Logger::error(); }

inline void setLevel(Logger::Level level) { Logger::setLevel(level); }
inline void enableColor(bool enable = true) { Logger::enableColor(enable); }
inline void enableTimestamp(bool enable = true) {
  Logger::enableTimestamp(enable);
}
}  // namespace logg

void fillNlistTensor(const std::vector<std::vector<int>>& data,
                     std::unique_ptr<paddle_infer::Tensor>& flat_tensor) {
  size_t total_size = 0;
  for (const auto& row : data) {
    total_size += row.size();
  }
  std::vector<int> flat_data;
  flat_data.reserve(total_size);
  for (const auto& row : data) {
    flat_data.insert(flat_data.end(), row.begin(), row.end());
  }

  int nloc = data.size();
  int nnei = nloc > 0 ? total_size / nloc : 0;
  flat_tensor->Reshape({1, nloc, nnei});
  flat_tensor->CopyFromCpu(flat_data.data());
}
DeepPotPD::DeepPotPD() : inited(false) {}
DeepPotPD::DeepPotPD(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content)
    : inited(false) {
  try {
    init(model, gpu_rank, file_content);
  } catch (...) {
    // Clean up and rethrow, as the destructor will not be called
    throw;
  }
}
void DeepPotPD::init(const std::string& model,
                     const int& gpu_rank,
                     const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }
  deepmd::load_op_library();
  // NOTE: Only support 1 GPU now.
  int gpu_num = 1;
  if (gpu_num > 0) {
    gpu_id = gpu_rank % gpu_num;
  } else {
    gpu_id = 0;
  }

  // initialize inference config
  config = std::make_shared<paddle_infer::Config>();
  config->DisableGlogInfo();
  config->EnableNewExecutor(true);
  config->EnableNewIR(true);
  config->EnableCustomPasses({"add_shadow_output_after_dead_parameter_pass"},
                             true);
  // config->SwitchIrOptim(false);

  // initialize inference config_fl
  config_fl = std::make_shared<paddle_infer::Config>();
  config_fl->DisableGlogInfo();
  config_fl->EnableNewExecutor(true);
  config_fl->EnableNewIR(true);
  config_fl->EnableCustomPasses({"add_shadow_output_after_dead_parameter_pass"},
                                true);
  // config_fl->SwitchIrOptim(false);

  // loading inference model
  std::string pdmodel_path, fl_pdmodel_path;
  std::string pdiparams_path, fl_pdiparams_path;
  if (model.find(".json") != std::string::npos) {
    // load inference of model.forward
    pdmodel_path = model;
    pdiparams_path = model;
    pdiparams_path.replace(pdiparams_path.find(".json"), 5,
                           std::string(".pdiparams"));

    // load inference of model.forward_lower
    fl_pdmodel_path = pdmodel_path;
    size_t last_slash_pos = fl_pdmodel_path.rfind('/');
    size_t dot_pos = fl_pdmodel_path.rfind('.');
    std::string filename = fl_pdmodel_path.substr(last_slash_pos + 1,
                                                  dot_pos - last_slash_pos - 1);
    filename = filename + "." + "forward_lower";
    fl_pdmodel_path.replace(last_slash_pos + 1, dot_pos - last_slash_pos - 1,
                            filename);

    fl_pdiparams_path = pdiparams_path;
    last_slash_pos = fl_pdiparams_path.rfind('/');
    dot_pos = fl_pdiparams_path.rfind('.');
    filename = fl_pdiparams_path.substr(last_slash_pos + 1,
                                        dot_pos - last_slash_pos - 1);
    filename = filename + "." + "forward_lower";
    fl_pdiparams_path.replace(last_slash_pos + 1, dot_pos - last_slash_pos - 1,
                              filename);

  } else if (model.find(".pdmodel") != std::string::npos) {
    pdmodel_path = model;
    pdiparams_path = model;
    pdiparams_path.replace(pdiparams_path.find(".pdmodel"), 8,
                           std::string(".pdiparams"));
  } else {
    throw deepmd::deepmd_exception("Given inference model: " + model +
                                   " do not exist, please check it.");
  }
  const char* use_cuda_toolkit = std::getenv("USE_CUDA_TOOLKIT");
  gpu_enabled = (use_cuda_toolkit && (std::string(use_cuda_toolkit) == "1"));
  config->SetModel(pdmodel_path, pdiparams_path);
  config_fl->SetModel(fl_pdmodel_path, fl_pdiparams_path);
  if (!gpu_enabled) {
    config->DisableGpu();
    config_fl->DisableGpu();
    logg::info() << "load model from: " << model << " to cpu " << std::endl;
  } else {
    config->EnableUseGpu(4096, 0);
    config_fl->EnableUseGpu(4096, 0);
    logg::info() << "load model from: " << model << " to gpu:" << gpu_id
                 << std::endl;
  }
  if (config->cinn_enabled()) {
    logg::info() << "model.forward will be compiled with cinn." << std::endl;
  } else {
    logg::info() << "NOTE: You can try: \n'export FLAGS_prim_all=true"
                    " FLAGS_enable_pir_in_executor=1"
                    " FLAGS_prim_enable_dynamic=true FLAGS_use_cinn=true' "
                    "to speed up C++ inference with paddle backend"
                 << std::endl;
  }
  if (config_fl->cinn_enabled()) {
    logg::info() << "model.forward_lower will be compiled with cinn."
                 << std::endl;
  } else {
    logg::info() << "NOTE: You can try: \n'export FLAGS_prim_all=true"
                    " FLAGS_enable_pir_in_executor=1"
                    " FLAGS_prim_enable_dynamic=true FLAGS_use_cinn=true' "
                    "to speed up C++ inference with paddle backend"
                 << std::endl;
  }

  // NOTE: Both set to 1 now.
  // get_env_nthreads(num_intra_nthreads,
  //                  num_inter_nthreads);  // need to be fixed as
  //                                        // DP_INTRA_OP_PARALLELISM_THREADS
  // num_intra_nthreads = 1;
  num_inter_nthreads = 1;
  if (num_inter_nthreads) {
    config->SetCpuMathLibraryNumThreads(num_inter_nthreads);
    config_fl->SetCpuMathLibraryNumThreads(num_inter_nthreads);
  }

  predictor = paddle_infer::CreatePredictor(*config);
  logg::info() << "Setup model.forward model" << std::endl;
  predictor_fl = paddle_infer::CreatePredictor(*config_fl);
  logg::info() << "Setup model.forward_lower" << std::endl;
  auto print_handle_names = [](const std::vector<std::string>& name_vec) {
    int n = name_vec.size();
    std::string ret;
    for (int i = 0; i < n; ++i) {
      ret += "[" + std::to_string(i) + "]" + name_vec[i] + " \n"[i == n - 1];
    }
    logg::debug() << ret;
  };
  logg::debug() << "Input names of model.forward below:" << std::endl;
  print_handle_names(predictor->GetInputNames());
  logg::debug() << "Output names of model.forward below:" << std::endl;
  print_handle_names(predictor->GetOutputNames());
  std::cout << std::endl;
  logg::debug() << "Input names of model.forward_lower below:" << std::endl;
  print_handle_names(predictor_fl->GetInputNames());
  logg::debug() << "Output names of model.forward_lower below:" << std::endl;
  print_handle_names(predictor_fl->GetOutputNames());

  // initialize hyper params from model buffers
  ntypes_spin = 0;
  DeepPotPD::get_buffer<int>("buffer_has_message_passing", do_message_passing);
  logg::debug() << "buffer_has_message_passing = " << this->do_message_passing
                << std::endl;
  DeepPotPD::get_buffer<double>("buffer_rcut", rcut);
  logg::debug() << "buffer_rcut = " << this->rcut << std::endl;
  DeepPotPD::get_buffer<int>("buffer_ntypes", ntypes);
  logg::debug() << "buffer_ntypes = " << this->ntypes << std::endl;
  DeepPotPD::get_buffer<int>("buffer_dfparam", dfparam);
  logg::debug() << "buffer_dfparam = " << this->dfparam << std::endl;
  DeepPotPD::get_buffer<int>("buffer_daparam", daparam);
  logg::debug() << "buffer_daparam = " << this->daparam << std::endl;
  DeepPotPD::get_buffer<int>("buffer_aparam_nall", aparam_nall);
  logg::debug() << "buffer_aparam_nall = " << this->aparam_nall << std::endl;
  inited = true;
}
DeepPotPD::~DeepPotPD() {}

template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute(ENERGYVTYPE& ener,
                        std::vector<VALUETYPE>& force,
                        std::vector<VALUETYPE>& virial,
                        std::vector<VALUETYPE>& atom_energy,
                        std::vector<VALUETYPE>& atom_virial,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const int nghost,
                        const InputNlist& lmp_list,
                        const int& ago,
                        const std::vector<VALUETYPE>& fparam,
                        const std::vector<VALUETYPE>& aparam,
                        const bool atomic) {
  int natoms = atype.size();
  // select real atoms
  std::vector<VALUETYPE> dcoord, dforce, aparam_, datom_energy, datom_virial;
  std::vector<int> datype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = natoms;
  select_real_atoms_coord(dcoord, datype, aparam_, nghost_real, fwd_map,
                          bkw_map, nall_real, nloc_real, coord, atype, aparam,
                          nghost, ntypes, 1, daparam, nall, aparam_nall);
  int nloc = nall_real - nghost_real;
  int nframes = 1;
  std::vector<VALUETYPE> coord_wrapped = dcoord;
  auto coord_wrapped_Tensor = predictor_fl->GetInputHandle("coord");
  coord_wrapped_Tensor->Reshape({1, nall_real, 3});
  coord_wrapped_Tensor->CopyFromCpu(coord_wrapped.data());
  auto atype_Tensor = predictor_fl->GetInputHandle("atype");
  atype_Tensor->Reshape({1, nall_real});
  atype_Tensor->CopyFromCpu(datype.data());
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
    nlist_data.padding();
    if (do_message_passing) {
      auto sendproc_tensor = predictor_fl->GetInputHandle("send_proc");
      auto recvproc_tensor = predictor_fl->GetInputHandle("recv_proc");
      auto recvnum_tensor = predictor_fl->GetInputHandle("recv_num");
      auto sendnum_tensor = predictor_fl->GetInputHandle("send_num");
      auto communicator_tensor = predictor_fl->GetInputHandle("communicator");
      auto sendlist_tensor = predictor_fl->GetInputHandle("send_list");

      int nswap = lmp_list.nswap;
      sendproc_tensor->Reshape({nswap});
      sendproc_tensor->CopyFromCpu(lmp_list.sendproc);

      recvproc_tensor->Reshape({nswap});
      recvproc_tensor->CopyFromCpu(lmp_list.recvproc);

      recvnum_tensor->Reshape({nswap});
      recvnum_tensor->CopyFromCpu(lmp_list.recvnum);

      sendnum_tensor->Reshape({nswap});
      if (sizeof(lmp_list.sendnum[0]) != sizeof(int32_t)) {
        std::vector<int32_t> temp_data(nswap);
        for (int i = 0; i < nswap; i++) {
          temp_data[i] = static_cast<int32_t>(lmp_list.sendnum[i]);
        }
        sendnum_tensor->CopyFromCpu(temp_data.data());
      } else {
        sendnum_tensor->CopyFromCpu(lmp_list.sendnum);
      }
      communicator_tensor->Reshape({1});
      if (lmp_list.world) {
        communicator_tensor->CopyFromCpu(static_cast<int*>(lmp_list.world));
      }

      assert(sizeof(std::intptr_t) == 8);
      int total_send =
          std::accumulate(lmp_list.sendnum, lmp_list.sendnum + nswap, 0);
      sendlist_tensor->Reshape({total_send});

      /**
      ** NOTE: paddle do not support construct a Tensor with from_blob(T**, ...)
      ** from a double pointer, so we convert int* pointer to indptr_t for each
      ** entry and wrap it into int64 Tensor as a workaround.
      */
      std::vector<std::intptr_t> pointer_addresses;
      pointer_addresses.reserve(nswap);
      for (int iswap = 0; iswap < nswap; ++iswap) {
        std::intptr_t addr =
            reinterpret_cast<std::intptr_t>(lmp_list.sendlist[iswap]);
        pointer_addresses.push_back(addr);
      }
      sendlist_tensor->CopyFromCpu(pointer_addresses.data());
    }
    if (lmp_list.mapping) {
      std::vector<std::int64_t> mapping(nall_real);
      for (size_t ii = 0; ii < nall_real; ii++) {
        mapping[ii] = lmp_list.mapping[fwd_map[ii]];
      }
      this->mapping_tensor = predictor_fl->GetInputHandle("mapping");
      this->mapping_tensor->Reshape({1, nall_real});
      this->mapping_tensor->CopyFromCpu(mapping.data());
    }
  }
  this->firstneigh_tensor = predictor_fl->GetInputHandle("nlist");
  fillNlistTensor(nlist_data.jlist, this->firstneigh_tensor);
  bool do_atom_virial_tensor = atomic;
  if (!fparam.empty()) {
    std::unique_ptr<paddle_infer::Tensor> fparam_tensor;
    fparam_tensor = predictor_fl->GetInputHandle("fparam");
    fparam_tensor->Reshape({1, static_cast<int>(fparam.size())});
    fparam_tensor->CopyFromCpu(fparam.data());
  }
  if (!aparam_.empty()) {
    std::unique_ptr<paddle_infer::Tensor> aparam_tensor;
    aparam_tensor = predictor_fl->GetInputHandle("aparam");
    aparam_tensor->Reshape(
        {1, lmp_list.inum, static_cast<int>(aparam_.size()) / lmp_list.inum});
    aparam_tensor->CopyFromCpu((aparam_.data()));
  }

  if (!predictor_fl->Run()) {
    throw deepmd::deepmd_exception("Paddle inference run failed");
  }
  auto output_names = predictor_fl->GetOutputNames();

  auto energy_ = predictor_fl->GetOutputHandle(output_names.at(1));
  auto force_ = predictor_fl->GetOutputHandle(output_names.at(2));
  auto virial_ = predictor_fl->GetOutputHandle(output_names.at(4));
  size_t output_energy_size = numel(*energy_);
  size_t output_force_size = numel(*force_);
  size_t output_virial_size = numel(*virial_);
  // output energy
  ener.resize(output_energy_size);
  energy_->CopyToCpu(ener.data());

  // output force
  dforce.resize(output_force_size);
  force_->CopyToCpu(dforce.data());

  // output virial
  virial.resize(output_virial_size);
  virial_->CopyToCpu(virial.data());

  // bkw map
  force.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  select_map<VALUETYPE>(force, dforce, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  if (atomic) {
    auto atom_virial_ = predictor_fl->GetOutputHandle(output_names.at(3));
    auto atom_energy_ = predictor_fl->GetOutputHandle(output_names.at(0));
    datom_energy.resize(nall_real,
                        0.0);  // resize to nall to be consistenet with TF.
    atom_energy_->CopyToCpu(datom_energy.data());
    datom_virial.resize(numel(*atom_virial_));
    atom_virial_->CopyToCpu(datom_virial.data());
    atom_energy.resize(static_cast<size_t>(nframes) * fwd_map.size());
    atom_virial.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
    select_map<VALUETYPE>(atom_energy, datom_energy, bkw_map, 1, nframes,
                          fwd_map.size(), nall_real);
    select_map<VALUETYPE>(atom_virial, datom_virial, bkw_map, 9, nframes,
                          fwd_map.size(), nall_real);
  }
}
template void DeepPotPD::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);
template void DeepPotPD::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const int nghost,
    const InputNlist& lmp_list,
    const int& ago,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);
// ENERGYVTYPE: std::vector<ENERGYTYPE> or ENERGYTYPE
template <typename VALUETYPE, typename ENERGYVTYPE>
void DeepPotPD::compute(ENERGYVTYPE& ener,
                        std::vector<VALUETYPE>& force,
                        std::vector<VALUETYPE>& virial,
                        std::vector<VALUETYPE>& atom_energy,
                        std::vector<VALUETYPE>& atom_virial,
                        const std::vector<VALUETYPE>& coord,
                        const std::vector<int>& atype,
                        const std::vector<VALUETYPE>& box,
                        const std::vector<VALUETYPE>& fparam,
                        const std::vector<VALUETYPE>& aparam,
                        const bool atomic) {
  // select real atoms
  std::vector<VALUETYPE> coord_wrapped = coord;
  int natoms = atype.size();
  int nframes = 1;
  auto coord_wrapped_Tensor = predictor->GetInputHandle("coord");
  coord_wrapped_Tensor->Reshape({1, natoms, 3});
  coord_wrapped_Tensor->CopyFromCpu(coord_wrapped.data());

  auto atype_Tensor = predictor->GetInputHandle("atype");
  atype_Tensor->Reshape({1, natoms});
  std::vector<std::int64_t> atype_64(atype.begin(), atype.end());
  atype_Tensor->CopyFromCpu(atype_64.data());

  std::unique_ptr<paddle_infer::Tensor> box_Tensor;
  if (!box.empty()) {
    box_Tensor = predictor->GetInputHandle("box");
    box_Tensor->Reshape({1, 9});
    box_Tensor->CopyFromCpu((box.data()));
  }
  if (!fparam.empty()) {
    std::unique_ptr<paddle_infer::Tensor> fparam_tensor;
    fparam_tensor = predictor->GetInputHandle("fparam");
    fparam_tensor->Reshape({1, static_cast<int>(fparam.size())});
    fparam_tensor->CopyFromCpu((fparam.data()));
  }
  if (!aparam.empty()) {
    std::unique_ptr<paddle_infer::Tensor> aparam_tensor;
    aparam_tensor = predictor->GetInputHandle("aparam");
    aparam_tensor->Reshape(
        {1, natoms, static_cast<int>(aparam.size()) / natoms});
    aparam_tensor->CopyFromCpu((aparam.data()));
  }

  bool do_atom_virial_tensor = atomic;
  if (!predictor->Run()) {
    throw deepmd::deepmd_exception("Paddle inference run failed");
  }

  auto output_names = predictor->GetOutputNames();
  auto energy_ = predictor->GetOutputHandle(output_names.at(2));
  auto force_ = predictor->GetOutputHandle(output_names.at(3));
  auto virial_ = predictor->GetOutputHandle(output_names.at(5));

  size_t enery_numel = numel(*energy_);
  assert(enery_numel > 0);
  ener.resize(enery_numel);
  energy_->CopyToCpu(ener.data());

  size_t force_numel = numel(*force_);
  assert(force_numel > 0);
  force.resize(force_numel);
  force_->CopyToCpu(force.data());

  size_t virial_numel = numel(*virial_);
  assert(virial_numel > 0);
  virial.resize(virial_numel);
  virial_->CopyToCpu(virial.data());

  if (atomic) {
    auto atom_energy_ = predictor->GetOutputHandle(output_names.at(0));
    auto atom_virial_ = predictor->GetOutputHandle(output_names.at(1));
    size_t atom_energy_numel = numel(*atom_energy_);
    size_t atom_virial_numel = numel(*atom_virial_);
    assert(atom_energy_numel > 0);
    assert(atom_virial_numel > 0);
    atom_energy.resize(atom_energy_numel);
    atom_energy_->CopyToCpu(atom_energy.data());
    atom_virial.resize(atom_virial_numel);
    atom_virial_->CopyToCpu(atom_virial.data());
  }
}

template void DeepPotPD::compute<double, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<double>& force,
    std::vector<double>& virial,
    std::vector<double>& atom_energy,
    std::vector<double>& atom_virial,
    const std::vector<double>& coord,
    const std::vector<int>& atype,
    const std::vector<double>& box,
    const std::vector<double>& fparam,
    const std::vector<double>& aparam,
    const bool atomic);

template void DeepPotPD::compute<float, std::vector<ENERGYTYPE>>(
    std::vector<ENERGYTYPE>& ener,
    std::vector<float>& force,
    std::vector<float>& virial,
    std::vector<float>& atom_energy,
    std::vector<float>& atom_virial,
    const std::vector<float>& coord,
    const std::vector<int>& atype,
    const std::vector<float>& box,
    const std::vector<float>& fparam,
    const std::vector<float>& aparam,
    const bool atomic);

/* type_map is regarded as a special string buffer
that need to be postprocessed */
void DeepPotPD::get_type_map(std::string& type_map) {
  auto type_map_tensor = predictor->GetOutputHandle("buffer_type_map");
  size_t type_map_size = numel(*type_map_tensor);

  std::vector<int> type_map_arr(type_map_size, 0);
  type_map_tensor->CopyToCpu(type_map_arr.data());
  for (auto char_c : type_map_arr) {
    type_map += std::string(1, char_c);
  }
}

/* general function except for string buffer */
template <typename BUFFERTYPE>
void DeepPotPD::get_buffer(const std::string& buffer_name,
                           std::vector<BUFFERTYPE>& buffer_array) {
  auto buffer_tensor = predictor->GetOutputHandle(buffer_name);
  size_t buffer_size = numel(*buffer_tensor);
  buffer_array.resize(buffer_size);
  buffer_tensor->CopyToCpu(buffer_array.data());
}

template <typename BUFFERTYPE>
void DeepPotPD::get_buffer(const std::string& buffer_name,
                           BUFFERTYPE& buffer_scalar) {
  std::vector<BUFFERTYPE> buffer_array(1);
  DeepPotPD::get_buffer<BUFFERTYPE>(buffer_name, buffer_array);
  buffer_scalar = buffer_array[0];
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
  throw deepmd::deepmd_exception(
      "computew_mixed_type is not implemented in paddle backend yet");
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
  throw deepmd::deepmd_exception(
      "computew_mixed_type is not implemented in paddle backend yet");
}
#endif
