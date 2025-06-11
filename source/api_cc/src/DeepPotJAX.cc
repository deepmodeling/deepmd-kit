// SPDX-License-Identifier: LGPL-3.0-or-later
#if defined(BUILD_TENSORFLOW) || defined(BUILD_JAX)

#include "DeepPotJAX.h"

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "common.h"
#include "device.h"
#include "errors.h"

#define PADDING_FACTOR 1.05

inline void check_status(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    throw deepmd::deepmd_exception("TensorFlow C API Error: " +
                                   std::string(TF_Message(status)));
  }
}

inline void find_function(TF_Function*& found_func,
                          const std::vector<TF_Function*>& funcs,
                          const std::string func_name) {
  for (size_t i = 0; i < funcs.size(); i++) {
    TF_Function* func = funcs[i];
    const char* name = TF_FunctionName(func);
    std::string name_(name);
    // remove trailing integer e.g. _123
    std::string::size_type pos = name_.find_last_not_of("0123456789_");
    if (pos != std::string::npos) {
      name_ = name_.substr(0, pos + 1);
    }
    if (name_ == "__inference_" + func_name) {
      found_func = func;
      return;
    }
  }
  found_func = NULL;
}

inline TF_DataType get_data_tensor_type(const std::vector<double>& data) {
  return TF_DOUBLE;
}

inline TF_DataType get_data_tensor_type(const std::vector<float>& data) {
  return TF_FLOAT;
}

inline TF_DataType get_data_tensor_type(const std::vector<int32_t>& data) {
  return TF_INT32;
}

inline TF_DataType get_data_tensor_type(const std::vector<int64_t>& data) {
  return TF_INT64;
}

struct tf_function_not_found : public deepmd::deepmd_exception {
 public:
  tf_function_not_found() : deepmd_exception() {};
  tf_function_not_found(const std::string& msg) : deepmd_exception(msg) {};
};

inline TFE_Op* get_func_op(TFE_Context* ctx,
                           const std::string func_name,
                           const std::vector<TF_Function*>& funcs,
                           const std::string device,
                           TF_Status* status) {
  TF_Function* func = NULL;
  find_function(func, funcs, func_name);
  if (func == NULL) {
    throw tf_function_not_found("Function " + func_name + " not found");
  }
  const char* real_func_name = TF_FunctionName(func);
  // execute the function
  TFE_Op* op = TFE_NewOp(ctx, real_func_name, status);
  check_status(status);
  TFE_OpSetDevice(op, device.c_str(), status);
  check_status(status);
  return op;
}

template <typename T>
inline T get_scalar(TFE_Context* ctx,
                    const std::string func_name,
                    const std::vector<TF_Function*>& funcs,
                    const std::string device,
                    TF_Status* status) {
  TFE_Op* op = get_func_op(ctx, func_name, funcs, device, status);
  check_status(status);
  TFE_TensorHandle* retvals[1];
  int nretvals = 1;
  TFE_Execute(op, retvals, &nretvals, status);
  check_status(status);
  TFE_TensorHandle* retval = retvals[0];
  TF_Tensor* tensor = TFE_TensorHandleResolve(retval, status);
  check_status(status);
  T* data = (T*)TF_TensorData(tensor);
  // copy data
  T result = *data;
  // deallocate
  TFE_DeleteOp(op);
  TF_DeleteTensor(tensor);
  TFE_DeleteTensorHandle(retval);
  return result;
}

template <typename T>
inline std::vector<T> get_vector(TFE_Context* ctx,
                                 const std::string func_name,
                                 const std::vector<TF_Function*>& funcs,
                                 const std::string device,
                                 TF_Status* status) {
  TFE_Op* op = get_func_op(ctx, func_name, funcs, device, status);
  check_status(status);
  TFE_TensorHandle* retvals[1];
  int nretvals = 1;
  TFE_Execute(op, retvals, &nretvals, status);
  check_status(status);
  TFE_TensorHandle* retval = retvals[0];
  // copy data
  std::vector<T> result;
  tensor_to_vector(result, retval, status);
  // deallocate
  TFE_DeleteTensorHandle(retval);
  TFE_DeleteOp(op);
  return result;
}

inline std::vector<std::string> get_vector_string(
    TFE_Context* ctx,
    const std::string func_name,
    const std::vector<TF_Function*>& funcs,
    const std::string device,
    TF_Status* status) {
  TFE_Op* op = get_func_op(ctx, func_name, funcs, device, status);
  check_status(status);
  TFE_TensorHandle* retvals[1];
  int nretvals = 1;
  TFE_Execute(op, retvals, &nretvals, status);
  check_status(status);
  TFE_TensorHandle* retval = retvals[0];
  TF_Tensor* tensor = TFE_TensorHandleResolve(retval, status);
  check_status(status);
  // calculate the number of bytes in each string
  const void* data = TF_TensorData(tensor);
  int64_t bytes_each_string =
      TF_TensorByteSize(tensor) / TF_TensorElementCount(tensor);
  // copy data
  std::vector<std::string> result;
  for (int ii = 0; ii < TF_TensorElementCount(tensor); ++ii) {
    const TF_TString* datastr =
        static_cast<const TF_TString*>(static_cast<const void*>(
            static_cast<const char*>(data) + ii * bytes_each_string));
    const char* dst = TF_TString_GetDataPointer(datastr);
    size_t dst_len = TF_TString_GetSize(datastr);
    result.push_back(std::string(dst, dst_len));
  }

  // deallocate
  TFE_DeleteOp(op);
  TF_DeleteTensor(tensor);
  TFE_DeleteTensorHandle(retval);
  return result;
}

template <typename T>
inline TF_Tensor* create_tensor(const std::vector<T>& data,
                                const std::vector<int64_t>& shape) {
  TF_Tensor* tensor =
      TF_AllocateTensor(get_data_tensor_type(data), shape.data(), shape.size(),
                        data.size() * sizeof(T));
  memcpy(TF_TensorData(tensor), data.data(), TF_TensorByteSize(tensor));
  return tensor;
}

template <typename T>
inline TFE_TensorHandle* add_input(TFE_Op* op,
                                   const std::vector<T>& data,
                                   const std::vector<int64_t>& data_shape,
                                   TF_Tensor*& data_tensor,
                                   TF_Status* status) {
  data_tensor = create_tensor(data, data_shape);
  TFE_TensorHandle* handle = TFE_NewTensorHandle(data_tensor, status);
  check_status(status);

  TFE_OpAddInput(op, handle, status);
  check_status(status);
  return handle;
}

template <typename T>
inline void tensor_to_vector(std::vector<T>& result,
                             TFE_TensorHandle* retval,
                             TF_Status* status) {
  TF_Tensor* tensor = TFE_TensorHandleResolve(retval, status);
  check_status(status);
  T* data = (T*)TF_TensorData(tensor);
  // copy data
  result.resize(TF_TensorElementCount(tensor));
  for (int i = 0; i < TF_TensorElementCount(tensor); i++) {
    result[i] = data[i];
  }
  // Delete the tensor to free memory
  TF_DeleteTensor(tensor);
}

deepmd::DeepPotJAX::DeepPotJAX() : inited(false) {}
deepmd::DeepPotJAX::DeepPotJAX(const std::string& model,
                               const int& gpu_rank,
                               const std::string& file_content)
    : inited(false) {
  init(model, gpu_rank, file_content);
}
void deepmd::DeepPotJAX::init(const std::string& model,
                              const int& gpu_rank,
                              const std::string& file_content) {
  if (inited) {
    std::cerr << "WARNING: deepmd-kit should not be initialized twice, do "
                 "nothing at the second call of initializer"
              << std::endl;
    return;
  }

  const char* saved_model_dir = model.c_str();
  graph = TF_NewGraph();
  status = TF_NewStatus();

  sessionopts = TF_NewSessionOptions();
  int num_intra_nthreads, num_inter_nthreads;
  get_env_nthreads(num_intra_nthreads, num_inter_nthreads);
  // https://github.com/Neargye/hello_tf_c_api/blob/51516101cf59408a6bb456f7e5f3c6628e327b3a/src/tf_utils.cpp#L400-L401
  // https://github.com/Neargye/hello_tf_c_api/blob/51516101cf59408a6bb456f7e5f3c6628e327b3a/src/tf_utils.cpp#L364-L379
  // The following is an equivalent of setting this in Python:
  // config = tf.ConfigProto( allow_soft_placement = True )
  // config.gpu_options.allow_growth = True
  // config.gpu_options.per_process_gpu_memory_fraction = percentage
  // Create a byte-array for the serialized ProtoConfig, set the mandatory bytes
  // (first three and last four)
  std::array<std::uint8_t, 19> config = {
      {0x10, static_cast<std::uint8_t>(num_intra_nthreads), 0x28,
       static_cast<std::uint8_t>(num_inter_nthreads), 0x32, 0xb, 0x9, 0xFF,
       0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x20, 0x1, 0x38, 0x1}};

  // Convert the desired percentage into a byte-array.
  double gpu_memory_fraction = 0.9;
  auto bytes = reinterpret_cast<std::uint8_t*>(&gpu_memory_fraction);

  // Put it to the config byte-array, from 7 to 14:
  for (std::size_t i = 0; i < sizeof(gpu_memory_fraction); ++i) {
    config[i + 7] = bytes[i];
  }

  TF_SetConfig(sessionopts, config.data(), config.size(), status);
  check_status(status);

  TF_Buffer* runopts = NULL;

  const char* tags = "serve";
  int ntags = 1;

  session = TF_LoadSessionFromSavedModel(sessionopts, runopts, saved_model_dir,
                                         &tags, ntags, graph, NULL, status);
  check_status(status);

  int nfuncs = TF_GraphNumFunctions(graph);
  // allocate memory for the TF_Function* array
  func_vector.resize(nfuncs);
  TF_Function** funcs = func_vector.data();
  TF_GraphGetFunctions(graph, funcs, nfuncs, status);
  check_status(status);

  ctx_opts = TFE_NewContextOptions();
  TFE_ContextOptionsSetConfig(ctx_opts, config.data(), config.size(), status);
  check_status(status);
  ctx = TFE_NewContext(ctx_opts, status);
  check_status(status);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  int gpu_num;
  DPGetDeviceCount(gpu_num);  // check current device environment
  if (gpu_num > 0 && gpu_rank >= 0) {
    DPErrcheck(DPSetDevice(gpu_rank % gpu_num));
    device = "/gpu:" + std::to_string(gpu_rank % gpu_num);
  } else {
    device = "/cpu:0";
  }
#else
  device = "/cpu:0";
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  // add all functions, otherwise the function will not be found
  // even for tf.cond
  for (size_t i = 0; i < func_vector.size(); i++) {
    TF_Function* func = func_vector[i];
    TFE_ContextAddFunction(ctx, func, status);
    check_status(status);
  }

  rcut = get_scalar<double>(ctx, "get_rcut", func_vector, device, status);
  dfparam =
      get_scalar<int64_t>(ctx, "get_dim_fparam", func_vector, device, status);
  daparam =
      get_scalar<int64_t>(ctx, "get_dim_aparam", func_vector, device, status);
  std::vector<std::string> type_map_ =
      get_vector_string(ctx, "get_type_map", func_vector, device, status);
  // deepmd-kit stores type_map as a concatenated string, split by ' '
  type_map = type_map_[0];
  for (size_t i = 1; i < type_map_.size(); i++) {
    type_map += " " + type_map_[i];
  }
  ntypes = type_map_.size();
  sel = get_vector<int64_t>(ctx, "get_sel", func_vector, device, status);
  nnei = std::accumulate(sel.begin(), sel.end(), decltype(sel)::value_type(0));
  try {
    do_message_passing = get_scalar<bool>(ctx, "do_message_passing",
                                          func_vector, device, status);
  } catch (tf_function_not_found& e) {
    // compatibile with models generated by v3.0.0rc0
    do_message_passing = false;
  }
  inited = true;
}

deepmd::DeepPotJAX::~DeepPotJAX() {
  if (inited) {
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(sessionopts);
    TF_DeleteStatus(status);
    TFE_DeleteContext(ctx);
    TFE_DeleteContextOptions(ctx_opts);
    for (size_t i = 0; i < func_vector.size(); i++) {
      TF_DeleteFunction(func_vector[i]);
    }
  }
}

template <typename VALUETYPE>
void deepmd::DeepPotJAX::compute(std::vector<ENERGYTYPE>& ener,
                                 std::vector<VALUETYPE>& force_,
                                 std::vector<VALUETYPE>& virial,
                                 std::vector<VALUETYPE>& atom_energy_,
                                 std::vector<VALUETYPE>& atom_virial_,
                                 const std::vector<VALUETYPE>& dcoord,
                                 const std::vector<int>& datype,
                                 const std::vector<VALUETYPE>& box,
                                 const std::vector<VALUETYPE>& fparam,
                                 const std::vector<VALUETYPE>& aparam_,
                                 const bool atomic) {
  std::vector<VALUETYPE> coord, force, aparam, atom_energy, atom_virial;
  std::vector<double> ener_double, force_double, virial_double,
      atom_energy_double, atom_virial_double;
  std::vector<int> atype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = datype.size();
  // nlist passed to the model
  int nframes = nall > 0 ? (dcoord.size() / 3 / nall) : 1;
  int nghost = 0;

  select_real_atoms_coord(coord, atype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord, datype, aparam_, nghost,
                          ntypes, nframes, daparam, nall, false);

  if (nloc_real == 0) {
    // no real atoms, fill 0 for all outputs
    // this can prevent a Xla error
    ener.resize(nframes, 0.0);
    force_.resize(static_cast<size_t>(nframes) * nall * 3, 0.0);
    virial.resize(static_cast<size_t>(nframes) * 9, 0.0);
    atom_energy_.resize(static_cast<size_t>(nframes) * nall, 0.0);
    atom_virial_.resize(static_cast<size_t>(nframes) * nall * 9, 0.0);
    return;
  }

  // cast coord, fparam, and aparam to double - I think it's useless to have a
  // float model interface
  std::vector<double> coord_double(coord.begin(), coord.end());
  std::vector<double> box_double(box.begin(), box.end());
  std::vector<double> fparam_double(fparam.begin(), fparam.end());
  std::vector<double> aparam_double(aparam.begin(), aparam.end());

  TFE_Op* op;
  if (atomic) {
    op = get_func_op(ctx, "call_with_atomic_virial", func_vector, device,
                     status);
  } else {
    op = get_func_op(ctx, "call_without_atomic_virial", func_vector, device,
                     status);
  }
  std::vector<TFE_TensorHandle*> input_list(5);
  std::vector<TF_Tensor*> data_tensor(5);
  // coord
  std::vector<int64_t> coord_shape = {nframes, nloc_real, 3};
  input_list[0] =
      add_input(op, coord_double, coord_shape, data_tensor[0], status);
  // atype
  std::vector<int64_t> atype_shape = {nframes, nloc_real};
  input_list[1] = add_input(op, atype, atype_shape, data_tensor[1], status);
  // box
  int box_size = box_double.size() > 0 ? 3 : 0;
  std::vector<int64_t> box_shape = {nframes, box_size, box_size};
  input_list[2] = add_input(op, box_double, box_shape, data_tensor[2], status);
  // fparam
  std::vector<int64_t> fparam_shape = {nframes, dfparam};
  input_list[3] =
      add_input(op, fparam_double, fparam_shape, data_tensor[3], status);
  // aparam
  std::vector<int64_t> aparam_shape = {nframes, nloc_real, daparam};
  input_list[4] =
      add_input(op, aparam_double, aparam_shape, data_tensor[4], status);
  // execute the function
  int nretvals = 6;
  TFE_TensorHandle* retvals[nretvals];

  TFE_Execute(op, retvals, &nretvals, status);
  check_status(status);

  // copy data
  // for atom virial, the order is:
  // Identity_15 energy -1, -1, 1
  // Identity_16 energy_derv_c -1, -1, 1, 9 (may pop)
  // Identity_17 energy_derv_c_redu -1, 1, 9
  // Identity_18 energy_derv_r -1, -1, 1, 3
  // Identity_19 energy_redu -1, 1
  // Identity_20 mask (int32) -1, -1
  //
  // for no atom virial, the order is:
  // Identity_15 energy -1, -1, 1
  // Identity_16 energy_derv_c -1, 1, 9
  // Identity_17 energy_derv_r -1, -1, 1, 3
  // Identity_18 energy_redu -1, 1
  // Identity_19 mask (int32) -1, -1
  //
  // it seems the order is the alphabet order?
  // not sure whether it is safe to assume the order
  if (atomic) {
    tensor_to_vector(ener_double, retvals[4], status);
    tensor_to_vector(force_double, retvals[3], status);
    tensor_to_vector(virial_double, retvals[2], status);
    tensor_to_vector(atom_energy_double, retvals[0], status);
    tensor_to_vector(atom_virial_double, retvals[1], status);
  } else {
    tensor_to_vector(ener_double, retvals[3], status);
    tensor_to_vector(force_double, retvals[2], status);
    tensor_to_vector(virial_double, retvals[1], status);
    tensor_to_vector(atom_energy_double, retvals[0], status);
  }

  // cast back to VALUETYPE
  ener = std::vector<ENERGYTYPE>(ener_double.begin(), ener_double.end());
  force = std::vector<VALUETYPE>(force_double.begin(), force_double.end());
  virial = std::vector<VALUETYPE>(virial_double.begin(), virial_double.end());
  atom_energy = std::vector<VALUETYPE>(atom_energy_double.begin(),
                                       atom_energy_double.end());
  atom_virial = std::vector<VALUETYPE>(atom_virial_double.begin(),
                                       atom_virial_double.end());
  force.resize(static_cast<size_t>(nframes) * nall_real * 3);
  atom_virial.resize(static_cast<size_t>(nframes) * nall_real * 9);

  // nall atom_energy is required in the C++ API;
  // we always forget it!
  atom_energy.resize(static_cast<size_t>(nframes) * nall_real, 0.0);

  force_.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  atom_energy_.resize(static_cast<size_t>(nframes) * fwd_map.size());
  atom_virial_.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
  select_map<VALUETYPE>(force_, force, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(atom_energy_, atom_energy, bkw_map, 1, nframes,
                        fwd_map.size(), nall_real);
  select_map<VALUETYPE>(atom_virial_, atom_virial, bkw_map, 9, nframes,
                        fwd_map.size(), nall_real);

  // cleanup input_list, etc
  for (size_t i = 0; i < 5; i++) {
    TFE_DeleteTensorHandle(input_list[i]);
    TF_DeleteTensor(data_tensor[i]);
  }
  for (size_t i = 0; i < nretvals; i++) {
    TFE_DeleteTensorHandle(retvals[i]);
  }
  TFE_DeleteOp(op);
}

template <typename VALUETYPE>
void deepmd::DeepPotJAX::compute(std::vector<ENERGYTYPE>& ener,
                                 std::vector<VALUETYPE>& force_,
                                 std::vector<VALUETYPE>& virial,
                                 std::vector<VALUETYPE>& atom_energy_,
                                 std::vector<VALUETYPE>& atom_virial_,
                                 const std::vector<VALUETYPE>& dcoord,
                                 const std::vector<int>& datype,
                                 const std::vector<VALUETYPE>& box,
                                 const int nghost,
                                 const InputNlist& lmp_list,
                                 const int& ago,
                                 const std::vector<VALUETYPE>& fparam,
                                 const std::vector<VALUETYPE>& aparam_,
                                 const bool atomic) {
  std::vector<VALUETYPE> coord, force, aparam, atom_energy, atom_virial;
  std::vector<double> ener_double, force_double, virial_double,
      atom_energy_double, atom_virial_double;
  std::vector<int> atype, fwd_map, bkw_map;
  int nghost_real, nall_real, nloc_real;
  int nall = datype.size();
  // nlist passed to the model
  int nframes = 1;

  select_real_atoms_coord(coord, atype, aparam, nghost_real, fwd_map, bkw_map,
                          nall_real, nloc_real, dcoord, datype, aparam_, nghost,
                          ntypes, nframes, daparam, nall, false);

  if (nloc_real == 0) {
    // no real atoms, fill 0 for all outputs
    // this can prevent a Xla error
    ener.resize(nframes, 0.0);
    force_.resize(static_cast<size_t>(nframes) * nall * 3, 0.0);
    virial.resize(static_cast<size_t>(nframes) * 9, 0.0);
    atom_energy_.resize(static_cast<size_t>(nframes) * nall, 0.0);
    atom_virial_.resize(static_cast<size_t>(nframes) * nall * 9, 0.0);
    return;
  }

  // cast coord, fparam, and aparam to double - I think it's useless to have a
  // float model interface
  std::vector<double> coord_double(coord.begin(), coord.end());
  std::vector<double> fparam_double(fparam.begin(), fparam.end());
  std::vector<double> aparam_double(aparam.begin(), aparam.end());

  if (padding_for_nloc != nloc_real) {
    padding_to_nall = nall_real * PADDING_FACTOR;
    padding_for_nloc = nloc_real;
  }
  while (padding_to_nall < nall_real) {
    padding_to_nall *= PADDING_FACTOR;
  }
  // do padding
  coord_double.resize(nframes * padding_to_nall * 3, 0.0);
  atype.resize(nframes * padding_to_nall, -1);

  TFE_Op* op;
  if (atomic) {
    op = get_func_op(ctx, "call_lower_with_atomic_virial", func_vector, device,
                     status);
  } else {
    op = get_func_op(ctx, "call_lower_without_atomic_virial", func_vector,
                     device, status);
  }
  std::vector<TFE_TensorHandle*> input_list(6);
  std::vector<TF_Tensor*> data_tensor(6);
  // coord
  std::vector<int64_t> coord_shape = {nframes, padding_to_nall, 3};
  input_list[0] =
      add_input(op, coord_double, coord_shape, data_tensor[0], status);
  // atype
  std::vector<int64_t> atype_shape = {nframes, padding_to_nall};
  input_list[1] = add_input(op, atype, atype_shape, data_tensor[1], status);
  // nlist
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list, nall - nghost);
    nlist_data.shuffle_exclude_empty(fwd_map);
  }
  size_t max_size = 0;
  for (const auto& row : nlist_data.jlist) {
    max_size = std::max(max_size, row.size());
  }
  std::vector<int64_t> nlist_shape = {nframes, nloc_real,
                                      static_cast<int64_t>(max_size)};
  std::vector<int64_t> nlist(static_cast<size_t>(nframes) * nloc_real *
                             max_size);
  // pass nlist_data.jlist to nlist
  for (int ii = 0; ii < nloc_real; ii++) {
    for (int jj = 0; jj < max_size; jj++) {
      if (jj < nlist_data.jlist[ii].size()) {
        nlist[ii * max_size + jj] = nlist_data.jlist[ii][jj];
      } else {
        nlist[ii * max_size + jj] = -1;
      }
    }
  }
  input_list[2] = add_input(op, nlist, nlist_shape, data_tensor[2], status);
  // mapping; for now, set it to -1, assume it is not used
  std::vector<int64_t> mapping_shape = {nframes, padding_to_nall};
  std::vector<int64_t> mapping(nframes * padding_to_nall, -1);
  // pass mapping if it is given in the neighbor list
  if (lmp_list.mapping) {
    // assume nframes is 1
    for (size_t ii = 0; ii < nall_real; ii++) {
      mapping[ii] = lmp_list.mapping[fwd_map[ii]];
    }
  } else if (nloc_real == nall_real) {
    // no ghost atoms
    for (size_t ii = 0; ii < nall_real; ii++) {
      mapping[ii] = ii;
    }
  } else if (do_message_passing) {
    throw deepmd::deepmd_exception(
        "Mapping is required for a message passing model. If you are using "
        "LAMMPS, set `atom_modify map yes`");
  }
  input_list[3] = add_input(op, mapping, mapping_shape, data_tensor[3], status);
  // fparam
  std::vector<int64_t> fparam_shape = {nframes, dfparam};
  input_list[4] =
      add_input(op, fparam_double, fparam_shape, data_tensor[4], status);
  // aparam
  std::vector<int64_t> aparam_shape = {nframes, nloc_real, daparam};
  input_list[5] =
      add_input(op, aparam_double, aparam_shape, data_tensor[5], status);
  // execute the function
  int nretvals = 6;
  TFE_TensorHandle* retvals[nretvals];

  TFE_Execute(op, retvals, &nretvals, status);
  check_status(status);

  // copy data
  // the order is:
  // energy
  // energy_derv_c
  // energy_derv_c_redu
  // energy_derv_r
  // energy_redu
  // mask
  // it seems the order is the alphabet order?
  // not sure whether it is safe to assume the order
  tensor_to_vector(ener_double, retvals[4], status);
  tensor_to_vector(force_double, retvals[3], status);
  tensor_to_vector(virial_double, retvals[2], status);
  tensor_to_vector(atom_energy_double, retvals[0], status);
  tensor_to_vector(atom_virial_double, retvals[1], status);

  // cast back to VALUETYPE
  ener = std::vector<ENERGYTYPE>(ener_double.begin(), ener_double.end());
  force = std::vector<VALUETYPE>(force_double.begin(), force_double.end());
  virial = std::vector<VALUETYPE>(virial_double.begin(), virial_double.end());
  atom_energy = std::vector<VALUETYPE>(atom_energy_double.begin(),
                                       atom_energy_double.end());
  atom_virial = std::vector<VALUETYPE>(atom_virial_double.begin(),
                                       atom_virial_double.end());
  force.resize(static_cast<size_t>(nframes) * nall_real * 3);
  atom_virial.resize(static_cast<size_t>(nframes) * nall_real * 9);

  // nall atom_energy is required in the C++ API;
  // we always forget it!
  atom_energy.resize(static_cast<size_t>(nframes) * nall_real, 0.0);

  force_.resize(static_cast<size_t>(nframes) * fwd_map.size() * 3);
  atom_energy_.resize(static_cast<size_t>(nframes) * fwd_map.size());
  atom_virial_.resize(static_cast<size_t>(nframes) * fwd_map.size() * 9);
  select_map<VALUETYPE>(force_, force, bkw_map, 3, nframes, fwd_map.size(),
                        nall_real);
  select_map<VALUETYPE>(atom_energy_, atom_energy, bkw_map, 1, nframes,
                        fwd_map.size(), nall_real);
  select_map<VALUETYPE>(atom_virial_, atom_virial, bkw_map, 9, nframes,
                        fwd_map.size(), nall_real);

  // cleanup input_list, etc
  for (size_t i = 0; i < 6; i++) {
    TFE_DeleteTensorHandle(input_list[i]);
    TF_DeleteTensor(data_tensor[i]);
  }
  for (size_t i = 0; i < nretvals; i++) {
    TFE_DeleteTensorHandle(retvals[i]);
  }
  TFE_DeleteOp(op);
}

template void deepmd::DeepPotJAX::compute<double>(
    std::vector<deepmd::ENERGYTYPE>& dener,
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

template void deepmd::DeepPotJAX::compute<float>(
    std::vector<deepmd::ENERGYTYPE>& dener,
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

void deepmd::DeepPotJAX::get_type_map(std::string& type_map_) {
  type_map_ = type_map;
}

// forward to template method
void deepmd::DeepPotJAX::computew(std::vector<double>& ener,
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
void deepmd::DeepPotJAX::computew(std::vector<double>& ener,
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
void deepmd::DeepPotJAX::computew(std::vector<double>& ener,
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
void deepmd::DeepPotJAX::computew(std::vector<double>& ener,
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
void deepmd::DeepPotJAX::computew_mixed_type(std::vector<double>& ener,
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
  throw deepmd::deepmd_exception("not implemented");
}
void deepmd::DeepPotJAX::computew_mixed_type(std::vector<double>& ener,
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
  throw deepmd::deepmd_exception("not implemented");
}
#endif
