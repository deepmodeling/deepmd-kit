// SPDX-License-Identifier: LGPL-3.0-or-later
#ifdef BUILD_TENSORFLOW

#include "DeepPotJAX.h"

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>

#include <cstdio>
#include <iostream>
#include <numeric>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "common.h"
#include "device.h"
#include "errors.h"

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

inline TFE_Op* get_func_op(TFE_Context* ctx,
                           const std::string func_name,
                           const std::vector<TF_Function*>& funcs,
                           const std::string device,
                           TF_Status* status) {
  TF_Function* func = NULL;
  find_function(func, funcs, func_name);
  if (func == NULL) {
    throw std::runtime_error("Function " + func_name + " not found");
  }
  TFE_ContextAddFunction(ctx, func, status);
  check_status(status);
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
  char* data = (char*)TF_TensorData(tensor);
  // calculate the number of bytes in each string
  int64_t bytes_each_string =
      TF_TensorByteSize(tensor) / TF_TensorElementCount(tensor);
  // copy data
  std::vector<std::string> result;
  for (int ii = 0; ii < TF_TensorElementCount(tensor); ++ii) {
    result.push_back(std::string(data + ii * bytes_each_string));
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
                                   TF_Status* status) {
  TF_Tensor* data_tensor = create_tensor(data, data_shape);
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
  ctx = TFE_NewContext(ctx_opts, status);
  check_status(status);
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  int gpu_num;
  DPGetDeviceCount(gpu_num);  // check current device environment
  DPErrcheck(DPSetDevice(gpu_rank % gpu_num));
  if (gpu_num > 0) {
    device = "/gpu:" + std::to_string(gpu_rank % gpu_num);
  } else {
    device = "/cpu:0";
  }
#else
  device = "/cpu:0";
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

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

  // cast coord, fparam, and aparam to double - I think it's useless to have a
  // float model interface
  std::vector<double> coord_double(coord.begin(), coord.end());
  std::vector<double> fparam_double(fparam.begin(), fparam.end());
  std::vector<double> aparam_double(aparam.begin(), aparam.end());

  TFE_Op* op;
  if (atomic) {
    op = get_func_op(ctx, "call_lower_with_atomic_virial", func_vector, device,
                     status);
  } else {
    op = get_func_op(ctx, "call_lower_without_atomic_virial", func_vector,
                     device, status);
  }
  std::vector<TFE_TensorHandle*> input_list(6);
  // coord
  std::vector<int64_t> coord_shape = {nframes, nall_real, 3};
  input_list[0] = add_input(op, coord_double, coord_shape, status);
  // atype
  std::vector<int64_t> atype_shape = {nframes, nall_real};
  input_list[1] = add_input(op, atype, atype_shape, status);
  // nlist
  if (ago == 0) {
    nlist_data.copy_from_nlist(lmp_list);
    nlist_data.shuffle_exclude_empty(fwd_map);
  }
  std::vector<int64_t> nlist_shape = {nframes, nloc_real, nnei};
  std::vector<int64_t> nlist(static_cast<size_t>(nframes) * nloc_real * nnei);
  // pass nlist_data.jlist to nlist
  for (int ii = 0; ii < nloc_real; ii++) {
    for (int jj = 0; jj < nnei; jj++) {
      if (jj < nlist_data.jlist[ii].size()) {
        nlist[ii * nnei + jj] = nlist_data.jlist[ii][jj];
      } else {
        nlist[ii * nnei + jj] = -1;
      }
    }
    if (nnei < nlist_data.jlist[ii].size()) {
      std::cerr << "WARNING: nnei < nlist_data.jlist[ii].size(); JAX backend "
                   "never handles this."
                << std::endl;
    }
  }
  input_list[2] = add_input(op, nlist, nlist_shape, status);
  // mapping; for now, set it to -1, assume it is not used
  std::vector<int64_t> mapping_shape = {nframes, nall_real};
  std::vector<int64_t> mapping(nframes * nall_real, -1);
  input_list[3] = add_input(op, mapping, mapping_shape, status);
  // fparam
  std::vector<int64_t> fparam_shape = {nframes, dfparam};
  input_list[4] = add_input(op, fparam_double, fparam_shape, status);
  // aparam
  std::vector<int64_t> aparam_shape = {nframes, nloc_real, daparam};
  input_list[5] = add_input(op, aparam_double, aparam_shape, status);
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
  for (int i = 0; i < 6; i++) {
    TFE_DeleteTensorHandle(input_list[i]);
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
  throw deepmd::deepmd_exception("not implemented");
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
  throw deepmd::deepmd_exception("not implemented");
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