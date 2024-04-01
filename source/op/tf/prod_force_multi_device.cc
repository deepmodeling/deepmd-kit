// SPDX-License-Identifier: LGPL-3.0-or-later
#include "custom_op.h"
#include "errors.h"
#include "prod_force.h"

REGISTER_OP("ProdForceSeA")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("force: T");

// compatible with v0.12
REGISTER_OP("ProdForceNorot")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Output("force: T");

// rename temp op
REGISTER_OP("ParallelProdForceSeA")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Attr("n_a_sel: int")
    .Attr("n_r_sel: int")
    .Attr("parallel: bool = false")
    .Attr("start_frac: float = 0.")
    .Attr("end_frac: float = 1.")
    .Output("force: T");

REGISTER_OP("ProdForceSeR")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("nlist: int32")
    .Input("natoms: int32")
    .Output("force: T");

template <typename Device, typename FPTYPE>
class ProdForceSeAOp : public OpKernel {
 public:
  explicit ProdForceSeAOp(OpKernelConstruction* context) : OpKernel(context) {
    if (context->HasAttr("parallel")) {
      OP_REQUIRES_OK(context, context->GetAttr("parallel", &parallel));
    }
    if (context->HasAttr("start_frac")) {
      OP_REQUIRES_OK(context, context->GetAttr("start_frac", &start_frac));
    }
    if (context->HasAttr("end_frac")) {
      OP_REQUIRES_OK(context, context->GetAttr("end_frac", &end_frac));
    }
  }

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& net_deriv_tensor = context->input(context_input_index++);
    const Tensor& in_deriv_tensor = context->input(context_input_index++);
    const Tensor& nlist_tensor = context->input(context_input_index++);
    const Tensor& natoms_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (net_deriv_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of net deriv should be 2"));
    OP_REQUIRES(context, (in_deriv_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input deriv should be 2"));
    OP_REQUIRES(context, (nlist_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of nlist should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                errors::InvalidArgument(
                    "number of atoms should be larger than (or equal to) 3"));
    const int* natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nframes = net_deriv_tensor.shape().dim_size(0);
    int ndescrpt = nloc > 0 ? net_deriv_tensor.shape().dim_size(1) / nloc : 0;
    int nnei = nloc > 0 ? nlist_tensor.shape().dim_size(1) / nloc : 0;
    // check the sizes
    OP_REQUIRES(context, (nframes == in_deriv_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nframes == nlist_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(
        context,
        (int_64(nloc) * ndescrpt * 3 == in_deriv_tensor.shape().dim_size(1)),
        errors::InvalidArgument("number of descriptors should match"));
    // Create an output tensor
    TensorShape force_shape;
    force_shape.AddDim(nframes);
    force_shape.AddDim(3 * static_cast<int64_t>(nall));
    Tensor* force_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++, force_shape,
                                            &force_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());
    assert(nframes == force_shape.dim_size(0));
    assert(nframes == net_deriv_tensor.shape().dim_size(0));
    assert(nframes == in_deriv_tensor.shape().dim_size(0));
    assert(nframes == nlist_tensor.shape().dim_size(0));
    assert(nall * 3 == force_shape.dim_size(1));
    assert(static_cast<int64_t>(nloc) * ndescrpt ==
           net_deriv_tensor.shape().dim_size(1));
    assert(static_cast<int64_t>(nloc) * ndescrpt * 3 ==
           in_deriv_tensor.shape().dim_size(1));
    assert(static_cast<int64_t>(nloc) * nnei ==
           nlist_tensor.shape().dim_size(1));
    assert(nnei * 4 == ndescrpt);

    // flat the tensors
    FPTYPE* p_force = force_tensor->flat<FPTYPE>().data();
    const FPTYPE* p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE* p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const int* p_nlist = nlist_tensor.flat<int>().data();

    int start_index = 0, end_index = nloc, nloc_loc = nloc;
    if (parallel) {
      if (device != "CPU") {
        throw deepmd::deepmd_exception(
            "Auto parallelization for ProdForceA is not supported on GPUs!");
      }
      // we split in_deriv, net_deriv, and nlist along nloc
      // compute start and end index along nloc
      // frac belongs to [0, 1]
      // end_index will be not visited, only visit end_index-1
      start_index = lround(start_frac * nloc);
      end_index = lround(end_frac * nloc);
      nloc_loc = end_index - start_index;
    }

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::prod_force_a_gpu(p_force, p_net_deriv, p_in_deriv, p_nlist, nloc,
                               nall, nnei, nframes);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::prod_force_a_cpu(p_force, p_net_deriv, p_in_deriv, p_nlist, nloc,
                               nall, nnei, nframes, nloc_loc,
                               start_index = start_index);
    }
  }

 private:
  std::string device;
  bool parallel = false;
  float start_frac = 0.f;
  float end_frac = 1.f;
};

template <typename Device, typename FPTYPE>
class ProdForceSeROp : public OpKernel {
 public:
  explicit ProdForceSeROp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& net_deriv_tensor = context->input(context_input_index++);
    const Tensor& in_deriv_tensor = context->input(context_input_index++);
    const Tensor& nlist_tensor = context->input(context_input_index++);
    const Tensor& natoms_tensor = context->input(context_input_index++);
    // set size of the sample
    OP_REQUIRES(context, (net_deriv_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of net deriv should be 2"));
    OP_REQUIRES(context, (in_deriv_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input deriv should be 2"));
    OP_REQUIRES(context, (nlist_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of nlist should be 2"));
    OP_REQUIRES(context, (natoms_tensor.shape().dims() == 1),
                errors::InvalidArgument("Dim of natoms should be 1"));
    OP_REQUIRES(context, (natoms_tensor.shape().dim_size(0) >= 3),
                errors::InvalidArgument(
                    "number of atoms should be larger than (or equal to) 3"));
    const int* natoms = natoms_tensor.flat<int>().data();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nframes = net_deriv_tensor.shape().dim_size(0);
    int ndescrpt = nloc > 0 ? net_deriv_tensor.shape().dim_size(1) / nloc : 0;
    int nnei = nloc > 0 ? nlist_tensor.shape().dim_size(1) / nloc : 0;
    // check the sizes
    OP_REQUIRES(context, (nframes == in_deriv_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nframes == nlist_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context,
                (static_cast<int64_t>(nloc) * ndescrpt * 3 ==
                 in_deriv_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of descriptors should match"));
    // Create an output tensor
    TensorShape force_shape;
    force_shape.AddDim(nframes);
    force_shape.AddDim(3 * static_cast<int64_t>(nall));
    Tensor* force_tensor = NULL;
    int context_output_index = 0;
    OP_REQUIRES_OK(context,
                   context->allocate_output(context_output_index++, force_shape,
                                            &force_tensor));
    DeviceFunctor()(device, context->eigen_device<Device>());
    assert(nframes == force_shape.dim_size(0));
    assert(nframes == net_deriv_tensor.shape().dim_size(0));
    assert(nframes == in_deriv_tensor.shape().dim_size(0));
    assert(nframes == nlist_tensor.shape().dim_size(0));
    assert(nall * 3 == force_shape.dim_size(1));
    assert(static_cast<int64_t>(nloc) * ndescrpt ==
           net_deriv_tensor.shape().dim_size(1));
    assert(static_cast<int64_t>(nloc) * ndescrpt * 3 ==
           in_deriv_tensor.shape().dim_size(1));
    assert(nloc * nnei == nlist_tensor.shape().dim_size(1));
    assert(nnei * 1 == ndescrpt);
    // flat the tensors
    FPTYPE* p_force = force_tensor->flat<FPTYPE>().data();
    const FPTYPE* p_net_deriv = net_deriv_tensor.flat<FPTYPE>().data();
    const FPTYPE* p_in_deriv = in_deriv_tensor.flat<FPTYPE>().data();
    const int* p_nlist = nlist_tensor.flat<int>().data();

    if (device == "GPU") {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      deepmd::prod_force_r_gpu(p_force, p_net_deriv, p_in_deriv, p_nlist, nloc,
                               nall, nnei, nframes);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    } else if (device == "CPU") {
      deepmd::prod_force_r_cpu(p_force, p_net_deriv, p_in_deriv, p_nlist, nloc,
                               nall, nnei, nframes);
    }
  }

 private:
  std::string device;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ProdForceSeA").Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
      ProdForceSeAOp<CPUDevice, T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ProdForceNorot").Device(DEVICE_CPU).TypeConstraint<T>("T"),       \
      ProdForceSeAOp<CPUDevice, T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ParallelProdForceSeA").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ProdForceSeAOp<CPUDevice, T>);                                          \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ProdForceSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),         \
      ProdForceSeROp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
// Register the GPU kernels.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU(T)                                  \
  REGISTER_KERNEL_BUILDER(Name("ProdForceSeA")           \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("natoms"),     \
                          ProdForceSeAOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ProdForceNorot")         \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("natoms"),     \
                          ProdForceSeAOp<GPUDevice, T>); \
  REGISTER_KERNEL_BUILDER(Name("ProdForceSeR")           \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<T>("T")    \
                              .HostMemory("natoms"),     \
                          ProdForceSeROp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
