// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ComputeDescriptor.h"
#include "custom_op.h"
#include "device.h"
#include "neighbor_list.h"

#define GGELU 0.044715

REGISTER_OP("UnaggregatedDyDxS")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("y: T")
    .Input("w: T")
    .Input("xbar: T")
    .Input("functype: int32")
    .Output("dy_dx: T");

REGISTER_OP("UnaggregatedDyDx")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("z: T")
    .Input("w: T")
    .Input("dy_dx: T")
    .Input("ybar: T")
    .Input("functype: int32")
    .Output("dz_dx: T");

REGISTER_OP("UnaggregatedDy2DxS")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("y: T")
    .Input("dy: T")
    .Input("w: T")
    .Input("xbar: T")
    .Input("functype: int32")
    .Output("dy2_dx: T");

REGISTER_OP("UnaggregatedDy2Dx")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("z: T")
    .Input("w: T")
    .Input("dy_dx: T")
    .Input("dy2_dx: T")
    .Input("ybar: T")
    .Input("functype: int32")
    .Output("dz2_dx: T");
template <typename FPTYPE>
FPTYPE grad(const FPTYPE xbar,
            const FPTYPE y,
            const int functype)  // functype=tanh, gelu, ..
{
  switch (functype) {
    case 1:
      return (1 - y * y);
    case 2: {
      const FPTYPE var = tanh(SQRT_2_PI * (xbar + GGELU * xbar * xbar * xbar));
      return 0.5 * SQRT_2_PI * xbar * (1 - var * var) *
                 (3 * GGELU * xbar * xbar + 1) +
             0.5 * var + 0.5;
    }
    case 3: {
      if (xbar <= 0) {
        return 0;
      } else {
        return 1;
      }
    }
    case 4: {
      if (xbar <= 0 || xbar >= 6) {
        return 0;
      } else {
        return 1;
      }
    }
    case 5: {
      return 1.0 - 1.0 / (1.0 + exp(xbar));
    }
    case 6: {
      return y * (1 - y);
    }
    default:
      return -1;
  }
}

template <typename FPTYPE>
FPTYPE grad_grad(const FPTYPE xbar, const FPTYPE y, const int functype) {
  switch (functype) {
    case 1:
      return -2 * y * (1 - y * y);
    case 2: {
      const FPTYPE var1 = tanh(SQRT_2_PI * (xbar + GGELU * xbar * xbar * xbar));
      const FPTYPE var2 =
          SQRT_2_PI * (1 - var1 * var1) * (3 * GGELU * xbar * xbar + 1);
      return 3 * GGELU * SQRT_2_PI * xbar * xbar * (1 - var1 * var1) -
             SQRT_2_PI * xbar * var2 * (3 * GGELU * xbar * xbar + 1) * var1 +
             var2;
    }
    case 3: {
      return 0;
    }
    case 4: {
      return 0;
    }
    case 5: {
      return exp(xbar) / ((1 + exp(xbar)) * (1 + exp(xbar)));
    }
    case 6: {
      return y * (1 - y) * (1 - 2 * y);
    }
    default:
      return -1;
  }
}

template <typename FPTYPE>
struct UnaggregatedDyDxSFunctor {
  void operator()(const CPUDevice& d,
                  const FPTYPE* y,
                  const FPTYPE* w,
                  const FPTYPE* xbar,
                  const int length,
                  const int width,
                  FPTYPE* dy_dx,
                  const int functype) {
#pragma omp parallel for
    for (int ii = 0; ii < length; ii++) {
      for (int jj = 0; jj < width; jj++) {
        dy_dx[ii * width + jj] =
            grad(xbar[ii * width + jj], y[ii * width + jj], functype) * w[jj];
      }
    }
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  void operator()(const GPUDevice& d,
                  const FPTYPE* y,
                  const FPTYPE* w,
                  const int length,
                  const int width,
                  FPTYPE* dy_dx) {
    // Currently, Do nothing at all!
    return;
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
};

// calculate the gradient for all variables!
template <typename FPTYPE>
struct UnaggregatedDyDxFunctor {
  void operator()(const CPUDevice& d,
                  const FPTYPE* z,
                  const FPTYPE* w,
                  const FPTYPE* dy_dx,
                  const FPTYPE* ybar,
                  const int length,
                  const int width,
                  const int size,
                  FPTYPE* dz_dx,
                  const int functype) {
// width=2*size
#pragma omp parallel for
    for (int kk = 0; kk < length; kk++) {
      for (int ii = 0; ii < width; ii++) {
        // FPTYPE dz_drou = 1 - (z[kk * width + ii] - y[kk * size + ii % size])
        // * (z[kk * width + ii] - y[kk * size + ii % size]);
        FPTYPE dz_drou =
            grad(ybar[kk * width + ii], z[kk * width + ii], functype);
        FPTYPE accumulator = 0.0;
        for (int jj = 0; jj < size; jj++) {
          accumulator += w[jj * width + ii] * dy_dx[kk * size + jj];
        }
        dz_drou *= accumulator;
        if (width == 2 * size || width == size) {
          dz_drou += dy_dx[kk * size + ii % size];
        }
        dz_dx[kk * width + ii] = dz_drou;
      }
    }
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  void operator()(const GPUDevice& d,
                  const FPTYPE* z,
                  const FPTYPE* w,
                  const FPTYPE* dy_dx,
                  const int length,
                  const int width,
                  const int size,
                  FPTYPE* dz_dx) {
    // Currently, Do nothing at all!
    return;
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
};

template <typename FPTYPE>
struct UnaggregatedDy2DxSFunctor {
  void operator()(const CPUDevice& d,
                  const FPTYPE* y,
                  const FPTYPE* dy,
                  const FPTYPE* w,
                  const FPTYPE* xbar,
                  const int length,
                  const int width,
                  FPTYPE* dy2_dx,
                  const int functype) {
#pragma omp parallel for
    for (int ii = 0; ii < length; ii++) {
      for (int jj = 0; jj < width; jj++) {
        dy2_dx[ii * width + jj] =
            grad_grad(xbar[ii * width + jj], y[ii * width + jj], functype) *
            w[jj] * w[jj];
      }
    }
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  void operator()(const GPUDevice& d,
                  const FPTYPE* y,
                  const FPTYPE* w,
                  const int length,
                  const int width,
                  FPTYPE* dy_dx) {
    // Currently, Do nothing at all!
    return;
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
};

// calculate the gradient for all variables!
template <typename FPTYPE>
struct UnaggregatedDy2DxFunctor {
  void operator()(const CPUDevice& d,
                  const FPTYPE* z,
                  const FPTYPE* w,
                  const FPTYPE* dy_dx,
                  const FPTYPE* dy2_dx,
                  const FPTYPE* ybar,
                  const int length,
                  const int width,
                  const int size,
                  FPTYPE* dz2_dx,
                  const int functype) {
#pragma omp parallel for
    for (int kk = 0; kk < length; kk++) {
      for (int ii = 0; ii < width; ii++) {
        // FPTYPE dz_drou = 1 - (z[kk * width + ii] - y[kk * size + ii % size])
        // * (z[kk * width + ii] - y[kk * size + ii % size]);
        FPTYPE dz_drou =
            grad(ybar[kk * width + ii], z[kk * width + ii], functype);
        FPTYPE accumulator = 0.0;
        for (int jj = 0; jj < size; jj++) {
          accumulator += w[jj * width + ii] * dy2_dx[kk * size + jj];
        }
        dz_drou *= accumulator;
        accumulator = 0.0;
        for (int jj = 0; jj < size; jj++) {
          accumulator += w[jj * width + ii] * dy_dx[kk * size + jj];
        }
        dz_drou +=
            grad_grad(ybar[kk * width + ii], z[kk * width + ii], functype) *
            accumulator * accumulator;
        if (width == 2 * size || width == size) {
          dz_drou += dy2_dx[kk * size + ii % size];
        }
        dz2_dx[kk * width + ii] = dz_drou;
      }
    }
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  void operator()(const GPUDevice& d,
                  const FPTYPE* z,
                  const FPTYPE* w,
                  const FPTYPE* dz_dx,
                  const FPTYPE* dy_dx,
                  const FPTYPE* dy2_dx,
                  const int length,
                  const int width,
                  const int size,
                  FPTYPE* dz2_dx) {
    // Currently, Do nothing at all!
    return;
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
};

template <typename Device, typename FPTYPE>
class UnaggregatedDyDxSOp : public OpKernel {
 public:
  explicit UnaggregatedDyDxSOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    // xbar=xw+b
    int context_input_index = 0;
    const Tensor& y = context->input(context_input_index++);
    const Tensor& w = context->input(context_input_index++);
    const Tensor& xbar = context->input(context_input_index++);
    const Tensor& functype = context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES(context, (y.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (w.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (xbar.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    // check functype

    int context_output_index = 0;
    Tensor* dy_dx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     y.shape(), &dy_dx));

    UnaggregatedDyDxSFunctor<FPTYPE>()(
        context
            ->eigen_device<Device>(),  // define actually graph execution device
        y.flat<FPTYPE>().data(), w.flat<FPTYPE>().data(),
        xbar.flat<FPTYPE>().data(), y.shape().dim_size(0),
        y.shape().dim_size(1), dy_dx->flat<FPTYPE>().data(),
        functype.flat<int32>()(0));
  }

 private:
};

template <typename Device, typename FPTYPE>
class UnaggregatedDy2DxSOp : public OpKernel {
 public:
  explicit UnaggregatedDy2DxSOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& y = context->input(context_input_index++);
    const Tensor& dy = context->input(context_input_index++);
    const Tensor& w = context->input(context_input_index++);
    const Tensor& xbar = context->input(context_input_index++);
    const Tensor& functype = context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES(context, (y.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (dy.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (w.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (xbar.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));

    int context_output_index = 0;
    Tensor* dy2_dx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     y.shape(), &dy2_dx));

    UnaggregatedDy2DxSFunctor<FPTYPE>()(
        context
            ->eigen_device<Device>(),  // define actually graph execution device
        y.flat<FPTYPE>().data(), dy.flat<FPTYPE>().data(),
        w.flat<FPTYPE>().data(), xbar.flat<FPTYPE>().data(),
        y.shape().dim_size(0), y.shape().dim_size(1),
        dy2_dx->flat<FPTYPE>().data(), functype.flat<int32>()(0));
  }

 private:
};

template <typename Device, typename FPTYPE>
class UnaggregatedDyDxOp : public OpKernel {
 public:
  explicit UnaggregatedDyDxOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& z = context->input(context_input_index++);
    const Tensor& w = context->input(context_input_index++);
    const Tensor& dy_dx = context->input(context_input_index++);
    const Tensor& ybar = context->input(context_input_index++);
    const Tensor& functype = context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES(context, (z.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (w.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (dy_dx.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (ybar.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));

    int context_output_index = 0;
    Tensor* dz_dx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     z.shape(), &dz_dx));

    UnaggregatedDyDxFunctor<FPTYPE>()(
        context
            ->eigen_device<Device>(),  // define actually graph execution device
        z.flat<FPTYPE>().data(), w.flat<FPTYPE>().data(),
        dy_dx.flat<FPTYPE>().data(), ybar.flat<FPTYPE>().data(),
        z.shape().dim_size(0),
        z.shape().dim_size(1),  // N1
        w.shape().dim_size(0),  // N0 , N1=2N0
        dz_dx->flat<FPTYPE>().data(), functype.flat<int32>()(0));
  }

 private:
};

template <typename Device, typename FPTYPE>
class UnaggregatedDy2DxOp : public OpKernel {
 public:
  explicit UnaggregatedDy2DxOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext* context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext* context) {
    // Grab the input tensor
    int context_input_index = 0;
    const Tensor& z = context->input(context_input_index++);
    const Tensor& w = context->input(context_input_index++);
    const Tensor& dy_dx = context->input(context_input_index++);
    const Tensor& dy2_dx = context->input(context_input_index++);
    const Tensor& ybar = context->input(context_input_index++);
    const Tensor& functype = context->input(context_input_index++);

    // set size of the sample
    OP_REQUIRES(context, (z.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (w.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (dy_dx.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (dy2_dx.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));
    OP_REQUIRES(context, (ybar.shape().dims() == 2),
                errors::InvalidArgument("Dim of input should be 2"));

    int context_output_index = 0;
    Tensor* dz2_dx = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
                                                     z.shape(), &dz2_dx));

    UnaggregatedDy2DxFunctor<FPTYPE>()(
        context
            ->eigen_device<Device>(),  // define actually graph execution device
        z.flat<FPTYPE>().data(), w.flat<FPTYPE>().data(),
        dy_dx.flat<FPTYPE>().data(), dy2_dx.flat<FPTYPE>().data(),
        ybar.flat<FPTYPE>().data(), z.shape().dim_size(0),
        z.shape().dim_size(1), w.shape().dim_size(0),
        dz2_dx->flat<FPTYPE>().data(), functype.flat<int32>()(0));
  }

 private:
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("UnaggregatedDyDxS").Device(DEVICE_CPU).TypeConstraint<T>("T"),  \
      UnaggregatedDyDxSOp<CPUDevice, T>);                                   \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("UnaggregatedDyDx").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      UnaggregatedDyDxOp<CPUDevice, T>);                                    \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("UnaggregatedDy2DxS").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      UnaggregatedDy2DxSOp<CPUDevice, T>);                                  \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("UnaggregatedDy2Dx").Device(DEVICE_CPU).TypeConstraint<T>("T"),  \
      UnaggregatedDy2DxOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
// Not required in the current situation
// // Register the GPU kernels.
// #if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// #define REGISTER_GPU(T) \
// REGISTER_KERNEL_BUILDER( \
//     Name("UnaggregatedDyDxS").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
//     UnaggregatedDyDxSOp<GPUDevice, T>); \
// REGISTER_KERNEL_BUILDER( \
//     Name("UnaggregatedDyDx").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
//     UnaggregatedDyDxOp<GPUDevice, T>);
// REGISTER_GPU(float);
// REGISTER_GPU(double);
// #endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
