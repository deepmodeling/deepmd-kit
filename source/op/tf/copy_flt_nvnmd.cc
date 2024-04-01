

// SPDX-License-Identifier: LGPL-3.0-or-later
//
// --------------------------------------------------------------------
/*

# Function
y1 = float(x)
y2 = float(x)

# float64:
1 bit sign
11 bits exponent
52 bits fraction

# float
1 bit sign
8 bits exponent
21 bits fraction

# there
1 bit sign
11 bits exponent
21 bits fraction

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include <vector>

#include "custom_op.h"
#include "env_mat_nvnmd.h"
#include "math.h"

using namespace tensorflow;

//- register the operator
REGISTER_OP("CopyFltNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Output("y1: T")
    .Output("y2: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class CopyFltNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit CopyFltNvnmdOp(OpKernelConstruction* context) : OpKernel(context){};

  /// Compute the descriptor
  /// param: context
  void Compute(OpKernelContext* context) override {
    // check
    DCHECK_EQ(1, context->num_inputs());
    const Tensor& X = context->input(0);

    const TensorShape& shX = X.shape();
    TensorShape shY;

    int H, N, M;
    if (shX.dims() == 3) {
      H = shX.dim_size(0);
      N = shX.dim_size(1);
      M = shX.dim_size(2);

      shY.AddDim(H);
      shY.AddDim(N);
      shY.AddDim(M);
    }
    if (shX.dims() == 2) {
      // process 2-dimension as 3-dimension
      H = 1;
      N = shX.dim_size(0);
      M = shX.dim_size(1);

      shY.AddDim(N);
      shY.AddDim(M);
    }

    // create output
    Tensor* Y1 = NULL;
    Tensor* Y2 = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y1));
    OP_REQUIRES_OK(context, context->allocate_output(1, shY, &Y2));

    // compute
    auto x = X.flat<FPTYPE>().data();
    auto y1 = Y1->flat<FPTYPE>().data();
    auto y2 = Y2->flat<FPTYPE>().data();

    int ii;
    U_Flt64_Int64 ufi;

    for (ii = 0; ii < H * N * M; ii++) {
      ufi.nflt = x[ii];
      // 1.52 - 1.21 = 32
      ufi.nint &= FLT_MASK;
      y1[ii] = ufi.nflt;
      y2[ii] = ufi.nflt;
    }
  }  // Compute

};  // CopyFltNvnmdOp

#define REGISTER_CPU(T)                                               \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("CopyFltNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CopyFltNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
