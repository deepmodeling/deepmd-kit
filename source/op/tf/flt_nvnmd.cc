

// SPDX-License-Identifier: LGPL-3.0-or-later
//
// --------------------------------------------------------------------
/*

# Function
y = float(x)

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
REGISTER_OP("FltNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class FltNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit FltNvnmdOp(OpKernelConstruction* context) : OpKernel(context){};

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
    Tensor* Y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));

    // compute
    auto x = X.flat<FPTYPE>().data();
    auto y = Y->flat<FPTYPE>().data();

    int ii;
    U_Flt64_Int64 ufi;

    for (ii = 0; ii < H * N * M; ii++) {
      ufi.nflt = x[ii];
      ufi.nint &= FLT_MASK;
      y[ii] = ufi.nflt;
    }

  }  // Compute

};  // FltNvnmdOp

#define REGISTER_CPU(T)                                           \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("FltNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FltNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
