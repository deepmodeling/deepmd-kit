

// SPDX-License-Identifier: LGPL-3.0-or-later
//
// --------------------------------------------------------------------
/*

# Function
y = float(float(x) + float(w))

# float64:
1 bit sign
11 bits exponent
52 bits fraction

# float29
1 bit sign
8 bits exponent
20 bits fraction

# there
1 bit sign
11 bits exponent
20 bits fraction

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include <vector>

#include "custom_op.h"
#include "env_mat_nvnmd.h"
#include "math.h"

using namespace tensorflow;

template <class T>  // float and double
void mul_flt_nvnmd(T& y, T x1, T x2);

//- register the operator
REGISTER_OP("MulFltNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Input("w: T")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class MulFltNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit MulFltNvnmdOp(OpKernelConstruction* context) : OpKernel(context){};

  /// Compute the descriptor
  /// param: context
  void Compute(OpKernelContext* context) override {
    // check
    DCHECK_EQ(2, context->num_inputs());
    const Tensor& X = context->input(0);
    const Tensor& W = context->input(1);

    const TensorShape& shX = X.shape();
    const TensorShape& shW = W.shape();
    TensorShape shY;

    DCHECK_EQ(shW.dims(), shX.dims());

    int H, N, M1, M2;
    if (shX.dims() == 3) {
      DCHECK_EQ(shW.dim_size(0), shX.dim_size(0));
      DCHECK_EQ(shW.dim_size(1), shX.dim_size(1));
      DCHECK_EQ(shW.dim_size(2), shX.dim_size(2));

      H = shX.dim_size(0);
      N = shX.dim_size(1);
      M1 = shX.dim_size(2);
      M2 = shW.dim_size(2);

      shY.AddDim(H);
      shY.AddDim(N);
      shY.AddDim(M2);
    }
    if (shX.dims() == 2) {
      DCHECK_EQ(shW.dim_size(0), shX.dim_size(0));
      DCHECK_EQ(shW.dim_size(1), shX.dim_size(1));

      H = 1;
      N = shX.dim_size(0);
      M1 = shX.dim_size(1);
      M2 = shW.dim_size(1);

      shY.AddDim(N);
      shY.AddDim(M2);
    }

    if (M1 != M2) {
      DCHECK_EQ(1, M1);
    }

    // create output
    Tensor* Y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));

    // compute
    auto x = X.flat<FPTYPE>().data();
    auto w = W.flat<FPTYPE>().data();
    auto y = Y->flat<FPTYPE>().data();

    int ii, jj;
    U_Flt64_Int64 ufi1, ufi2, ufi3;
    int64_t sign1, sign2, sign3;
    int64_t expo1, expo2, expo3;
    int64_t mant1, mant2, mant3;
    int64_t expos;

    if (M1 == M2) {
      for (ii = 0; ii < H * N * M2; ii++) {
        mul_flt_nvnmd(y[ii], x[ii], w[ii]);
      }
    } else {
      for (ii = 0; ii < H * N; ii++) {
        for (jj = 0; jj < M2; jj++) {
          mul_flt_nvnmd(y[ii * M2 + jj], x[ii], w[ii * M2 + jj]);
        }
      }
    }

  }  // Compute

};  // MulFltNvnmdOp

#define REGISTER_CPU(T)                                              \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MulFltNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MulFltNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
