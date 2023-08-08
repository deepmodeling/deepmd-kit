

// SPDX-License-Identifier: LGPL-3.0-or-later
//
// --------------------------------------------------------------------
/*

# Function
y = matmul(x, w)

# Note
consider DSP is 27bit x 18bit
we change the DSP into 22 x 22

in the float64:
1 bit sign
11 bits exponent
52 bits fraction

# Attr
modx = 0: normalize x[hh, : , : ]
modx = 1: normalize x[hh, ii, : ]
modw = 0: normalize w[hh, : , : ]
modw = 1: normalize w[hh, : , kk]

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include <vector>

#include "custom_op.h"
#include "env_mat_nvnmd.h"
#include "math.h"

using namespace tensorflow;

template <class T>
void split_flt(T x, int64_t &sign, int64_t &expo, int64_t &mant);

// read matmul_flt_nvnmd.cc
template <class T>  // float and double
void find_max_expo(int64_t &max_expo, T *x, int64_t M);

// read matmul_flt_nvnmd.cc
template <class T>  // float and double
void find_max_expo(int64_t &max_expo, T *x, int64_t N, int64_t M);

//- register the operator
REGISTER_OP("DotmulFltNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Input("w: T")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class DotmulFltNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit DotmulFltNvnmdOp(OpKernelConstruction *context)
      : OpKernel(context){};

  /// Compute the descriptor
  /// param: context
  void Compute(OpKernelContext *context) override {
    // check
    DCHECK_EQ(2, context->num_inputs());
    const Tensor &X = context->input(0);
    const Tensor &W = context->input(1);

    const TensorShape &shX = X.shape();
    const TensorShape &shW = W.shape();
    TensorShape shY;
    DCHECK_EQ(shW.dims(), shX.dims());

    int H, N, M;
    if (shX.dims() == 3) {
      H = shX.dim_size(0);
      N = shX.dim_size(1);
      M = shX.dim_size(2);

      DCHECK_EQ(H, shW.dim_size(0));
      DCHECK_EQ(N, shW.dim_size(1));
      DCHECK_EQ(M, shW.dim_size(2));

      shY.AddDim(H);
      shY.AddDim(N);
      shY.AddDim(1);
    }
    if (shX.dims() == 2) {
      // process 2-dimension as 3-dimension
      H = 1;
      N = shX.dim_size(0);
      M = shX.dim_size(1);

      DCHECK_EQ(N, shW.dim_size(0));
      DCHECK_EQ(M, shW.dim_size(1));

      shY.AddDim(N);
      shY.AddDim(1);
    }

    // create output
    Tensor *Y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));

    // compute
    auto x = X.flat<FPTYPE>().data();
    auto w = W.flat<FPTYPE>().data();
    auto y = Y->flat<FPTYPE>().data();

    int hh, ii, jj;

    int nshift1, nshift2;
    int64_t s;

    U_Flt64_Int64 ufi1, ufi2, ufi3;
    int64_t sign1, sign2, sign3;
    int64_t expo1, expo2, expo3;
    int64_t mant1, mant2, mant3;
    int64_t expos;

    int64_t expo_max1, expo_max2;
    std::vector<int> expo_max1s;
    std::vector<int> expo_max2s;
    expo_max1s.resize(N);
    expo_max2s.resize(N);

    for (ii = 0; ii < H * N; ii++) {
      // find x max exponnet
      find_max_expo(expo_max1, (FPTYPE *)&x[ii * M], M);
      find_max_expo(expo_max2, (FPTYPE *)&w[ii * M], M);
      //
      s = 0;
      for (jj = 0; jj < M; jj++) {
        // x
        split_flt(x[ii * M + jj], sign1, expo1, mant1);
        mant1 >>= NBIT_CUTF;
        expos = expo_max1 - expo1;
        expos = (expos > 63) ? 63 : expos;
        mant1 >>= expos;
        // w
        split_flt(w[ii * M + jj], sign2, expo2, mant2);
        mant2 >>= NBIT_CUTF;
        expos = expo_max2 - expo2;
        expos = (expos > 63) ? 63 : expos;
        mant2 >>= expos;
        // multiply
        mant3 = mant1 * mant2;
        mant3 = (sign1 ^ sign2) ? -mant3 : mant3;
        s += mant3;
      }
      // y * 2^(e_a+e_b)
      ufi3.nflt =
          FPTYPE(s) * pow(2.0, expo_max1 + expo_max2 - NBIT_FLTF - NBIT_FLTF);
      ufi3.nint &= FLT_MASK;
      y[ii] = ufi3.nflt;
    }  // loop ii
  }    // Compute

};  // DotmulFltNvnmdOp

#define REGISTER_CPU(T)                                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DotmulFltNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DotmulFltNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
