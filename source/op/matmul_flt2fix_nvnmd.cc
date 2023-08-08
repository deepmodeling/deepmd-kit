

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
void split_flt(T x, int64_t& sign, int64_t& expo, int64_t& mant);

//- register the operator
REGISTER_OP("MatmulFlt2fixNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Input("w: T")
    .Attr("nbit: int")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class MatmulFlt2fixNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit MatmulFlt2fixNvnmdOp(OpKernelConstruction* context)
      : OpKernel(context) {
    // nbit is nits of fraction part of fixed-point number
    OP_REQUIRES_OK(context, context->GetAttr("nbit", &nbit));
  };

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

    int H, N, M, K;
    if (shX.dims() == 3) {
      H = shX.dim_size(0);
      N = shX.dim_size(1);
      M = shX.dim_size(2);
      K = shW.dim_size(2);

      shY.AddDim(H);
      shY.AddDim(N);
      shY.AddDim(K);
    }
    if (shX.dims() == 2) {
      // process 2-dimension as 3-dimension
      H = 1;
      N = shX.dim_size(0);
      M = shX.dim_size(1);
      K = shW.dim_size(1);

      shY.AddDim(N);
      shY.AddDim(K);
    }

    // create output
    Tensor* Y = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));

    // compute
    auto x = X.flat<FPTYPE>().data();
    auto w = W.flat<FPTYPE>().data();
    auto y = Y->flat<FPTYPE>().data();

    int hh, ii, jj, kk;

    U_Flt64_Int64 ufi;
    int64_t sign1, sign2, sign3;
    int64_t expo1, expo2, expo3;
    int64_t mant1, mant2, mant3;
    int64_t expos;

    int64_t s;

    for (hh = 0; hh < H; hh++) {
      // matmul
      for (ii = 0; ii < N; ii++) {
        for (kk = 0; kk < K; kk++) {
          s = 0;
          for (jj = 0; jj < M; jj++) {
            // x
            split_flt(x[hh * N * M + ii * M + jj], sign1, expo1, mant1);
            mant1 >>= NBIT_CUTF;
            // w
            split_flt(w[hh * M * K + jj * K + kk], sign2, expo2, mant2);
            mant2 >>= NBIT_CUTF;
            //
            mant3 = mant1 * mant2;
            expos = expo1 + expo2 - NBIT_FLTF - NBIT_FLTF - (-nbit);
            if (expos > 0) {
              mant3 <<= expos;
            } else {
              expos = -expos;
              expos = (expos > 63) ? 63 : expos;
              mant3 >>= expos;
            }
            //
            mant3 = (sign1 ^ sign2) ? -mant3 : mant3;
            s += mant3;
          }
          ufi.nflt = FPTYPE(s) * pow(2.0, -nbit);
          ufi.nint &= FLT_MASK;
          y[hh * N * K + ii * K + kk] = ufi.nflt;
        }  // loop jj
      }    // loop ii
    }      // loop hh
  }        // Compute

 private:
  int nbit;
};  // MatmulFlt2fixNvnmdOp

#define REGISTER_CPU(T)                                                     \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MatmulFlt2fixNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MatmulFlt2fixNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
