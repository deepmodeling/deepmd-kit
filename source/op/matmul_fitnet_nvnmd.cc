

// SPDX-License-Identifier: LGPL-3.0-or-later
//
// --------------------------------------------------------------------
/*

# Function
y = matmul(x, w)

# Note
consider DSP is 27bit x 18bit
integer part of x is set as 27 bit
integer part of w is set as 18 bit

in the float64:
1 bit sign
11 bits exponent
52 bits fraction

x use 27.23 bit fixed point number
w use 18.16 bit fixed point number add a exponent of normalization

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include <vector>

#include "custom_op.h"
#include "env_mat_nvnmd.h"
#include "math.h"

using namespace tensorflow;

// read matmul_flt_nvnmd.cc
template <class T>  // float and double
void find_max_expo(int64_t& max_expo, T* x, int64_t M);

// read matmul_flt_nvnmd.cc
template <class T>  // float and double
void find_max_expo(int64_t& max_expo, T* x, int64_t N, int64_t M);

//- register the operator
REGISTER_OP("MatmulFitnetNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Input("w: T")
    .Attr("nbitx: int")
    .Attr("nbitw: int")
    .Attr("normw: int")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class MatmulFitnetNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit MatmulFitnetNvnmdOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("nbitx", &nbitx));
    OP_REQUIRES_OK(context, context->GetAttr("nbitw", &nbitw));
    OP_REQUIRES_OK(context, context->GetAttr("normw", &normw));
  }

  /// Compute the descriptor
  /// param: context
  void Compute(OpKernelContext* context) override {
    /*
     * Get input
     * 1.check
     * 2.get tensor
     * 3.get shape and check
     */

    //- 1.check
    DCHECK_EQ(2, context->num_inputs());

    //- 2.get tensor
    const Tensor& X = context->input(0);
    const Tensor& W = context->input(1);

    //- 3. get shape and check
    const TensorShape& shX = X.shape();
    const TensorShape& shW = W.shape();

    int N = shX.dim_size(0);
    int M = shX.dim_size(1);
    int K = shW.dim_size(1);

    DCHECK_EQ(M, shW.dim_size(0));

    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */

    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(N);
    shY.AddDim(K);

    Tensor* Y = NULL;

    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    auto x = X.flat<FPTYPE>().data();
    auto w = W.flat<FPTYPE>().data();
    auto y = Y->flat<FPTYPE>().data();

    // calculate
    int ii, jj, kk;

    U_Flt64_Int64 ufi;
    int64_t expo, expo_max;
    FPTYPE prec, div_prec;
    FPTYPE precx, div_precx;
    FPTYPE precw, div_precw;
    precw = pow((FPTYPE)2.0, nbitw);
    div_precw = (FPTYPE)1.0 / precw;
    precx = pow((FPTYPE)2.0, nbitx);
    div_precx = (FPTYPE)1.0 / precx;

    FPTYPE xij, wjk, s;

    // find max exponent of w
    std::vector<int> expo_maxs;
    expo_maxs.resize(K);

    if (normw == 0) {
      find_max_expo(expo_max, (FPTYPE*)&w[0], static_cast<int64_t>(M) * K);
      for (kk = 0; kk < K; kk++) {
        expo_maxs[kk] = expo_max;
      }
    } else {
      for (kk = 0; kk < K; kk++) {
        find_max_expo(expo_max, (FPTYPE*)&w[kk], M, K);
        expo_maxs[kk] = expo_max;
      }
    }

    // calculate
    for (kk = 0; kk < K; kk++) {
      expo_max = expo_maxs[kk];
      prec = pow((FPTYPE)2.0, expo_max);
      div_prec = (FPTYPE)1.0 / prec;
      // matmul
      for (ii = 0; ii < N; ii++) {
        s = 0;
        for (jj = 0; jj < M; jj++) {
          wjk = floor(w[jj * K + kk] * div_prec * precw) * div_precw;
          xij = floor(x[ii * M + jj] * precx) * div_precx;
          s += xij * wjk;
        }
        s = floor(s * prec * precx) * div_precx;
        y[ii * K + kk] = s;
      }  // loop xx
    }    // loop kk

  }  // Compute

  //- define the private variable for calculation
 private:
  int nbitx, nbitw;
  int normw;
};  // MatmulFitnetNvnmd

#define REGISTER_CPU(T)                                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatmulFitnetNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MatmulFitnetNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
