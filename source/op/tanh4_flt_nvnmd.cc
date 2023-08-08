

// SPDX-License-Identifier: LGPL-3.0-or-later
// New Activation Function of NVNMD
// --------------------------------------------------------------------
/*

# Function
y = tanh4(x)
y = f(x) = a*x3*|x| + b*x3 + d*x
a = 1/16
b = -1/4
d = 1

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include "custom_op.h"
#include "env_mat_nvnmd.h"

using namespace tensorflow;

//- register the operator
REGISTER_OP("Tanh4FltNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class Tanh4FltNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit Tanh4FltNvnmdOp(OpKernelConstruction* context) : OpKernel(context) {
    //- define the attribute of context
    //* the context is the input from your tensorflow code
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
    DCHECK_EQ(1, context->num_inputs());

    //- 2.get tensor
    const Tensor& X = context->input(0);

    //- 3. get shape and check
    const TensorShape& shX = X.shape();

    int N = shX.dim_size(0);
    int M = shX.dim_size(1);

    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */

    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(N);
    shY.AddDim(M);

    Tensor* Y = NULL;

    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    auto xs = X.matrix<FPTYPE>();
    auto ys = Y->matrix<FPTYPE>();
    FPTYPE prec23, prec21, prec19, prec17, prec15;
    FPTYPE prechi, preclo;
    FPTYPE x, xhi, xlo, xa, xx, xxhi, xxlo;
    FPTYPE y;

    // calculate
    int ii, jj, kk;

    prec23 = (FPTYPE)8388608.0;  // 2^23
    prec21 = (FPTYPE)2097152.0;  // 2^32
    prec19 = (FPTYPE)524288.0;   // 2^19
    prec17 = (FPTYPE)131072.0;   // 2^17
    prec15 = (FPTYPE)32768.0;    // 2^15

    prechi = prec23;
    preclo = prec19;

    for (ii = 0; ii < N; ii++) {
      for (jj = 0; jj < M; jj++) {
        x = xs(ii, jj);
        xa = (x < 0) ? (-x) : x;
        xhi = floor(xa * prechi) / prechi;
        xlo = floor(xa * preclo) / preclo;
        xx = xhi * xlo;
        xxhi = floor(xx * prechi) / prechi;
        xxlo = floor(xx * preclo) / preclo;
        //
        if (xa < (FPTYPE)2.0) {
          y = xxhi * (xxhi * (FPTYPE)0.0625 - xhi * (FPTYPE)0.25) + xhi;
          // y = xxlo * (xxhi * (FPTYPE)0.0625 - xhi * (FPTYPE)0.25) + xhi;
        } else {
          y = 1;
        }
        //
        y = floor(y * prechi) / prechi;
        ys(ii, jj) = (x < 0) ? (-y) : y;
      }  // loop jj
    }    // loop ii
  }      // Compute

  //- define the private variable for calculation
};  // Tanh4FltNvnmd

#define REGISTER_CPU(T)                                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("Tanh4FltNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Tanh4FltNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
