

// SPDX-License-Identifier: LGPL-3.0-or-later
// Quantization Operator of NVNMD
// --------------------------------------------------------------------
/*

# Function
prec = 2**nbit
y = quantize(x * prec) / prec
quantize is floor/round

# Parameter
@nbit nbit for x
@nbit2 nbit for dy_dx
@nbit3 nbit for dy2_dx2

# Note
1. if nbit < 0， y = x
2. The operator is only used for 2D tensor.

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include "custom_op.h"

using namespace tensorflow;

//- register the operator
REGISTER_OP("QuantizeNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Attr("isround: int")
    .Attr("nbit1: int")
    .Attr("nbit2: int")
    .Attr("nbit3: int")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class QuantizeNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit QuantizeNvnmdOp(OpKernelConstruction* context) : OpKernel(context) {
    //- define the attribute of context
    //* the context is the input from your tensorflow code
    OP_REQUIRES_OK(context, context->GetAttr("nbit1", &nbit1));
    OP_REQUIRES_OK(context, context->GetAttr("nbit2", &nbit2));
    OP_REQUIRES_OK(context, context->GetAttr("nbit3", &nbit3));
    OP_REQUIRES_OK(context, context->GetAttr("isround", &isround));
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

    TensorShape shY;

    int N;
    if (shX.dims() == 1) {
      shY.AddDim(shX.dim_size(0));
      N = shX.dim_size(0);
    }
    if (shX.dims() == 2) {
      shY.AddDim(shX.dim_size(0));
      shY.AddDim(shX.dim_size(1));
      N = shX.dim_size(0) * shX.dim_size(1);
    }
    if (shX.dims() == 3) {
      shY.AddDim(shX.dim_size(0));
      shY.AddDim(shX.dim_size(1));
      shY.AddDim(shX.dim_size(2));
      N = shX.dim_size(0) * shX.dim_size(1) * shX.dim_size(2);
    }

    /*
     * Calculate the output
     */

    Tensor* Y = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    auto x = X.flat<FPTYPE>().data();
    auto y = Y->flat<FPTYPE>().data();
    FPTYPE prec;

    // calculate
    int ii;

    if (this->nbit1 < 0) {
      for (ii = 0; ii < N; ii++) {
        y[ii] = x[ii];
      }
    }
    //
    else {
      prec = 1 << this->nbit1;

      if (this->isround) {
        for (ii = 0; ii < N; ii++) {
          y[ii] = round(x[ii] * prec) / prec;
        }
      } else {
        for (ii = 0; ii < N; ii++) {
          y[ii] = floor(x[ii] * prec) / prec;
        }
      }
    }
  }  // Compute

  //- define the private variable for calculation
 private:
  int nbit1, nbit2, nbit3;
  int isround;
};

#define REGISTER_CPU(T)                                                \
  REGISTER_KERNEL_BUILDER(                                             \
      Name("QuantizeNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      QuantizeNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
