

//
// --------------------------------------------------------------------
/*

# 功能
将输入的 x 量化为 nbit 位小数位的定点数，然后输出

# 参数
nbit x 量化的小数位
nbit2 dy_dx量化的小数位
nbit3 dy2_dx2量化的小数位

# 注意
1. nbit < 0 时， 表示不需要量化，直接输出浮点数
2. 为了实现简便，且范用，x的维度为2个维度

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include "custom_op.h"
#include <cmath>
#include <stdio.h>

using namespace tensorflow;


//- register the operator
REGISTER_OP("QuantizeNvnmd")
  .Attr("T: {float, double} = DT_DOUBLE")
  .Input("x: T")
  .Attr("isround: int")
  .Attr("nbit1: int")
  .Attr("nbit2: int")
  .Attr("nbit3: int")
  .Output("y: T")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c){
    shape_inference::ShapeHandle shX;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &shX));
    shape_inference::DimensionHandle D1 = c->Dim(shX, 0);
    shape_inference::DimensionHandle D2 = c->Dim(shX, 1);
    c->set_output(0, c->Matrix(D1, D2));
    return Status::OK();
  });



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
    
    int D1 = shX.dim_size(0);
    int D2 = shX.dim_size(1);
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(D1);
    shY.AddDim(D2);
    
    Tensor* Y = NULL;
    
    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    auto x = X.matrix<FPTYPE>();
    auto y = Y->matrix<FPTYPE>();
    FPTYPE prec;
    
    // calculate
    int ii, jj;

    if (this->nbit1 < 0){
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          y(ii, jj) = x(ii, jj);
        }
      }
    }
    //
    else {
      prec = 1 << this->nbit1;

      if (this->isround)
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          y(ii, jj) = round(x(ii, jj) * prec) / prec;
        }
      }
      //
      else
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          y(ii, jj) = floor(x(ii, jj) * prec) / prec;
        }
      }
    }
  }
  
//- define the private variable for calculation
private:
int nbit1, nbit2, nbit3;
int isround;
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER( \
    Name("QuantizeNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    QuantizeNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);                  
REGISTER_CPU(double);



