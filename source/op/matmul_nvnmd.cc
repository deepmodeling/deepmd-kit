

//
// --------------------------------------------------------------------
/*

# Function
y = q(matmul(x, w), nbit)
q is a quantization function

# Parameter
nbit nbit for x
nbit2 nbit for dy_dx
nbit3 nbit for dy2_dx2

# Note
1. if nbit < 0, y = matmul(x, w)
2. the input and output dimensions must be 2

*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include "custom_op.h"

using namespace tensorflow;


//- register the operator
REGISTER_OP("MatmulNvnmd")
  .Attr("T: {float, double} = DT_DOUBLE")
  .Input("x: T")
  .Input("w: T")
  .Attr("isround: int")
  .Attr("nbit1: int")
  .Attr("nbit2: int")
  .Attr("nbit3: int")
  .Output("y: T");



//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class MatmulNvnmdOp : public OpKernel {
public:

  /// Constructor.
  explicit MatmulNvnmdOp(OpKernelConstruction* context) : OpKernel(context) {
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
    DCHECK_EQ(2, context->num_inputs());
    
    //- 2.get tensor
    const Tensor& X = context->input(0);
    const Tensor& W = context->input(1);
    
    //- 3. get shape and check
    const TensorShape& shX = X.shape();
    const TensorShape& shW = W.shape();
    
    int D1 = shX.dim_size(0);
    int D2 = shX.dim_size(1);
    int D3 = shW.dim_size(0);
    int D4 = shW.dim_size(1);
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(D1);
    shY.AddDim(D4);
    
    Tensor* Y = NULL;
    
    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    auto x = X.matrix<FPTYPE>();
    auto w = W.matrix<FPTYPE>();
    auto y = Y->matrix<FPTYPE>();
    FPTYPE prec;

    
    // calculate
    int ii, jj, kk;
    
    for(ii=0; ii<D1; ii++){
        for(kk=0; kk<D4; kk++){
          y(ii, kk) = (FPTYPE)0.0;
        }
      }

    if (this->nbit1 < 0){
      for(ii=0; ii<D1; ii++){
        for(kk=0; kk<D4; kk++){
          for(jj=0; jj<D2; jj++){
            y(ii, kk) += x(ii, jj) * w(jj, kk);
          }
        }
      }
    }
    //
    else {
      prec = 1 << this->nbit1;

      if (this->isround)
        for(ii=0; ii<D1; ii++){
          for(kk=0; kk<D4; kk++){
            for(jj=0; jj<D2; jj++){
              y(ii, kk) += x(ii, jj) * w(jj, kk);
            }
            y(ii, kk) = round(y(ii, kk) * prec)/ prec;
          }
        }
      //
      else
        for(ii=0; ii<D1; ii++){
          for(kk=0; kk<D4; kk++){
            for(jj=0; jj<D2; jj++){
              y(ii, kk) += x(ii, jj) * w(jj, kk);
            }
            y(ii, kk) = floor(y(ii, kk) * prec)/ prec;
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
    Name("MatmulNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    MatmulNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);                  
REGISTER_CPU(double);



