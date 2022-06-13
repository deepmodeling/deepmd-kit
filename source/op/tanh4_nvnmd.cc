

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

using namespace tensorflow;


//- register the operator
REGISTER_OP("Tanh4Nvnmd")
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
class Tanh4NvnmdOp : public OpKernel {
public:

  /// Constructor.
  explicit Tanh4NvnmdOp(OpKernelConstruction* context) : OpKernel(context) {
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
    auto xs = X.matrix<FPTYPE>();
    auto ys = Y->matrix<FPTYPE>();
    FPTYPE prec, prec4;
    FPTYPE x, xa, x1, x2, x3, x4, xx, xxa, xx4;
    FPTYPE a, b, d;
    FPTYPE a1, b1, d1;
    FPTYPE a2, b2, d2;
    FPTYPE y, y1, y2;
    FPTYPE H1, H2;

    
    // calculate
    int ii, jj;
    bool  sign;
    

    if (this->nbit1 < 0){
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          x = xs(ii, jj);
          xa = (x < 0) ? (-x) : x;
          xx = x*x;
          //
          if (xa<2) {
            y = xx * (xx * (FPTYPE)0.0625 - xa * (FPTYPE)0.25) + xa; 
          } else {
            y = 1;
          }
          //
          ys(ii, jj) = (x<0) ? (-y) : y;
        }
      }
    }
    //
    else {
      prec = 1 << this->nbit1;

      if (this->isround)
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          x = xs(ii, jj);
          xa = (x<0) ? (-x) : x;
          xx = x * x;
          xx = round(xx * prec) / prec;
          //
          if (xa<2) {
            y = xx * (xx * (FPTYPE)0.0625 - xa * (FPTYPE)0.25) + xa;
          } else {
            y = 1;
          }
          //
          y = round(y * prec) / prec;
          ys(ii, jj) = (x<0) ? (-y) : y;
        }
      }
      //
      else
      for(ii=0; ii<D1; ii++){
        for(jj=0; jj<D2; jj++){
          x = xs(ii, jj);
          xa = (x<0) ? (-x) : x;
          xx = x * x;
          xx = floor(xx * prec) / prec;
          //
          if (xa<2) {
            y = xx * (xx * (FPTYPE)0.0625 - xa * (FPTYPE)0.25) + xa;
          } else {
            y = 1;
          }
          //
          y = floor(y * prec) / prec;
          ys(ii, jj) = (x<0) ? (-y) : y;
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
    Name("Tanh4Nvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    Tanh4NvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);                  
REGISTER_CPU(double);

