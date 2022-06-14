
//
// --------------------------------------------------------------------
/*

# Function
x = xk+dx
y = vk+dvk*dx

build a mapping table V, use the X as index to select value Y

# Parameters
x index
v mapping table
dv mapping table of slope
grad_v mapping table of 1st order derivative
grad_dv  mapping table of slope of 1st order derivative
prec precision
nbit number of bits
y output
*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include "custom_op.h"

using namespace tensorflow;

//- register the operator
// prec = 2^n, so it doesn't need to match `T`
REGISTER_OP("MapNvnmd")
  .Attr("T: {float, double} = DT_DOUBLE")
  .Input("x: T")
  .Input("v: T")
  .Input("dv: T")
  .Input("grad_v: T")
  .Input("grad_dv: T")
  .Attr("prec: float")
  .Attr("nbit: int")
  .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class MapNvnmdOp : public OpKernel {
public:

  /// Constructor.
  explicit MapNvnmdOp(OpKernelConstruction* context) : OpKernel(context) {	  
    OP_REQUIRES_OK(context, context->GetAttr("prec", &prec));
    div_prec = 1.0 / prec;
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
    DCHECK_EQ(5, context->num_inputs());
    
    //- 2.get tensor
    const Tensor& X = context->input(0);
    const Tensor& V = context->input(1);
    const Tensor& DV = context->input(2);
    
    //- 3. get shape and check
    const TensorShape& shX = X.shape();
    const TensorShape& shV = V.shape();
    const TensorShape& shDV = DV.shape();
    
    int D1 = shX.dim_size(0);
    int D2 = shX.dim_size(1);
    int D3 = shV.dim_size(0);
    int D4 = shV.dim_size(1);
	
    DCHECK_EQ(shX.dims(), 2);
    DCHECK_EQ(shV.dims(), 2);
    
    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */
    
    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(D1);
    shY.AddDim(D2*D4);
    Tensor* Y = NULL;
    
    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &Y));
    auto x = X.matrix<FPTYPE>();
    auto v = V.matrix<FPTYPE>();
    auto dv = DV.matrix<FPTYPE>();
    auto y = Y->matrix<FPTYPE>();

    int ii, jj, kk, jk, n;
    FPTYPE dx;
    for(ii=0; ii<D1; ii++){
      n = floor(x(ii, 0) * div_prec);
      dx = x(ii, 0) - n * prec;
      //check
      if (n < 0)  {
        std::cerr<<"ERROR: index is smaller than 0 \n"; 
        n = 0;
      }
      if (n > D3) {
        std::cerr<<"ERROR: index is bigger  than range \n";
        n = 0;
      }
      n = (n == D3) ? 0 : n;
      //map
      for(kk=0; kk<D4; kk++){
        y(ii, kk) = v(n, kk) + dv(n, kk) * dx;
      }
    }

  }
//- define the private variable for calculation
private:
float prec, div_prec;
};

#define REGISTER_CPU(T) \
REGISTER_KERNEL_BUILDER( \
    Name("MapNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
    MapNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);                  
REGISTER_CPU(double);



