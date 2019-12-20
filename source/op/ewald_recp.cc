#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <iostream>

#include "Ewald.h"

typedef double boxtensor_t ;

using namespace tensorflow;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE ;
#else 
typedef float  VALUETYPE ;
#endif

#ifdef HIGH_PREC
REGISTER_OP("EwaldRecp")
.Input("coord: double")
.Input("charge: double")
.Input("natoms: int32")
.Input("box: double")
.Attr("ewald_beta: float")
.Attr("ewald_h: float")
.Output("energy: double")
.Output("force: double")
.Output("virial: double");
#else
REGISTER_OP("EwaldRecp")
.Input("coord: float")
.Input("charge: float")
.Input("natoms: int32")
.Input("box: float")
.Attr("ewald_beta: float")
.Attr("ewald_h: float")
.Output("energy: float")
.Output("force: float")
.Output("virial: float");
#endif

class EwaldRecpOp : public OpKernel {
public:
  explicit EwaldRecpOp(OpKernelConstruction* context) : OpKernel(context) {
    float beta, spacing;
    OP_REQUIRES_OK(context, context->GetAttr("ewald_beta", &(beta)));
    OP_REQUIRES_OK(context, context->GetAttr("ewald_h", &(spacing)));
    ep.beta = beta;
    ep.spacing = spacing;
  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    int cc = 0;
    const Tensor& coord_tensor	= context->input(cc++);
    const Tensor& charge_tensor	= context->input(cc++);
    const Tensor& natoms_tensor	= context->input(cc++);
    const Tensor& box_tensor	= context->input(cc++);

    // set size of the sample
    OP_REQUIRES (context, (coord_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of coord should be 1"));
    OP_REQUIRES (context, (charge_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of charge should be 1"));
    OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) == 1),	errors::InvalidArgument ("size of natoms should be 1"));
    OP_REQUIRES (context, (box_tensor.shape().dims() == 1),		errors::InvalidArgument ("Dim of box should be 1"));
    auto natoms	= natoms_tensor.flat<int>();
    int nloc = natoms(0);
    int nsamples = coord_tensor.shape().dim_size(0) / (nloc * 3);

    // check the sizes
    OP_REQUIRES (context, (nsamples * nloc * 3 == coord_tensor.shape().dim_size(0)),	errors::InvalidArgument ("coord  number of samples should match"));
    OP_REQUIRES (context, (nsamples * nloc * 1 == charge_tensor.shape().dim_size(0)),	errors::InvalidArgument ("charge number of samples should match"));
    OP_REQUIRES (context, (nsamples * 9 == box_tensor.shape().dim_size(0)),		errors::InvalidArgument ("box    number of samples should match"));

    // Create an output tensor
    TensorShape energy_shape ;
    energy_shape.AddDim (nsamples);
    TensorShape force_shape ;
    force_shape.AddDim (nsamples);
    force_shape.AddDim (nloc * 3);
    TensorShape virial_shape ;
    virial_shape.AddDim (nsamples);
    virial_shape.AddDim (9);

    cc = 0;
    Tensor* energy_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(cc++, energy_shape, &energy_tensor));
    Tensor* force_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(cc++, force_shape, &force_tensor));
    Tensor* virial_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(cc++, virial_shape, &virial_tensor));
    
    auto coord	= coord_tensor	.flat<VALUETYPE>();
    auto charge	= charge_tensor	.flat<VALUETYPE>();
    auto box	= box_tensor	.flat<VALUETYPE>();
    auto energy	= energy_tensor	->flat<VALUETYPE>();
    auto force	= force_tensor	->matrix<VALUETYPE>();
    auto virial	= virial_tensor	->matrix<VALUETYPE>();

    for (int kk = 0; kk < nsamples; ++kk){
      int box_iter = kk * 9;
      int coord_iter = kk * nloc * 3;
      int charge_iter = kk * nloc;
      // set region
      boxtensor_t boxt [9] = {0};
      for (int dd = 0; dd < 9; ++dd) {
	boxt[dd] = box(box_iter + dd);
      }
      SimulationRegion<boxtensor_t > region;
      region.reinitBox (boxt);

      // set & normalize coord
      vector<boxtensor_t > d_coord3_ (nloc*3);
      for (int ii = 0; ii < nloc; ++ii){
	for (int dd = 0; dd < 3; ++dd){
	  d_coord3_[ii*3+dd] = coord(coord_iter + ii*3+dd);
	}
	double inter[3];
	region.phys2Inter (inter, &d_coord3_[3*ii]);
	for (int dd = 0; dd < 3; ++dd){
	  if      (inter[dd] < 0 ) inter[dd] += 1.;
	  else if (inter[dd] >= 1) inter[dd] -= 1.;
	}
      }
      vector<VALUETYPE > d_coord3 (nloc*3);
      for (int ii = 0; ii < nloc * 3; ++ii) {
	d_coord3[ii] = d_coord3_[ii];
      }

      // set charge
      vector<VALUETYPE > d_charge (nloc);
      for (int ii = 0; ii < nloc; ++ii) d_charge[ii] = charge(charge_iter + ii);

      // prepare outputs vectors
      VALUETYPE d_ener;
      vector<VALUETYPE> d_force(nloc*3);
      vector<VALUETYPE> d_virial(9);

      // compute
      EwaldReciprocal(d_ener, d_force, d_virial, d_coord3, d_charge, region, ep);

      // copy output
      energy(kk) = d_ener;
      for (int ii = 0; ii < nloc * 3; ++ii){
	force(kk, ii) = d_force[ii];
      }
      for (int ii = 0; ii < 9; ++ii){
	virial(kk, ii) = d_virial[ii];
      }
    }
  }
private:
  EwaldParameters<VALUETYPE> ep;
};

REGISTER_KERNEL_BUILDER(Name("EwaldRecp").Device(DEVICE_CPU), EwaldRecpOp);

