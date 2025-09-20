// SPDX-License-Identifier: LGPL-3.0-or-later
#include "custom_op.h"

REGISTER_OP("ProdForceSeAMaskGrad")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("grad: T")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("mask: int32")
    .Input("nlist: int32")
    .Attr("total_atom_num: int")
    .Output("grad_net: T");

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename FPTYPE>
class ProdForceSeAMaskGradOp : public OpKernel {
 public:
  explicit ProdForceSeAMaskGradOp(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("total_atom_num", &total_atom_num));
  }

  void Compute(OpKernelContext *context) override {
    deepmd::safe_compute(
        context, [this](OpKernelContext *context) { this->_Compute(context); });
  }

  void _Compute(OpKernelContext *context) {
    // Grab the input tensor
    const Tensor &grad_tensor = context->input(0);
    const Tensor &net_deriv_tensor = context->input(1);
    const Tensor &in_deriv_tensor = context->input(2);
    const Tensor &mask_tensor = context->input(3);
    const Tensor &nlist_tensor = context->input(4);

    // set size of the sample
    TensorShape grad_shape = grad_tensor.shape();
    TensorShape net_deriv_shape = net_deriv_tensor.shape();
    TensorShape in_deriv_shape = in_deriv_tensor.shape();
    TensorShape mask_shape = mask_tensor.shape();
    TensorShape nlist_shape = nlist_tensor.shape();

    OP_REQUIRES(context, (grad_shape.dims() == 2),
                errors::InvalidArgument("Dim of grad should be 2"));
    OP_REQUIRES(context, (net_deriv_shape.dims() == 2),
                errors::InvalidArgument("Dim of net deriv should be 2"));
    OP_REQUIRES(context, (in_deriv_shape.dims() == 2),
                errors::InvalidArgument("Dim of input deriv should be 2"));
    OP_REQUIRES(context, (mask_shape.dims() == 2),
                errors::InvalidArgument("Dim of mask should be 2"));
    OP_REQUIRES(context, (nlist_shape.dims() == 2),
                errors::InvalidArgument("Dim of nlist should be 2"));

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = total_atom_num;
    int ndescrpt = nloc > 0 ? net_deriv_tensor.shape().dim_size(1) / nloc : 0;
    int nnei = total_atom_num;

    // check the sizes
    OP_REQUIRES(context, (nframes == grad_shape.dim_size(0)),
                errors::InvalidArgument("number of frames should match"));
    OP_REQUIRES(context, (nframes == in_deriv_shape.dim_size(0)),
                errors::InvalidArgument("number of frames should match"));
    OP_REQUIRES(context, (nframes == nlist_shape.dim_size(0)),
                errors::InvalidArgument("number of frames should match"));
    OP_REQUIRES(context, (nframes == mask_shape.dim_size(0)),
                errors::InvalidArgument("number of frames should match"));

    OP_REQUIRES(
        context, (nloc * 3 == grad_shape.dim_size(1)),
        errors::InvalidArgument("input grad shape should be 3 x natoms"));
    OP_REQUIRES(context,
                (static_cast<int64_t>(nloc) * ndescrpt * 3 ==
                 in_deriv_shape.dim_size(1)),
                errors::InvalidArgument("number of descriptors should match"));

    // Create an output tensor
    TensorShape grad_net_shape;
    grad_net_shape.AddDim(nframes);
    grad_net_shape.AddDim(static_cast<int64_t>(nloc) * ndescrpt);

    // allocate the output tensor
    Tensor *grad_net_tensor = NULL;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, grad_net_shape, &grad_net_tensor));

    // flat the tensors
    auto grad = grad_tensor.flat<FPTYPE>();
    auto net_deriv = net_deriv_tensor.flat<FPTYPE>();
    auto in_deriv = in_deriv_tensor.flat<FPTYPE>();
    auto mask = mask_tensor.flat<int>();
    auto nlist = nlist_tensor.flat<int>();
    auto grad_net = grad_net_tensor->flat<FPTYPE>();

    // loop over frames
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk) {
      int grad_iter = kk * nloc * 3;
      int net_iter = kk * nloc * ndescrpt;
      int in_iter = kk * nloc * ndescrpt * 3;
      int nlist_iter = kk * nloc * nnei;
      int mask_iter = kk * nloc;
      int grad_net_iter = kk * nloc * ndescrpt;

      // reset the frame to 0
      for (int ii = 0; ii < nloc; ++ii) {
        for (int aa = 0; aa < ndescrpt; ++aa) {
          grad_net(grad_net_iter + ii * ndescrpt + aa) = 0.0;
        }
      }

      // compute grad of one frame
      for (int ii = 0; ii < nloc; ++ii) {
        int i_idx = ii;

        // deriv wrt center atom
        for (int aa = 0; aa < ndescrpt; ++aa) {
          for (int dd = 0; dd < 3; ++dd) {
            grad_net(grad_net_iter + i_idx * ndescrpt + aa) -=
                grad(grad_iter + i_idx * 3 + dd) *
                in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + dd);
          }
        }

        // loop over neighbors
        for (int jj = 0; jj < nnei; ++jj) {
          int j_idx = nlist(nlist_iter + i_idx * nnei + jj);
          // Check if atom j_idx is virtual or if the i_idx is virtual.
          if (j_idx == i_idx || j_idx < 0) {
            continue;
          }
          /*
if (j_idx > nloc)
  j_idx = j_idx % nloc;
if (j_idx < 0)
  continue;
*/
          int aa_start, aa_end;
          aa_start = jj * 4;
          aa_end = jj * 4 + 4;
          // make_descript_range (aa_start, aa_end, jj);
          for (int aa = aa_start; aa < aa_end; ++aa) {
            for (int dd = 0; dd < 3; ++dd) {
              grad_net(grad_net_iter + i_idx * ndescrpt + aa) +=
                  grad(grad_iter + j_idx * 3 + dd) *
                  in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + dd);
            }
          }
        }
      }
    }
  }

 private:
  int total_atom_num;
};

#define REGISTER_CPU(T)                                                       \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("ProdForceSeAMaskGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ProdForceSeAMaskGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
