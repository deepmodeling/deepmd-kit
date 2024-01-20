// SPDX-License-Identifier: LGPL-3.0-or-later
#include "custom_op.h"

REGISTER_OP("ProdForceSeAMask")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("net_deriv: T")
    .Input("in_deriv: T")
    .Input("mask: int32")
    .Input("nlist: int32")
    .Attr("total_atom_num: int")
    .Output("force: T");

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;

template <typename Device, typename FPTYPE>
class ProdForceSeAMaskOp : public OpKernel {
 public:
  explicit ProdForceSeAMaskOp(OpKernelConstruction *context)
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
    const Tensor &net_deriv_tensor = context->input(0);
    const Tensor &in_deriv_tensor = context->input(1);
    const Tensor &mask_tensor = context->input(2);
    const Tensor &nlist_tensor = context->input(3);

    // set size of the sample
    OP_REQUIRES(context, (net_deriv_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of net deriv should be 2"));
    OP_REQUIRES(context, (in_deriv_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of input deriv should be 2"));
    OP_REQUIRES(context, (mask_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of mask matrix should be 2"));
    OP_REQUIRES(context, (nlist_tensor.shape().dims() == 2),
                errors::InvalidArgument("Dim of nlist should be 2"));

    int nframes = net_deriv_tensor.shape().dim_size(0);
    int nloc = total_atom_num;
    int nall = total_atom_num;
    int ndescrpt = nall * 4;
    int nnei = nloc > 0 ? nlist_tensor.shape().dim_size(1) / nloc : 0;

    // check the sizes
    OP_REQUIRES(context, (nframes == in_deriv_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context, (nframes == nlist_tensor.shape().dim_size(0)),
                errors::InvalidArgument("number of samples should match"));
    OP_REQUIRES(context,
                (static_cast<int64_t>(nloc) * ndescrpt * 3 ==
                 in_deriv_tensor.shape().dim_size(1)),
                errors::InvalidArgument("number of descriptors should match"));

    // Create an output tensor
    TensorShape force_shape;
    force_shape.AddDim(nframes);
    force_shape.AddDim(3 * static_cast<int64_t>(nall));
    // std::cout << "forcesahpe " << force_shape.dim_size(0) << " " <<
    // force_shape.dim_size(1) << std::endl;
    Tensor *force_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, force_shape, &force_tensor));

    // flat the tensors
    auto net_deriv = net_deriv_tensor.flat<FPTYPE>();
    auto in_deriv = in_deriv_tensor.flat<FPTYPE>();
    auto nlist = nlist_tensor.flat<int>();
    auto mask = mask_tensor.flat<int>();
    auto force = force_tensor->flat<FPTYPE>();

// loop over samples
#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk) {
      int natoms = total_atom_num;
      int nloc = natoms;

      int force_iter = kk * natoms * 3;
      int net_iter = kk * natoms * ndescrpt;
      int in_iter = kk * natoms * ndescrpt * 3;
      int mask_iter = kk * natoms;
      int nlist_iter = kk * natoms * natoms;

      for (int ii = 0; ii < natoms; ii++) {
        int i_idx = ii;
        force(force_iter + i_idx * 3 + 0) = 0.0;
        force(force_iter + i_idx * 3 + 1) = 0.0;
        force(force_iter + i_idx * 3 + 2) = 0.0;
      }

      for (int ii = 0; ii < natoms; ii++) {
        int i_idx = ii;
        // Check if the atom ii is virtual particle or not.
        if (mask(mask_iter + i_idx) == 0) {
          continue;
        }
        // derivation with center atom. (x_j - x_i). x_i is the center atom.
        // Derivation with center atom.
        for (int aa = 0; aa < natoms * 4; ++aa) {
          force(force_iter + i_idx * 3 + 0) -=
              net_deriv(net_iter + i_idx * ndescrpt + aa) *
              in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0);
          force(force_iter + i_idx * 3 + 1) -=
              net_deriv(net_iter + i_idx * ndescrpt + aa) *
              in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1);
          force(force_iter + i_idx * 3 + 2) -=
              net_deriv(net_iter + i_idx * ndescrpt + aa) *
              in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2);
        }
        // Derivation with other atom.
        for (int jj = 0; jj < natoms; jj++) {
          // Get the neighbor index from nlist tensor.
          int j_idx = nlist(nlist_iter + i_idx * natoms + jj);

          if (j_idx == i_idx) {
            continue;
          }
          int aa_start, aa_end;
          aa_start = jj * 4;
          aa_end = jj * 4 + 4;
          for (int aa = aa_start; aa < aa_end; aa++) {
            force(force_iter + j_idx * 3 + 0) +=
                net_deriv(net_iter + i_idx * ndescrpt + aa) *
                in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0);
            force(force_iter + j_idx * 3 + 1) +=
                net_deriv(net_iter + i_idx * ndescrpt + aa) *
                in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1);
            force(force_iter + j_idx * 3 + 2) +=
                net_deriv(net_iter + i_idx * ndescrpt + aa) *
                in_deriv(in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2);
          }
        }
      }
    }
  }

 private:
  int total_atom_num;
};

#define REGISTER_CPU(T)                                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("ProdForceSeAMask").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ProdForceSeAMaskOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
