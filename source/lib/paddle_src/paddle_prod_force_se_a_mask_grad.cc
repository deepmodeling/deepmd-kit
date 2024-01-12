#include "paddle/extension.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT_READY(x) \
  PD_CHECK(x.initialized(), #x " must be initialized before usage.")

template <typename data_t>
void ProdForceSeAMaskOpCPUBackwardKernel(int nloc,
                                         int nframes,
                                         int ndescrpt,
                                         int nnei,
                                         const data_t* grad,
                                         const data_t* net_deriv,
                                         const data_t* in_deriv,
                                         const int* mask,
                                         const int* nlist,
                                         data_t* grad_net) {
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
        grad_net[grad_net_iter + ii * ndescrpt + aa] = 0.0;
      }
    }

    // compute grad of one frame
    for (int ii = 0; ii < nloc; ++ii) {
      int i_idx = ii;

      // deriv wrt center atom
      for (int aa = 0; aa < ndescrpt; ++aa) {
        for (int dd = 0; dd < 3; ++dd) {
          grad_net[grad_net_iter + i_idx * ndescrpt + aa] -=
              grad[grad_iter + i_idx * 3 + dd] *
              in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + dd];
        }
      }

      // loop over neighbors
      for (int jj = 0; jj < nnei; ++jj) {
        int j_idx = nlist[nlist_iter + i_idx * nnei + jj];
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
            grad_net[grad_net_iter + i_idx * ndescrpt + aa] +=
                grad[grad_iter + j_idx * 3 + dd] *
                in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + dd];
          }
        }
      }
    }
  }
}

std::vector<paddle::Tensor> ProdForceSeAMaskOpCPUBackward(
    const paddle::Tensor& grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& mask_tensor,
    const paddle::Tensor& nlist_tensor,
    int total_atom_num) {
  CHECK_INPUT(grad_tensor);
  CHECK_INPUT(net_deriv_tensor);
  CHECK_INPUT(in_deriv_tensor);
  CHECK_INPUT(mask_tensor);
  CHECK_INPUT(nlist_tensor);

  CHECK_INPUT_DIM(grad_tensor, 2);
  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(mask_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);

  PD_CHECK(total_atom_num >= 3,
           "Number of atoms should be larger than (or equal to) 3");

  int nframes = net_deriv_tensor.shape()[0];
  int nloc = total_atom_num;
  int ndescrpt = nloc > 0 ? net_deriv_tensor.shape()[1] / nloc : 0;
  int nnei = total_atom_num;

  PD_CHECK(nframes == grad_tensor.shape()[0], "Number of frames should match");
  PD_CHECK(nframes == in_deriv_tensor.shape()[0],
           "Number of frames should match");
  PD_CHECK(nframes == nlist_tensor.shape()[0], "Number of frames should match");
  PD_CHECK(nframes == mask_tensor.shape()[0], "Number of frames should match");

  PD_CHECK(nloc * 3 == grad_tensor.shape()[1],
           "input grad shape should be 3 x natoms");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1],
           "Number of descriptors should match");

  // Create an output tensor
  std::vector<int64_t> grad_net_shape{nframes, nloc * ndescrpt};
  paddle::Tensor grad_net_tensor =
      paddle::empty(grad_net_shape, grad_tensor.dtype(), grad_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      grad_tensor.type(), "prod_force_se_a_mask_cpu_backward_kernel", ([&] {
        ProdForceSeAMaskOpCPUBackwardKernel<data_t>(
            nloc, nframes, ndescrpt, nnei, grad_tensor.data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            mask_tensor.data<int>(), nlist_tensor.data<int>(),
            grad_net_tensor.data<data_t>());
      }));

  return {grad_net_tensor};
}

std::vector<paddle::Tensor> ProdForceSeAMaskBackward(
    const paddle::Tensor& grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& mask_tensor,
    const paddle::Tensor& nlist_tensor,
    int total_atom_num) {
  return ProdForceSeAMaskOpCPUBackward(grad_tensor, net_deriv_tensor,
                                       in_deriv_tensor, mask_tensor,
                                       nlist_tensor, total_atom_num);
}

PD_BUILD_GRAD_OP(prod_force_se_a_mask)
    .Inputs({paddle::Grad("force"), "net_deriv", "in_deriv", "mask", "nlist"})
    .Attrs({"total_atom_num: int"})
    .Outputs({paddle::Grad("net_deriv")})
    .SetKernelFn(PD_KERNEL(ProdForceSeAMaskBackward));
