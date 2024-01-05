#include "paddle/extension.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT_READY(x) \
  PD_CHECK(x.initialized(), #x " must be initialized before usage.")

template <typename data_t>
void ProdForceSeAMaskOpForwardCPUKernel(int nframes,
                                        int total_atom_num,
                                        const data_t* net_deriv,
                                        const data_t* in_deriv,
                                        const int* mask,
                                        const int* nlist,
                                        data_t* force) {
  int nloc = total_atom_num;
  int nall = total_atom_num;
  int ndescrpt = nall * 4;

#pragma omp parallel for
  for (int kk = 0; kk < nframes; ++kk) {
    int force_iter = kk * nall * 3;
    int net_iter = kk * nall * ndescrpt;
    int in_iter = kk * nall * ndescrpt * 3;
    int mask_iter = kk * nall;
    int nlist_iter = kk * nall * nall;

    for (int ii = 0; ii < nall; ii++) {
      int i_idx = ii;
      force[force_iter + i_idx * 3 + 0] = 0.0;
      force[force_iter + i_idx * 3 + 1] = 0.0;
      force[force_iter + i_idx * 3 + 2] = 0.0;
    }

    for (int ii = 0; ii < nall; ii++) {
      int i_idx = ii;
      // Check if the atom ii is a virtual particle or not.
      if (mask[mask_iter + i_idx] == 0) {
        continue;
      }
      // Derivation with center atom.
      for (int aa = 0; aa < nall * 4; ++aa) {
        force[force_iter + i_idx * 3 + 0] -=
            net_deriv[net_iter + i_idx * ndescrpt + aa] *
            in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0];
        force[force_iter + i_idx * 3 + 1] -=
            net_deriv[net_iter + i_idx * ndescrpt + aa] *
            in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1];
        force[force_iter + i_idx * 3 + 2] -=
            net_deriv[net_iter + i_idx * ndescrpt + aa] *
            in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2];
      }
      // Derivation with other atoms.
      for (int jj = 0; jj < nall; jj++) {
        // Get the neighbor index from nlist tensor.
        int j_idx = nlist[nlist_iter + i_idx * nall + jj];

        if (j_idx == i_idx) {
          continue;
        }
        int aa_start, aa_end;
        aa_start = jj * 4;
        aa_end = jj * 4 + 4;
        for (int aa = aa_start; aa < aa_end; aa++) {
          force[force_iter + j_idx * 3 + 0] +=
              net_deriv[net_iter + i_idx * ndescrpt + aa] *
              in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + 0];
          force[force_iter + j_idx * 3 + 1] +=
              net_deriv[net_iter + i_idx * ndescrpt + aa] *
              in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + 1];
          force[force_iter + j_idx * 3 + 2] +=
              net_deriv[net_iter + i_idx * ndescrpt + aa] *
              in_deriv[in_iter + i_idx * ndescrpt * 3 + aa * 3 + 2];
        }
      }
    }
  }
}

std::vector<paddle::Tensor> ProdForceSeAMaskForward(
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& mask_tensor,
    const paddle::Tensor& nlist_tensor,
    int total_atom_num) {
  CHECK_INPUT(net_deriv_tensor);
  CHECK_INPUT(in_deriv_tensor);
  CHECK_INPUT(mask_tensor);
  CHECK_INPUT(nlist_tensor);

  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(mask_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);

  PD_CHECK(total_atom_num >= 3,
           "Number of atoms should be larger than (or equal to) 3");

  int nframes = net_deriv_tensor.shape()[0];
  int nloc = total_atom_num;
  int nall = total_atom_num;
  int ndescrpt = nall * 4;
  int nnei = nloc > 0 ? nlist_tensor.shape()[1] / nloc : 0;

  PD_CHECK(nframes == in_deriv_tensor.shape()[0],
           "Number of samples should match");
  PD_CHECK(nframes == nlist_tensor.shape()[0],
           "Number of samples should match");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1],
           "Number of descriptors should match");

  // Create output tensor
  std::vector<int64_t> force_shape{nframes, 3 * nall};
  paddle::Tensor force_tensor = paddle::empty(
      force_shape, net_deriv_tensor.dtype(), net_deriv_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      net_deriv_tensor.type(), "prod_force_se_a_mask_cpu_forward_kernel", ([&] {
        ProdForceSeAMaskOpForwardCPUKernel<data_t>(
            nframes, total_atom_num, net_deriv_tensor.data<data_t>(),
            in_deriv_tensor.data<data_t>(), mask_tensor.data<int>(),
            nlist_tensor.data<int>(), force_tensor.data<data_t>());
      }));

  return {force_tensor};
}

std::vector<std::vector<int64_t>> ProdForceSeAMaskOpInferShape(
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> mask_shape,
    std::vector<int64_t> nlist_shape,
    const int& total_atom_num) {
  int64_t nall = total_atom_num;
  int64_t nframes = net_deriv_shape[0];

  std::vector<int64_t> force_shape = {nframes, 3 * nall};

  return {force_shape};
}

std::vector<paddle::DataType> ProdForceSeAMaskOpInferDtype(
    paddle::DataType net_deriv_dtype,
    paddle::DataType in_deriv_dtype,
    paddle::DataType mask_dtype,
    paddle::DataType nlist_dtype) {
  return {net_deriv_dtype};
}

PD_BUILD_OP(prod_force_se_a_mask)
    .Inputs({"net_deriv", "in_deriv", "mask", "nlist"})
    .Attrs({"total_atom_num: int"})
    .Outputs({"force"})
    .SetKernelFn(PD_KERNEL(ProdForceSeAMaskForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ProdForceSeAMaskOpInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ProdForceSeAMaskOpInferDtype));
