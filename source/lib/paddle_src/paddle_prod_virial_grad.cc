#include "paddle/extension.h"
#include "prod_virial_grad.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a GPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT_READY(x) \
  PD_CHECK(x.initialized(), #x " must be initialized before usage.")

template <typename data_t>
void ProdVirialSeAOpCPUBackwardKernel(int nloc,
                                      int nframes,
                                      int ndescrpt,
                                      int nnei,
                                      const data_t* p_grad,
                                      const data_t* p_net_deriv,
                                      const data_t* p_in_deriv,
                                      const data_t* p_rij,
                                      const int* p_nlist,
                                      data_t* p_grad_net) {
  // #pragma omp parallel for
  //   for (int kk = 0; kk < nframes; ++kk) {
  //     int grad_iter = kk * 9;
  //     int in_iter = kk * nloc * ndescrpt * 3;
  //     int rij_iter = kk * nloc * nnei * 3;
  //     int nlist_iter = kk * nloc * nnei;
  //     int grad_net_iter = kk * nloc * ndescrpt;

  //     deepmd::prod_virial_grad_a_cpu(&grad_net[grad_net_iter],
  //     &grad[grad_iter],
  //                                    &in_deriv[in_iter], &rij[rij_iter],
  //                                    &nlist[nlist_iter], nloc, nnei);
  //   }

  for (int kk = 0; kk < nframes; ++kk) {
    data_t* grad_net = p_grad_net + kk * nloc * ndescrpt;
    const data_t* grad = p_grad + kk * 9;
    const data_t* in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
    const data_t* rij = p_rij + kk * nloc * nnei * 3;
    const int* nlist = p_nlist + kk * nloc * nnei;
    deepmd::prod_virial_grad_a_cpu(grad_net, grad, in_deriv, rij, nlist, nloc,
                                   nnei);
  }
}

std::vector<paddle::Tensor> ProdVirialSeAOpCPUBackward(
    const paddle::Tensor& grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& rij_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  CHECK_INPUT_READY(grad_tensor);
  CHECK_INPUT_READY(net_deriv_tensor);
  CHECK_INPUT_READY(in_deriv_tensor);
  CHECK_INPUT_READY(rij_tensor);
  CHECK_INPUT_READY(nlist_tensor);
  CHECK_INPUT_READY(natoms_tensor);

  auto grad_shape = grad_tensor.shape();
  auto net_deriv_shape = net_deriv_tensor.shape();
  auto in_deriv_shape = in_deriv_tensor.shape();
  auto rij_shape = rij_tensor.shape();
  auto nlist_shape = nlist_tensor.shape();

  CHECK_INPUT_DIM(grad_tensor, 2);
  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(rij_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);

  PD_CHECK(natoms_tensor.shape()[0] >= 3,
           "number of atoms should be larger than (or equal to) 3");

  const int* natoms = natoms_tensor.data<int>();

  int nframes = net_deriv_shape[0];
  int nloc = natoms[0];
  int ndescrpt = net_deriv_shape[1] / nloc;
  int nnei = nlist_shape[1] / nloc;

  PD_CHECK(nframes == grad_shape[0], "number of frames should match");
  PD_CHECK(nframes == in_deriv_shape[0], "number of samples should match");
  PD_CHECK(nframes == rij_shape[0], "number of frames should match");
  PD_CHECK(nframes == nlist_shape[0], "number of samples should match");
  PD_CHECK(9 == grad_shape[1], "input grad shape should be 3 x natoms");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1],
           "number of descriptors should match");
  PD_CHECK(nloc * nnei * 3 == rij_shape[1], "dim of rij should be  nnei * 3");
  PD_CHECK(nnei == (n_a_sel + n_r_sel), "number of neighbors should match");

  std::vector<int64_t> grad_net_shape{nframes, (int64_t)nloc * ndescrpt};
  paddle::Tensor grad_net_tensor =
      paddle::empty(grad_net_shape, grad_tensor.dtype(), grad_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      grad_tensor.type(), "prod_force_se_a_cpu_backward_kernel", ([&] {
        ProdVirialSeAOpCPUBackwardKernel<data_t>(
            nloc, nframes, ndescrpt, nnei, grad_tensor.data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            rij_tensor.data<data_t>(), nlist_tensor.data<int>(),
            grad_net_tensor.data<data_t>());
      }));
  return {grad_net_tensor};
}

std::vector<paddle::Tensor> ProdVirialSeAOpCUDABackward(
    const paddle::Tensor& virial_grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& rij_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel);

std::vector<paddle::Tensor> ProdVirialSeABackward(
    const paddle::Tensor& virial_grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& rij_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  if (virial_grad_tensor.is_gpu()) {
    return ProdVirialSeAOpCUDABackward(
        virial_grad_tensor, net_deriv_tensor, in_deriv_tensor, rij_tensor,
        nlist_tensor, natoms_tensor.copy_to(paddle::CPUPlace(), false), n_a_sel, n_r_sel);
  } else if (virial_grad_tensor.is_cpu()) {
    return ProdVirialSeAOpCPUBackward(virial_grad_tensor, net_deriv_tensor,
                                      in_deriv_tensor, rij_tensor, nlist_tensor,
                                      natoms_tensor, n_a_sel, n_r_sel);
  } else {
    PD_THROW("Unsupported device type for ProdVirialSeAForward");
  }
}

PD_BUILD_GRAD_OP(prod_virial_se_a)
    .Inputs({paddle::Grad("virial"), "net_deriv", "in_deriv", "rij", "nlist",
             "natoms"})
    .Outputs({paddle::Grad("net_deriv")})
    .Attrs({"n_a_sel: int", "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(ProdVirialSeABackward));
