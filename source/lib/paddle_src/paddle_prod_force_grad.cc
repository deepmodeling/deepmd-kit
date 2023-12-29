#include "paddle/extension.h"
#include "prod_force_grad.h"

#define CHECK_INPUT_READY(x) \
  PD_CHECK(x.initialized(), #x " must be initialized before usage.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT(x) PD_CHECK(x.is_gpu(), #x " must be a GPU Tensor.")

template <typename data_t>
void ProdForceSeAOpCPUBackwardKernel(int nloc,
                                     int nframes,
                                     int ndescrpt,
                                     int nnei,
                                     const data_t* grad,
                                     const data_t* net_deriv,
                                     const data_t* in_deriv,
                                     const int* nlist,
                                     data_t* grad_net) {
  // #pragma omp parallel for
  //   for (int kk = 0; kk < nframes; ++kk){
  //     int grad_iter	= kk * nloc * 3;
  //     int in_iter	= kk * nloc * ndescrpt * 3;
  //     int nlist_iter	= kk * nloc * nnei;
  //     int grad_net_iter	= kk * nloc * ndescrpt;

  //     deepmd::prod_force_grad_a_cpu(
  //       &grad_net[grad_net_iter],
  //       &grad[grad_iter],
  //       &in_deriv[in_iter],
  //       &nlist[nlist_iter],
  //       nloc,
  //       nnei
  //     );
  //   }

  for (int kk = 0; kk < nframes; ++kk) {
    data_t* p_grad_net = grad_net + kk * nloc * ndescrpt;
    const data_t* p_grad = grad + kk * nloc * 3;
    const data_t* p_in_deriv = in_deriv + kk * nloc * ndescrpt * 3;
    const int* p_nlist = nlist + kk * nloc * nnei;

    deepmd::prod_force_grad_a_cpu(p_grad_net, p_grad, p_in_deriv, p_nlist, nloc,
                                  nnei);
  }
}

std::vector<paddle::Tensor> ProdForceSeAOpCPUBackward(
    const paddle::Tensor& grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  CHECK_INPUT_READY(grad_tensor);
  CHECK_INPUT_READY(net_deriv_tensor);
  CHECK_INPUT_READY(in_deriv_tensor);
  CHECK_INPUT_READY(nlist_tensor);
  CHECK_INPUT_READY(natoms_tensor);

  auto grad_shape = grad_tensor.shape();
  auto net_deriv_shape = net_deriv_tensor.shape();
  auto in_deriv_shape = in_deriv_tensor.shape();
  auto nlist_shape = nlist_tensor.shape();
  auto natoms_shape = natoms_tensor.shape();

  CHECK_INPUT_DIM(grad_tensor, 2);
  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);

  PD_CHECK(natoms_shape[0] >= 3,
           "number of atoms should be larger than (or equal to) 3");
  const int* natoms = natoms_tensor.data<int>();

  int nframes = net_deriv_shape[0];
  int nloc = natoms[0];
  int ndescrpt = net_deriv_shape[1] / nloc;
  int nnei = nlist_shape[1] / nloc;

  PD_CHECK(nframes == grad_shape[0], "number of frames should match");
  PD_CHECK(nframes == in_deriv_shape[0], "number of samples should match");
  PD_CHECK(nframes == nlist_shape[0], "number of samples should match");
  PD_CHECK((nloc * 3) == grad_shape[1],
           "input grad shape should be 3 x natoms");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1],
           "number of descriptors should match");
  PD_CHECK(nnei == (n_a_sel + n_r_sel), "number of neighbors should match");

  std::vector<int64_t> grad_net_shape{nframes, (int64_t)nloc * ndescrpt};

  paddle::Tensor grad_net_tensor =
      paddle::empty(grad_net_shape, grad_tensor.dtype(), grad_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      grad_tensor.type(), "prod_force_se_a_cpu_backward_kernel", ([&] {
        ProdForceSeAOpCPUBackwardKernel<data_t>(
            nloc, nframes, ndescrpt, nnei, grad_tensor.data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            nlist_tensor.data<int>(), grad_net_tensor.data<data_t>());
      }));
  return {grad_net_tensor};
}

std::vector<paddle::Tensor> ProdForceSeAOpCUDABackward(
    const paddle::Tensor& force_grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel);

std::vector<paddle::Tensor> ProdForceSeABackward(
    const paddle::Tensor& force_grad_tensor,
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  if (net_deriv_tensor.place() == paddle::GPUPlace()) {
    return ProdForceSeAOpCUDABackward(
        force_grad_tensor, net_deriv_tensor, in_deriv_tensor, nlist_tensor,
        natoms_tensor.copy_to(paddle::CPUPlace(), false), n_a_sel, n_r_sel);
  } else if (net_deriv_tensor.place() == paddle::CPUPlace()) {
    return ProdForceSeAOpCPUBackward(force_grad_tensor, net_deriv_tensor,
                                     in_deriv_tensor, nlist_tensor,
                                     natoms_tensor, n_a_sel, n_r_sel);
  } else {
    PD_THROW("No Such kernel for ProdForceSeABackward.");
  }
}

PD_BUILD_GRAD_OP(prod_force_se_a)
    .Inputs({paddle::Grad("force"), "net_deriv", "in_deriv", "nlist", "natoms"})
    .Outputs({paddle::Grad("net_deriv")})
    .Attrs({"n_a_sel: int", "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(ProdForceSeABackward));
