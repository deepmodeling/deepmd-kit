#include "paddle/extension.h"
#include "prod_force.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")

template <typename data_t>
void ProdForceSeAOpForwardCPUKernel(int nloc,
                                    int nall,
                                    int nframes,
                                    int ndescrpt,
                                    int nnei,
                                    data_t* p_force,
                                    const data_t* p_net_deriv,
                                    const data_t* p_in_deriv,
                                    const int* p_nlist) {
  for (int kk = 0; kk < nframes; ++kk) {
    data_t* force = p_force + kk * nall * 3;
    const data_t* net_deriv = p_net_deriv + kk * nloc * ndescrpt;
    const data_t* in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
    const int* nlist = p_nlist + kk * nloc * nnei;
    deepmd::prod_force_a_cpu(force, net_deriv, in_deriv, nlist, nloc, nall,
                             nnei, 0);
  }
}

std::vector<paddle::Tensor> ProdForceSeAOpCPUForward(
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  CHECK_INPUT(net_deriv_tensor);
  CHECK_INPUT(in_deriv_tensor);
  CHECK_INPUT(nlist_tensor);
  CHECK_INPUT(natoms_tensor);

  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);

  PD_CHECK(natoms_tensor.shape()[0] >= 3,
           "number of atoms should be larger than (or equal to) 3");
  const int* natoms = natoms_tensor.data<int>();
  int nloc = natoms[0];
  int nall = natoms[1];
  int nframes = net_deriv_tensor.shape()[0];
  int ndescrpt = net_deriv_tensor.shape()[1] / nloc;
  int nnei = nlist_tensor.shape()[1] / nloc;

  PD_CHECK(nframes == in_deriv_tensor.shape()[0],
           "number of samples should match");
  PD_CHECK(nframes == nlist_tensor.shape()[0],
           "number of samples should match");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1],
           "number of descriptors should match");

  std::vector<int64_t> force_shape{nframes, 3 * nall};
  paddle::Tensor force_tensor = paddle::empty(
      force_shape, net_deriv_tensor.dtype(), net_deriv_tensor.place());

  assert(nframes == force_shape[0]);
  assert(nframes == net_deriv_tensor.shape()[0]);
  assert(nframes == in_deriv_tensor.shape()[0]);
  assert(nframes == nlist_tensor.shape()[0]);
  assert(nall * 3 == force_shape[1]);
  assert(nloc * ndescrpt == net_deriv_tensor.shape()[1]);
  assert(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1]);
  assert(nloc * nnei == nlist_tensor.shape()[1]);
  assert(nnei * 4 == ndescrpt);

  PD_DISPATCH_FLOATING_TYPES(
      net_deriv_tensor.type(), "prod_force_se_a_cpu_forward_kernel", ([&] {
        ProdForceSeAOpForwardCPUKernel<data_t>(
            nloc, nall, nframes, ndescrpt, nnei,
            force_tensor.mutable_data<data_t>(),
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            nlist_tensor.data<int>());
      }));

  return {force_tensor};
}

std::vector<paddle::Tensor> ProdForceSeAOpCUDAForward(
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel);

std::vector<paddle::Tensor> ProdForceSeAForward(
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  if (net_deriv_tensor.place() == paddle::GPUPlace()) {
    return ProdForceSeAOpCUDAForward(
        net_deriv_tensor, in_deriv_tensor, nlist_tensor,
        natoms_tensor.copy_to(paddle::CPUPlace(), false), n_a_sel, n_r_sel);
  } else if (net_deriv_tensor.place() == paddle::CPUPlace()) {
    return ProdForceSeAOpCPUForward(net_deriv_tensor, in_deriv_tensor,
                                    nlist_tensor, natoms_tensor, n_a_sel,
                                    n_r_sel);
  } else {
    PD_THROW("No Such kernel for ProdForceSeAForward.");
  }
}

std::vector<std::vector<int64_t>> ProdForceSeAInferShape(
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> nlist_shape,
    std::vector<int64_t> natoms_shape,
    const int& n_a_sel,
    const int& n_r_sel) {
  int64_t nall = /*natoms[1]*/ 192;
  int64_t nframes = net_deriv_shape[0];
  std::vector<int64_t> force_shape = {nframes, 3 * nall};
  return {force_shape};
}

std::vector<paddle::DataType> ProdForceSeAInferDtype(
    paddle::DataType net_deriv_dtype,
    paddle::DataType in_deriv_dtype,
    paddle::DataType nlist_dtype,
    paddle::DataType natoms_dtype) {
  return {net_deriv_dtype};
}

PD_BUILD_OP(prod_force_se_a)
    .Inputs({"net_deriv", "in_deriv", "nlist", "natoms"})
    .Outputs({"force"})
    .Attrs({"n_a_sel: int", "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(ProdForceSeAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ProdForceSeAInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ProdForceSeAInferDtype));
