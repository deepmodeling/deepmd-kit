#include "paddle/extension.h"
#include "prod_virial.h"

#define CHECK_INPUT(x) PD_CHECK(x.is_cpu(), #x " must be a CPU Tensor.")
#define CHECK_INPUT_DIM(x, value) \
  PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")
#define CHECK_INPUT_READY(x) \
  PD_CHECK(x.initialized(), #x " must be initialized before usage.")

template <typename data_t>
void ProdVirialSeAOpForwardCPUKernel(int nloc,
                                     int nall,
                                     int ndescrpt,
                                     int nnei,
                                     int nframes,
                                     data_t* p_virial,
                                     data_t* p_atom_virial,
                                     const data_t* p_net_deriv,
                                     const data_t* p_in_deriv,
                                     const data_t* p_rij,
                                     const int* p_nlist) {
  for (int kk = 0; kk < nframes; ++kk) {
    data_t* virial = p_virial + kk * 9;
    data_t* atom_virial = p_atom_virial + kk * nall * 9;
    const data_t* net_deriv = p_net_deriv + kk * nloc * ndescrpt;
    const data_t* in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
    const data_t* rij = p_rij + kk * nloc * nnei * 3;
    const int* nlist = p_nlist + kk * nloc * nnei;
    deepmd::prod_virial_a_cpu(virial, atom_virial, net_deriv, in_deriv, rij,
                              nlist, nloc, nall, nnei);
  }
}

std::vector<paddle::Tensor> ProdVirialSeAOpCPUForward(
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& rij_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  CHECK_INPUT(net_deriv_tensor);
  CHECK_INPUT(in_deriv_tensor);
  CHECK_INPUT(rij_tensor);
  CHECK_INPUT(nlist_tensor);
  CHECK_INPUT(natoms_tensor);

  CHECK_INPUT_DIM(net_deriv_tensor, 2);
  CHECK_INPUT_DIM(in_deriv_tensor, 2);
  CHECK_INPUT_DIM(rij_tensor, 2);
  CHECK_INPUT_DIM(nlist_tensor, 2);
  CHECK_INPUT_DIM(natoms_tensor, 1);

  PD_CHECK(natoms_tensor.shape()[0] >= 3,
           "number of atoms should be larger than (or equal to) 3");
  const int* natoms = natoms_tensor.data<int>();
  int nloc = natoms[0];
  int nall = natoms[1];
  int nnei = nlist_tensor.shape()[1] / nloc;
  int nframes = net_deriv_tensor.shape()[0];
  int ndescrpt = net_deriv_tensor.shape()[1] / nloc;

  PD_CHECK(nframes == in_deriv_tensor.shape()[0],
           "number of samples should match");
  PD_CHECK(nframes == rij_tensor.shape()[0], "number of samples should match");
  PD_CHECK(nframes == nlist_tensor.shape()[0],
           "number of samples should match");
  PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1],
           "number of descriptors should match");
  PD_CHECK((nloc * nnei * 3) == rij_tensor.shape()[1],
           "dim of rij should be nnei * 3");

  std::vector<int64_t> virial_shape{nframes, 9};
  std::vector<int64_t> atom_virial_shape{nframes, 9 * nall};
  paddle::Tensor virial_tensor = paddle::empty(
      virial_shape, net_deriv_tensor.dtype(), net_deriv_tensor.place());
  paddle::Tensor atom_virial_tensor = paddle::empty(
      atom_virial_shape, net_deriv_tensor.dtype(), net_deriv_tensor.place());

  PD_DISPATCH_FLOATING_TYPES(
      net_deriv_tensor.type(), "prod_virial_se_a_cpu_forward_kernel", ([&] {
        ProdVirialSeAOpForwardCPUKernel<data_t>(
            nloc, nall, ndescrpt, nnei, nframes, virial_tensor.data<data_t>(),
            atom_virial_tensor.data<data_t>(), net_deriv_tensor.data<data_t>(),
            in_deriv_tensor.data<data_t>(), rij_tensor.data<data_t>(),
            nlist_tensor.data<int>());
      }));

  return {virial_tensor, atom_virial_tensor};
}

std::vector<paddle::Tensor> ProdVirialSeAOpCUDAForward(
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& rij_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel);

std::vector<paddle::Tensor> ProdVirialSeAForward(
    const paddle::Tensor& net_deriv_tensor,
    const paddle::Tensor& in_deriv_tensor,
    const paddle::Tensor& rij_tensor,
    const paddle::Tensor& nlist_tensor,
    const paddle::Tensor& natoms_tensor,
    int n_a_sel,
    int n_r_sel) {
  if (net_deriv_tensor.is_gpu()) {
    return ProdVirialSeAOpCUDAForward(
        net_deriv_tensor, in_deriv_tensor, rij_tensor, nlist_tensor,
        natoms_tensor.copy_to(paddle::CPUPlace(), false), n_a_sel, n_r_sel);
  } else if (net_deriv_tensor.is_cpu()) {
    return ProdVirialSeAOpCPUForward(net_deriv_tensor, in_deriv_tensor,
                                     rij_tensor, nlist_tensor, natoms_tensor.copy_to(paddle::CPUPlace(), false),
                                     n_a_sel, n_r_sel);
  } else {
    PD_THROW("Unsupported device type for ProdVirialSeAForward");
  }
}

std::vector<std::vector<int64_t>> ProdVirialSeAInferShape(
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> rij_shape,
    std::vector<int64_t> nlist_shape,
    std::vector<int64_t> natoms_shape,
    const int& n_a_sel,
    const int& n_r_sel) {
  // int64_t nloc = /*natoms[0]*/ 192;
  int64_t nall = /*natoms[1]*/ 192;
  int64_t nframes = net_deriv_shape[0];

  std::vector<int64_t> virial_shape = {nframes, 9};
  std::vector<int64_t> atom_virial_shape = {nframes, 9 * nall};

  return {virial_shape, atom_virial_shape};
}

std::vector<paddle::DataType> ProdVirialSeAInferDtype(
    paddle::DataType net_deriv_dtype,
    paddle::DataType in_deriv_dtype,
    paddle::DataType rij_dtype,
    paddle::DataType nlist_dtype,
    paddle::DataType natoms_dtype) {
  return {net_deriv_dtype, net_deriv_dtype};
}

PD_BUILD_OP(prod_virial_se_a)
    .Inputs({"net_deriv", "in_deriv", "rij", "nlist", "natoms"})
    .Outputs({"virial", "atom_virial"})
    .Attrs({"n_a_sel: int", "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(ProdVirialSeAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(ProdVirialSeAInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(ProdVirialSeAInferDtype));
