#include <assert.h>
#include "prod_virial.h"
#include "prod_virial_grad.h"
#ifdef ON_INFER
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif


#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")
#define CHECK_INPUT_READY(x) PD_CHECK(x.is_initialized(), #x " must be initialized before usage.")
#define CHECK_INPUT_DIM(x, value) PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")


#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> PdProdVirialSeAOpCUDAForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& rij_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel);
#endif

template <typename data_t>
void PdProdVirialSeAOpForwardCPUKernel(
    int nloc, int nall, int ndescrpt, int nnei, int nframes,
    data_t* p_virial, data_t* p_atom_virial, const data_t* p_net_deriv, 
    const data_t* p_in_deriv, const data_t* p_rij, const int* p_nlist){

    for(int kk = 0; kk < nframes; ++kk){
      data_t * virial = p_virial + kk * 9;
      data_t * atom_virial = p_atom_virial + kk * nall * 9;
      const data_t * net_deriv = p_net_deriv + kk * nloc * ndescrpt;
      const data_t * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
      const data_t * rij = p_rij + kk * nloc * nnei * 3;
      const int * nlist = p_nlist + kk * nloc * nnei;
      deepmd::prod_virial_a_cpu(    
          virial, atom_virial,
          net_deriv, in_deriv, rij, nlist, nloc, nall, nnei);
    }
}

std::vector<paddle::Tensor> PdProdVirialSeAOpCPUForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& rij_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel
){
    CHECK_INPUT_READY(net_deriv_tensor);
    CHECK_INPUT_READY(in_deriv_tensor);
    CHECK_INPUT_READY(rij_tensor);
    CHECK_INPUT_READY(nlist_tensor);
    CHECK_INPUT_READY(natoms_tensor);

    CHECK_INPUT_DIM(net_deriv_tensor, 2);
    CHECK_INPUT_DIM(in_deriv_tensor, 2);
    CHECK_INPUT_DIM(rij_tensor, 2);
    CHECK_INPUT_DIM(nlist_tensor, 2);
    CHECK_INPUT_DIM(natoms_tensor, 1);

    PD_CHECK(natoms_tensor.shape()[0] >= 3, "number of atoms should be larger than (or equal to) 3");
    // TODO:(jiabin) This code should be removed when virial cuda kernel fixed.
    const int* natoms = nullptr;
    natoms = natoms_tensor.data<int>();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nnei = nlist_tensor.shape()[1] / nloc;
    int nframes = net_deriv_tensor.shape()[0];
    int ndescrpt = net_deriv_tensor.shape()[1] / nloc;
    

    PD_CHECK(nframes == in_deriv_tensor.shape()[0], "number of samples should match");
    PD_CHECK(nframes == rij_tensor.shape()[0], "number of samples should match");
    PD_CHECK(nframes == nlist_tensor.shape()[0],"number of samples should match");
    PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1], "number of descriptors should match");
    PD_CHECK((nloc * nnei * 3) == rij_tensor.shape()[1], "dim of rij should be nnei * 3");

    std::vector<int64_t> virial_shape {nframes, 9};
    std::vector<int64_t> atom_virial_shape {nframes, 9 * nall};
    paddle::Tensor virial_tensor = paddle::Tensor(paddle::PlaceType::kCPU, virial_shape);
    paddle::Tensor atom_virial_tensor = paddle::Tensor(paddle::PlaceType::kCPU, atom_virial_shape);

    PD_DISPATCH_FLOATING_TYPES(
    net_deriv_tensor.type(), "pd_prod_virial_se_a_cpu_forward_kernel", ([&] {
        PdProdVirialSeAOpForwardCPUKernel<data_t>(
            nloc, nall, ndescrpt, nnei, nframes,
            virial_tensor.mutable_data<data_t>(), atom_virial_tensor.mutable_data<data_t>(), 
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            rij_tensor.data<data_t>(), nlist_tensor.data<int>());
    }));

    return {virial_tensor, atom_virial_tensor};
}

template <typename data_t>
void PdProdForceSeAOpCPUBackwardKernel(
    int nloc, int nframes, int ndescrpt, int nnei,
    const data_t* grad, const data_t* net_deriv, 
    const data_t* in_deriv, const data_t* rij, const int* nlist,
    data_t* grad_net){

#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){

      int grad_iter	= kk * 9;
      int in_iter	= kk * nloc * ndescrpt * 3;
      int rij_iter	= kk * nloc * nnei * 3;
      int nlist_iter	= kk * nloc * nnei;
      int grad_net_iter	= kk * nloc * ndescrpt;

      deepmd::prod_virial_grad_a_cpu(
	  &grad_net[grad_net_iter],
	  &grad[grad_iter],
	  &in_deriv[in_iter],
	  &rij[rij_iter],
	  &nlist[nlist_iter],
	  nloc,
	  nnei);
    }
}

std::vector<paddle::Tensor> PdProdVirialSeAOpCPUBackward(
const paddle::Tensor& grad_tensor,
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& rij_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel
){
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
    auto natoms_shape = natoms_tensor.shape();

    CHECK_INPUT_DIM(grad_tensor, 2);
    CHECK_INPUT_DIM(net_deriv_tensor, 2);
    CHECK_INPUT_DIM(in_deriv_tensor, 2);
    CHECK_INPUT_DIM(rij_tensor, 2);
    CHECK_INPUT_DIM(nlist_tensor, 2);
    CHECK_INPUT_DIM(natoms_tensor, 1);

    PD_CHECK(natoms_shape[0] >= 3, "number of atoms should be larger than (or equal to) 3");
    
    const int* natoms = nullptr;
    natoms = natoms_tensor.data<int>();
    int nframes = net_deriv_shape[0];
    int nloc = natoms[0];
    int ndescrpt = net_deriv_shape[1] / nloc;
    int nnei = nlist_shape[1] / nloc;

    PD_CHECK(nframes == grad_shape[0], "number of frames should match");
    PD_CHECK(nframes == in_deriv_shape[0], "number of samples should match");
    PD_CHECK(nframes == rij_shape[0], "number of frames should match");
    PD_CHECK(nframes == nlist_shape[0],"number of samples should match");
    PD_CHECK(9 == grad_shape[1], "input grad shape should be 3 x natoms");
    PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1], "number of descriptors should match");
    PD_CHECK(nloc * nnei * 3 == rij_shape[1], "dim of rij should be  nnei * 3");
    PD_CHECK(nnei == (n_a_sel + n_r_sel), "number of neighbors should match");

    std::vector<int64_t> grad_net_shape {nframes, nloc * ndescrpt};
    paddle::Tensor grad_net_tensor = paddle::Tensor(paddle::PlaceType::kCPU, grad_net_shape);

    PD_DISPATCH_FLOATING_TYPES(
    grad_tensor.type(), "pd_prod_force_se_a_cpu_backward_kernel", ([&] {
        PdProdForceSeAOpCPUBackwardKernel<data_t>(
            nloc, nframes, ndescrpt, nnei, 
            grad_tensor.data<data_t>(), 
            net_deriv_tensor.data<data_t>(), 
            in_deriv_tensor.data<data_t>(), 
            rij_tensor.data<data_t>(), nlist_tensor.data<int>(),
            grad_net_tensor.mutable_data<data_t>());
    }));

    return {grad_net_tensor};
}

std::vector<paddle::Tensor> PdProdVirialSeAForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& rij_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel){
    return PdProdVirialSeAOpCPUForward(net_deriv_tensor, in_deriv_tensor, rij_tensor, nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
    // TODO:(jiabin) Support this when virial cuda kernel fixed.
    // if(net_deriv_tensor.place() == paddle::PlaceType::kCPU){
    //     return PdProdVirialSeAOpCPUForward(net_deriv_tensor, in_deriv_tensor, rij_tensor, nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
    // }else if(net_deriv_tensor.place() == paddle::PlaceType::kGPU){
    //     return PdProdVirialSeAOpCUDAForward(net_deriv_tensor, in_deriv_tensor, rij_tensor, nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
    // }else{
    //     PD_THROW("No Such kernel for PdFrodForceSeAForward!");
    // }
}

std::vector<paddle::Tensor> PdProdVirialSeABackward(
const paddle::Tensor& grad_tensor,
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& rij_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel){
    return PdProdVirialSeAOpCPUBackward(
        grad_tensor, net_deriv_tensor, in_deriv_tensor,
        rij_tensor,
        nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);

}

std::vector<std::vector<int64_t>> PdProdVirialSeAOpForwardInferShape(
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> rij_shape,
    std::vector<int64_t> nlist_shape,
    std::vector<int64_t> natoms_shape
){
    std::vector<int64_t> virial_shape {net_deriv_shape[0], 9};
    std::vector<int64_t> atom_virial_shape {net_deriv_shape[0], -1};
    return {virial_shape, atom_virial_shape};
}

std::vector<std::vector<int64_t>> PdProdVirialSeAOpBackwardInferShape(
    std::vector<int64_t> grad_shape,
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> rij_shape,
    std::vector<int64_t> nlist_shape,
    std::vector<int64_t> natoms_shape
){
    return {net_deriv_shape};
}

std::vector<paddle::DataType> PdProdVirialSeAOpForwardInferDtype(
  paddle::DataType net_deriv_dtype,
  paddle::DataType in_deriv_dtype,
  paddle::DataType rij_dtype,
  paddle::DataType nlist_dtype,
  paddle::DataType natoms_dtype){
      return {net_deriv_dtype, net_deriv_dtype};
}

std::vector<paddle::DataType> PdProdVirialSeAOpBackwardInferDtype(
  paddle::DataType grad_type,
  paddle::DataType net_deriv_dtype,
  paddle::DataType in_deriv_dtype,
  paddle::DataType rij_dtype,
  paddle::DataType nlist_dtype,
  paddle::DataType natoms_dtype){
      return {net_deriv_dtype};
}

PD_BUILD_OP(prod_virial_se_a)
    .Inputs({"net_deriv", "in_deriv", "rij", "nlist", "natoms"})
    .Outputs({"virial", "atom_virial"})
    .Attrs({
    "n_a_sel : int",
    "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdVirialSeAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(PdProdVirialSeAOpForwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PdProdVirialSeAOpForwardInferDtype));

PD_BUILD_GRAD_OP(prod_virial_se_a)
    .Inputs({paddle::Grad("virial"), "net_deriv", "in_deriv", "rij", "nlist", "natoms"})
    .Outputs({paddle::Grad("net_deriv")})
    .Attrs({
    "n_a_sel : int",
    "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdVirialSeABackward));

// just for test

PD_BUILD_OP(prod_virial_se_a_grad2)
    .Inputs({paddle::Grad("virial"), "net_deriv", "in_deriv", "rij", "nlist", "natoms"})
    .Outputs({paddle::Grad("net_deriv")})
    .Attrs({
    "n_a_sel : int",
    "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdVirialSeABackward))
    .SetInferShapeFn(PD_INFER_SHAPE(PdProdVirialSeAOpBackwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PdProdVirialSeAOpBackwardInferDtype));
