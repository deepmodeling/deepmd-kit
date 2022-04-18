#include <assert.h>
#include "prod_force.h"
#include "prod_force_grad.h"
//#include "paddle/extension.h"
#include "paddle/include/experimental/ext_all.h"

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")
#define CHECK_INPUT_READY(x) PD_CHECK(x.is_initialized(), #x " must be initialized before usage.")
#define CHECK_INPUT_DIM(x, value) PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")



#ifdef PADDLE_WITH_CUDA
std::vector<paddle::Tensor> PdProdForceSeAOpCUDAForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel);
#endif

template <typename data_t>
void PdProdForceSeAOpForwardCPUKernel(
    int nloc, int nall, int nframes, int ndescrpt, int nnei,
    data_t* p_force, const data_t* p_net_deriv, const data_t* p_in_deriv, const int* p_nlist){

        for(int kk = 0; kk < nframes; ++kk){
            data_t * force = p_force + kk * nall * 3;
            const data_t * net_deriv = p_net_deriv + kk * nloc * ndescrpt;
            const data_t * in_deriv = p_in_deriv + kk * nloc * ndescrpt * 3;
            const int * nlist = p_nlist + kk * nloc * nnei;      
            deepmd::prod_force_a_cpu(    
                force, 
                net_deriv, in_deriv, nlist, nloc, nall, nnei);
        }
    }

std::vector<paddle::Tensor> PdProdForceSeAOpCPUForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel
){
    CHECK_INPUT(net_deriv_tensor);
    CHECK_INPUT(in_deriv_tensor);
    CHECK_INPUT(nlist_tensor);
    CHECK_INPUT(natoms_tensor);

    CHECK_INPUT_DIM(net_deriv_tensor, 2);
    CHECK_INPUT_DIM(in_deriv_tensor, 2);
    CHECK_INPUT_DIM(nlist_tensor, 2);
    CHECK_INPUT_DIM(natoms_tensor, 1);

    PD_CHECK(natoms_tensor.shape()[0] >= 3, "number of atoms should be larger than (or equal to) 3");
    // TODO: This code should be removed once cuda issue fixed.
    const int* natoms = nullptr;
    natoms = natoms_tensor.data<int>();
    int nloc = natoms[0];
    int nall = natoms[1];
    int nframes = net_deriv_tensor.shape()[0];
    int ndescrpt = net_deriv_tensor.shape()[1] / nloc;
    int nnei = nlist_tensor.shape()[1] / nloc;

    PD_CHECK(nframes == in_deriv_tensor.shape()[0], "number of samples should match");
    PD_CHECK(nframes == nlist_tensor.shape()[0],"number of samples should match");
    PD_CHECK(nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1], "number of descriptors should match");

    std::vector<int64_t> force_shape {nframes, 3 * nall};
    paddle::Tensor force_tensor = paddle::Tensor(paddle::PlaceType::kCPU, force_shape);

    assert (nframes == force_shape[0]);
    assert (nframes == net_deriv_tensor.shape()[0]);
    assert (nframes == in_deriv_tensor.shape()[0]);
    assert (nframes == nlist_tensor.shape()[0]);
    assert (nall * 3 == force_shape[1]);
    assert (nloc * ndescrpt == net_deriv_tensor.shape()[1]);
    assert (nloc * ndescrpt * 3 == in_deriv_tensor.shape()[1]);
    assert (nloc * nnei == nlist_tensor.shape()[1]);
    assert (nnei * 4 == ndescrpt);

    PD_DISPATCH_FLOATING_TYPES(
        net_deriv_tensor.type(), "pd_prod_force_se_a_cpu_forward_kernel", ([&] {
        PdProdForceSeAOpForwardCPUKernel<data_t>(
            nloc, nall, nframes, ndescrpt, nnei, 
            force_tensor.mutable_data<data_t>(), net_deriv_tensor.data<data_t>(), 
            in_deriv_tensor.data<data_t>(), nlist_tensor.data<int>());
        }));

    return {force_tensor};
}

template <typename data_t>
void PdProdForceSeAOpCPUBackwardKernel(
    int nloc, int nframes, int ndescrpt, int nnei,
    const data_t* grad, const data_t* net_deriv, 
    const data_t* in_deriv, const int* nlist,
    data_t* grad_net){

#pragma omp parallel for
    for (int kk = 0; kk < nframes; ++kk){

      int grad_iter	= kk * nloc * 3;
      int in_iter	= kk * nloc * ndescrpt * 3;
      int nlist_iter	= kk * nloc * nnei;
      int grad_net_iter	= kk * nloc * ndescrpt;


      deepmd::prod_force_grad_a_cpu(
	&grad_net[grad_net_iter],
	&grad[grad_iter],
	&in_deriv[in_iter],
	&nlist[nlist_iter],
	nloc, 
	nnei);
    }
}

std::vector<paddle::Tensor> PdProdForceSeAOpCPUBackward(
const paddle::Tensor& grad_tensor,
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel
){
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

    PD_CHECK(natoms_shape[0] >= 3, "number of atoms should be larger than (or equal to) 3");
    const int* natoms = nullptr;
    natoms = natoms_tensor.data<int>();
    int nloc = natoms[0];
    int nframes = net_deriv_shape[0];
    int ndescrpt = net_deriv_shape[1] / nloc;
    int nnei = nlist_shape[1] / nloc;

    PD_CHECK(nframes == grad_shape[0], "number of frames should match");
    PD_CHECK(nframes == in_deriv_shape[0], "number of samples should match");
    PD_CHECK(nframes == nlist_shape[0],"number of samples should match");
    PD_CHECK((nloc * 3) == grad_shape[1], "input grad shape should be 3 x natoms");
    PD_CHECK(nloc * ndescrpt * 3 == in_deriv_shape[1], "number of descriptors should match");
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
            nlist_tensor.data<int>(),
            grad_net_tensor.mutable_data<data_t>());
    }));

    return {grad_net_tensor};
}

std::vector<paddle::Tensor> PdProdForceSeAForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel){
    if(net_deriv_tensor.place() == paddle::PlaceType::kCPU){
        return PdProdForceSeAOpCPUForward(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
#ifdef PADDLE_WITH_CUDA
    }else if(net_deriv_tensor.place() == paddle::PlaceType::kGPU){
        return PdProdForceSeAOpCUDAForward(net_deriv_tensor, in_deriv_tensor, nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
#endif
    }else{
        PD_THROW("No Such kernel for PdFrodForceSeAForward!");
    }
}

std::vector<paddle::Tensor> PdProdForceSeABackward(
const paddle::Tensor& grad_tensor,
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel){
    return PdProdForceSeAOpCPUBackward(
            grad_tensor, net_deriv_tensor, in_deriv_tensor,
            nlist_tensor, natoms_tensor, n_a_sel, n_r_sel);
}
std::vector<std::vector<int64_t>> PdProdForceSeAOpForwardInferShape(
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> nlist_shape,
    std::vector<int64_t> natoms_shape
){
    std::vector<int64_t> force_shape{net_deriv_shape[0], -1};
    return {force_shape};
}

std::vector<std::vector<int64_t>> PdProdForceSeAOpBackwardInferShape(
    std::vector<int64_t> grad_shape,
    std::vector<int64_t> net_deriv_shape,
    std::vector<int64_t> in_deriv_shape,
    std::vector<int64_t> nlist_shape,
    std::vector<int64_t> natoms_shape
){
    return {net_deriv_shape};
}

std::vector<paddle::DataType> PdProdForceSeAOpForwardInferDtype(
  paddle::DataType net_deriv_dtype,
  paddle::DataType in_deriv_dtype,
  paddle::DataType nlist_dtype,
  paddle::DataType natoms_dtype){
      return {net_deriv_dtype};
}

std::vector<paddle::DataType> PdProdForceSeAOpBackwardInferDtype(
  paddle::DataType grad_dtype,
  paddle::DataType net_deriv_dtype,
  paddle::DataType in_deriv_dtype,
  paddle::DataType nlist_dtype,
  paddle::DataType natoms_dtype){
      return {net_deriv_dtype};
}
PD_BUILD_OP(prod_force_se_a)
    .Inputs({"net_deriv", "in_deriv", "nlist", "natoms"})
    .Outputs({"force"})
    .Attrs({
    "n_a_sel : int",
    "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdForceSeAForward))
    .SetInferShapeFn(PD_INFER_SHAPE(PdProdForceSeAOpForwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PdProdForceSeAOpForwardInferDtype));

PD_BUILD_GRAD_OP(prod_force_se_a)
    .Inputs({paddle::Grad("force"), "net_deriv", "in_deriv", "nlist", "natoms"})
    .Outputs({paddle::Grad("net_deriv")})
    .Attrs({
    "n_a_sel : int",
    "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdForceSeABackward));

// just for test 
PD_BUILD_OP(prod_force_se_a_grad2)
    .Inputs({paddle::Grad("force"), "net_deriv", "in_deriv", "nlist", "natoms"})
    .Outputs({paddle::Grad("net_deriv")})
    .Attrs({
    "n_a_sel : int",
    "n_r_sel: int"})
    .SetKernelFn(PD_KERNEL(PdProdForceSeABackward))
    .SetInferShapeFn(PD_INFER_SHAPE(PdProdForceSeAOpBackwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(PdProdForceSeAOpBackwardInferDtype));
