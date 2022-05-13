#define GOOGLE_CUDA 1
#include <assert.h>
#include "prod_virial.h"
#ifdef ON_INFER
#include "paddle/include/experimental/ext_all.h"
#else
#include "paddle/extension.h"
#endif

#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kGPU, #x " must be a GPU Tensor.")
#define CHECK_INPUT_DIM(x, value) PD_CHECK(x.shape().size() == value, #x "'s dim should be " #value ".")

template <typename data_t>
void PdProdVirialSeAOpForwardCUDAKernel(
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
      deepmd::prod_virial_a_gpu_cuda(    
          virial, atom_virial,
          net_deriv, in_deriv, rij, nlist, nloc, nall, nnei);
    }
}

std::vector<paddle::Tensor> PdProdVirialSeAOpCUDAForward(
const paddle::Tensor& net_deriv_tensor,
const paddle::Tensor& in_deriv_tensor,
const paddle::Tensor& rij_tensor,
const paddle::Tensor& nlist_tensor,
const paddle::Tensor& natoms_tensor,
int n_a_sel, 
int n_r_sel
){
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

    PD_CHECK(natoms_tensor.shape()[0] >= 3, "number of atoms should be larger than (or equal to) 3");
    const int* natoms = natoms_tensor.copy_to<int>(paddle::PlaceType::kCPU).data<int>();
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
    paddle::Tensor virial_tensor = paddle::Tensor(paddle::PlaceType::kGPU, virial_shape);
    paddle::Tensor atom_virial_tensor = paddle::Tensor(paddle::PlaceType::kGPU, atom_virial_shape);

    PD_DISPATCH_FLOATING_TYPES(
      net_deriv_tensor.type(), "pd_prod_virial_se_a_cpu_forward_kernel", ([&] {
        PdProdVirialSeAOpForwardCUDAKernel<data_t>(
            nloc, nall, ndescrpt, nnei, nframes,
            virial_tensor.mutable_data<data_t>(), atom_virial_tensor.mutable_data<data_t>(), 
            net_deriv_tensor.data<data_t>(), in_deriv_tensor.data<data_t>(),
            rij_tensor.data<data_t>(), nlist_tensor.data<int>());
      }));

    return {virial_tensor, atom_virial_tensor};
}