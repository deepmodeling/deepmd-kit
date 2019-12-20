#include <vector>
#include <string.h>
#include <iostream>
#include <cuda_runtime.h>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;  // NOLINT(build/namespaces)
#define MAGIC_NUMBER 256

#ifdef HIGH_PREC
    typedef double VALUETYPE ;
#else
    typedef float  VALUETYPE ;
#endif

typedef double compute_t;

#define cudaErrcheck(res) { cudaAssert((res), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

using GPUDevice = Eigen::GpuDevice;

#ifdef HIGH_PREC
REGISTER_OP("DescrptSeA")
    .Input("coord: double")     //atomic coordinates
    .Input("type: int32")       //atomic type
    .Input("natoms: int32")     //local atomic number; each type atomic number; daizheyingxiangqude atomic numbers
    .Input("box : double")
    .Input("mesh : int32")
    .Input("davg: double")      //average value of data
    .Input("dstd: double")      //standard deviation
    .Attr("rcut_a: float")      //no use
    .Attr("rcut_r: float")
    .Attr("rcut_r_smth: float")
    .Attr("sel_a: list(int)")
    .Attr("sel_r: list(int)")   //all zero
    .Output("descrpt: double")
    .Output("descrpt_deriv: double")
    .Output("rij: double")
    .Output("nlist: int32");
    // only sel_a and rcut_r uesd.
#else
REGISTER_OP("DescrptSeA")
    .Input("coord: float")
    .Input("type: int32")
    .Input("natoms: int32")
    .Input("box : float")
    .Input("mesh : int32")
    .Input("davg: float")
    .Input("dstd: float")
    .Attr("rcut_a: float")
    .Attr("rcut_r: float")
    .Attr("rcut_r_smth: float")
    .Attr("sel_a: list(int)")
    .Attr("sel_r: list(int)")
    .Output("descrpt: float")
    .Output("descrpt_deriv: float")
    .Output("rij: float")
    .Output("nlist: int32");
#endif

void DescrptSeALauncher(const VALUETYPE* coord,
                            const int* type,
                            const int* ilist,
                            const int* jrange,
                            const int* jlist,
                            int* array_int,
                            unsigned long long* array_longlong,
                            compute_t* array_double,
                            const VALUETYPE* avg,
                            const VALUETYPE* std,
                            VALUETYPE* descript,
                            VALUETYPE* descript_deriv,
                            VALUETYPE* rij,
                            int* nlist,
                            const int& ntypes,
                            const int& nloc,
                            const int& nall,
                            const int& nnei,
                            const float& rcut_r,
                            const float& rcut_r_smth,
                            const int& ndescrpt,
                            const std::vector<int>& sec_a,
                            const bool& fill_nei_a
);

class DescrptSeAOp : public OpKernel {
public:
    explicit DescrptSeAOp(OpKernelConstruction* context) : OpKernel(context) {
        float nloc_f, nall_f;
        OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
        OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
        OP_REQUIRES_OK(context, context->GetAttr("rcut_r_smth", &rcut_r_smth));
        OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
        OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
        // OP_REQUIRES_OK(context, context->GetAttr("nloc", &nloc_f));
        // OP_REQUIRES_OK(context, context->GetAttr("nall", &nall_f));
        cum_sum (sec_a, sel_a);
        cum_sum (sec_r, sel_r);
        ndescrpt_a = sec_a.back() * 4;
        ndescrpt_r = sec_r.back() * 1;
        ndescrpt = ndescrpt_a + ndescrpt_r;
        nnei_a = sec_a.back();
        nnei_r = sec_r.back();
        nnei = nnei_a + nnei_r;
        fill_nei_a = (rcut_a < 0);
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& coord_tensor	= context->input(context_input_index++);
        const Tensor& type_tensor	= context->input(context_input_index++);
        const Tensor& natoms_tensor	= context->input(context_input_index++);
        const Tensor& box_tensor	= context->input(context_input_index++);
        const Tensor& mesh_tensor   = context->input(context_input_index++);
        const Tensor& avg_tensor	= context->input(context_input_index++);
        const Tensor& std_tensor	= context->input(context_input_index++);
        // set size of the sample. assume 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]], then shape(t) ==> [2, 2, 3]
        OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of coord should be 2"));
        OP_REQUIRES (context, (type_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of type should be 2"));
        OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of natoms should be 1"));
        OP_REQUIRES (context, (box_tensor.shape().dims() == 2),	    errors::InvalidArgument ("Dim of box should be 2"));
        OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of mesh should be 1"));
        OP_REQUIRES (context, (avg_tensor.shape().dims() == 2),	    errors::InvalidArgument ("Dim of avg should be 2"));
        OP_REQUIRES (context, (std_tensor.shape().dims() == 2),	    errors::InvalidArgument ("Dim of std should be 2"));
        OP_REQUIRES (context, (fill_nei_a),                         errors::InvalidArgument ("Rotational free descriptor only support the case rcut_a < 0"));
        OP_REQUIRES (context, (sec_r.back() == 0),			        errors::InvalidArgument ("Rotational free descriptor only support all-angular information: sel_r should be all zero."));

        OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),		errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));

        int * natoms = new int[natoms_tensor.shape().dim_size(0)];
        cudaErrcheck(cudaMemcpy(natoms, natoms_tensor.flat<int>().data(), sizeof(int) * natoms_tensor.shape().dim_size(0), cudaMemcpyDeviceToHost));
        int nloc = natoms[0];
        int nall = natoms[1];
        int ntypes = natoms_tensor.shape().dim_size(0) - 2; //nloc and nall mean something.
        int nsamples = coord_tensor.shape().dim_size(0);
        //
        //// check the sizes
        OP_REQUIRES (context, (nsamples == type_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
        OP_REQUIRES (context, (nsamples == box_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
        OP_REQUIRES (context, (ntypes == avg_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of avg should be ntype"));
        OP_REQUIRES (context, (ntypes == std_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of std should be ntype"));
        
        OP_REQUIRES (context, (nall * 3 == coord_tensor.shape().dim_size(1)),	errors::InvalidArgument ("number of atoms should match"));
        OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of atoms should match"));
        OP_REQUIRES (context, (9 == box_tensor.shape().dim_size(1)),		    errors::InvalidArgument ("number of box should be 9"));
        OP_REQUIRES (context, (ndescrpt == avg_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of avg should be ndescrpt"));
        OP_REQUIRES (context, (ndescrpt == std_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of std should be ndescrpt"));   
        
        OP_REQUIRES (context, (ntypes == int(sel_a.size())),	errors::InvalidArgument ("number of types should match the length of sel array"));
        OP_REQUIRES (context, (ntypes == int(sel_r.size())),	errors::InvalidArgument ("number of types should match the length of sel array"));
        
        // Create output tensors
        TensorShape descrpt_shape ;
        descrpt_shape.AddDim (nsamples);
        descrpt_shape.AddDim (nloc * ndescrpt);
        TensorShape descrpt_deriv_shape ;
        descrpt_deriv_shape.AddDim (nsamples);
        descrpt_deriv_shape.AddDim (nloc * ndescrpt * 3);
        TensorShape rij_shape ;
        rij_shape.AddDim (nsamples);
        rij_shape.AddDim (nloc * nnei * 3);
        TensorShape nlist_shape ;
        nlist_shape.AddDim (nsamples);
        nlist_shape.AddDim (nloc * nnei);

        int context_output_index = 0;
        Tensor* descrpt_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     descrpt_shape,
	    					     &descrpt_tensor));
        Tensor* descrpt_deriv_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     descrpt_deriv_shape,
	    					     &descrpt_deriv_tensor));
        Tensor* rij_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     rij_shape,
	    					     &rij_tensor));
        Tensor* nlist_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(context_output_index++,
	    					     nlist_shape,
	    					     &nlist_tensor));
        
	    int * ilist = NULL, *jrange = NULL, *jlist = NULL;
        int * array_int = NULL; unsigned long long *array_longlong = NULL; compute_t *array_double = NULL;
        cudaErrcheck(cudaMemcpy(&(ilist), 4 + mesh_tensor.flat<int>().data(), sizeof(int *), cudaMemcpyDeviceToHost));
        cudaErrcheck(cudaMemcpy(&(jrange), 8 + mesh_tensor.flat<int>().data(), sizeof(int *), cudaMemcpyDeviceToHost));
        cudaErrcheck(cudaMemcpy(&(jlist), 12 + mesh_tensor.flat<int>().data(), sizeof(int *), cudaMemcpyDeviceToHost));
        cudaErrcheck(cudaMemcpy(&(array_int), 16 + mesh_tensor.flat<int>().data(), sizeof(int *), cudaMemcpyDeviceToHost));
        cudaErrcheck(cudaMemcpy(&(array_longlong), 20 + mesh_tensor.flat<int>().data(), sizeof(unsigned long long *), cudaMemcpyDeviceToHost));
        cudaErrcheck(cudaMemcpy(&(array_double), 24 + mesh_tensor.flat<int>().data(), sizeof(compute_t *), cudaMemcpyDeviceToHost));

        // cudaErrcheck(cudaMemcpy(jlist, host_jlist, sizeof(int) * nloc * MAGIC_NUMBER, cudaMemcpyHostToDevice));
        // Launch computation
        for (int II = 0; II < nsamples; II++) {
            DescrptSeALauncher(coord_tensor.matrix<VALUETYPE>().data() + II * (nall * 3),    // related to the kk argument
                        type_tensor.matrix<int>().data() + II * nall,           // also related to the kk argument
                        ilist,
                        jrange,
                        jlist,
                        array_int,
                        array_longlong,
                        array_double,
                        avg_tensor.matrix<VALUETYPE>().data(),
                        std_tensor.matrix<VALUETYPE>().data(),
                        descrpt_tensor->matrix<VALUETYPE>().data() + II * (nloc * ndescrpt),
                        descrpt_deriv_tensor->matrix<VALUETYPE>().data() + II * (nloc * ndescrpt * 3),
                        rij_tensor->matrix<VALUETYPE>().data() + II * (nloc * nnei * 3),
                        nlist_tensor->matrix<int>().data() + II * (nloc * nnei),
                        ntypes,
                        nloc,
                        nall,
                        nnei,
                        rcut_r,
                        rcut_r_smth,
                        ndescrpt,
                        sec_a,
                        fill_nei_a
            );
        }
        // std::cout << "done" << std::endl;
        delete[] natoms;
    }

/////////////////////////////////////////////////////////////////////////////////////////////
private:
    float rcut_a;
    float rcut_r;
    float rcut_r_smth;
    std::vector<int32> sel_r;
    std::vector<int32> sel_a;
    std::vector<int> sec_a;
    std::vector<int> sec_r;
    int ndescrpt, ndescrpt_a, ndescrpt_r;
    int nnei, nnei_a, nnei_r, nloc, nall;
    bool fill_nei_a;

    //private func
    void cum_sum (std::vector<int> & sec, const std::vector<int32> & n_sel) const {
        sec.resize (n_sel.size() + 1);
        sec[0] = 0;
        for (int ii = 1; ii < sec.size(); ++ii) {
            sec[ii] = sec[ii-1] + n_sel[ii-1];
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("DescrptSeA").Device(DEVICE_GPU), DescrptSeAOp);