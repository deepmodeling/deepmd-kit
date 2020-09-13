#include "common.h"
#include "CustomeOperation.h"

REGISTER_OP("DescrptSeR")
    .Attr("T: {float, double}")
    .Input("coord: T")
    .Input("type: int32")
    .Input("natoms: int32")
    .Input("box: T")
    .Input("mesh: int32")
    .Input("davg: T")
    .Input("dstd: T")
    .Attr("rcut: float")
    .Attr("rcut_smth: float")
    .Attr("sel: list(int)")
    .Output("descrpt: T")
    .Output("descrpt_deriv: T")
    .Output("rij: T")
    .Output("nlist: int32");

struct DeviceFunctor {
    void operator()(const CPUDevice& d, std::string& device) {
        device = "CPU";
    }
    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, std::string& device) {
        device = "GPU";
    }
    #endif // GOOGLE_CUDA
};

template <typename T>
struct DescrptSeRFunctor {
    void operator()(const CPUDevice& d, const T * coord, const int * type, const int * mesh, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const T * avg, const T * std, T * descrpt, T * descrpt_deriv, T * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ntypes, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
        DescrptSeRCPULauncher(coord, type, ilist, jrange, jlist, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ntypes, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, magic_number);
    }

    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const T * coord, const int * type, const int * mesh, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const T * avg, const T * std, T * descrpt, T * descrpt_deriv, T * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ntypes, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
        DescrptSeRGPULauncher(coord, type, ilist, jrange, jlist, array_int, array_longlong, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, magic_number);
    }
    #endif // GOOGLE_CUDA 
};

template<typename Device, typename FPTYPE>
class DescrptSeROp : public OpKernel {
public:
    explicit DescrptSeROp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("rcut", &rcut));
        OP_REQUIRES_OK(context, context->GetAttr("rcut_smth", &rcut_smth));
        OP_REQUIRES_OK(context, context->GetAttr("sel", &sel));
        cum_sum (sec, sel);
        sel_null.resize(3, 0);
        cum_sum (sec_null, sel_null);
        ndescrpt = sec.back() * 1;
        nnei = sec.back();
        fill_nei_a = true;
        magic_number = get_magic_number(nnei);
        // count_nei_idx_overflow = 0;
        // std::cout << "I'm in descrpt_se_r_gpu.cc" << std::endl;
    }

    void Compute(OpKernelContext* context) override {
        // Grab the input tensor
        int context_input_index = 0;
        const Tensor& coord_tensor	= context->input(context_input_index++);
        const Tensor& type_tensor	= context->input(context_input_index++);
        const Tensor& natoms_tensor	= context->input(context_input_index++);
        const Tensor& box_tensor	= context->input(context_input_index++);
        const Tensor& mesh_tensor	= context->input(context_input_index++);
        const Tensor& avg_tensor	= context->input(context_input_index++);
        const Tensor& std_tensor	= context->input(context_input_index++);

        // set size of the sample
        OP_REQUIRES (context, (coord_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of coord should be 2"));
        OP_REQUIRES (context, (type_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of type should be 2"));
        OP_REQUIRES (context, (natoms_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of natoms should be 1"));
        OP_REQUIRES (context, (box_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of box should be 2"));
        OP_REQUIRES (context, (mesh_tensor.shape().dims() == 1),	errors::InvalidArgument ("Dim of mesh should be 1"));
        OP_REQUIRES (context, (avg_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of avg should be 2"));
        OP_REQUIRES (context, (std_tensor.shape().dims() == 2),	errors::InvalidArgument ("Dim of std should be 2"));
        OP_REQUIRES (context, (fill_nei_a),				errors::InvalidArgument ("Rotational free descriptor only support the case rcut_a < 0"));

        OP_REQUIRES (context, (natoms_tensor.shape().dim_size(0) >= 3),		errors::InvalidArgument ("number of atoms should be larger than (or equal to) 3"));

        DeviceFunctor() (
            context->eigen_device<Device>(),
            device
        );

        const int * natoms = natoms_tensor.flat<int>().data();
        int nloc = natoms[0];
        int nall = natoms[1];
        int ntypes = natoms_tensor.shape().dim_size(0) - 2; //nloc and nall mean something.
        int nsamples = coord_tensor.shape().dim_size(0);
        //
        //// check the sizes
        // check the sizes
        OP_REQUIRES (context, (nsamples == type_tensor.shape().dim_size(0)),	errors::InvalidArgument ("number of samples should match"));
        OP_REQUIRES (context, (nsamples == box_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of samples should match"));
        OP_REQUIRES (context, (ntypes == avg_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of avg should be ntype"));
        OP_REQUIRES (context, (ntypes == std_tensor.shape().dim_size(0)),		errors::InvalidArgument ("number of std should be ntype"));

        OP_REQUIRES (context, (nall * 3 == coord_tensor.shape().dim_size(1)),	errors::InvalidArgument ("number of atoms should match"));
        OP_REQUIRES (context, (nall == type_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of atoms should match"));
        OP_REQUIRES (context, (9 == box_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of box should be 9"));
        OP_REQUIRES (context, (ndescrpt == avg_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of avg should be ndescrpt"));
        OP_REQUIRES (context, (ndescrpt == std_tensor.shape().dim_size(1)),		errors::InvalidArgument ("number of std should be ndescrpt")); 
        
        OP_REQUIRES (context, (nnei <= 4096),	                errors::InvalidArgument ("Assert failed, max neighbor size of atom(nnei) " + std::to_string(nnei) + " is larger than 4096, which currently is not supported by deepmd-kit."));

        // Create an output tensor
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
            
        if(device == "GPU") {
            // allocate temp memory, temp memory must not be used after this operation!
            Tensor int_temp;
            TensorShape int_shape;
            int_shape.AddDim(sec.size() + nloc * sec.size() + nloc);
            OP_REQUIRES_OK(context, context->allocate_temp(DT_INT32, int_shape, &int_temp));
            Tensor uint64_temp;
            TensorShape uint64_shape;
            uint64_shape.AddDim(nloc * magic_number * 2);
            OP_REQUIRES_OK(context, context->allocate_temp(DT_UINT64, uint64_shape, &uint64_temp));

            array_int = int_temp.flat<int>().data(); 
            array_longlong = uint64_temp.flat<unsigned long long>().data();

            nbor_update(mesh_tensor.flat<int>().data(), static_cast<int>(mesh_tensor.NumElements()));
        }
        else if (device == "CPU") {
            memcpy (&ilist,  4  + mesh_tensor.flat<int>().data(), sizeof(int *));
	        memcpy (&jrange, 8  + mesh_tensor.flat<int>().data(), sizeof(int *));
	        memcpy (&jlist,  12 + mesh_tensor.flat<int>().data(), sizeof(int *));
        }

        DescrptSeRFunctor<FPTYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            coord_tensor.matrix<FPTYPE>().data(),    // related to the kk argument
            type_tensor.matrix<int>().data(),           // also related to the kk argument
            mesh_tensor.flat<int>().data(),
            ilist,
            jrange,
            jlist,
            array_int,
            array_longlong,
            avg_tensor.matrix<FPTYPE>().data(),
            std_tensor.matrix<FPTYPE>().data(),
            descrpt_tensor->matrix<FPTYPE>().data(),
            descrpt_deriv_tensor->matrix<FPTYPE>().data(),
            rij_tensor->matrix<FPTYPE>().data(),
            nlist_tensor->matrix<int>().data(),
            nloc,
            nall,
            nnei,
            ntypes,
            ndescrpt,
            rcut,
            rcut_smth,
            sec,
            fill_nei_a,
            magic_number
        );
    }

/////////////////////////////////////////////////////////////////////////////////////////////

private:
    float rcut;
    float rcut_smth;
    std::vector<int32> sel;
    std::vector<int32> sel_null;
    std::vector<int> sec;
    std::vector<int> sec_null;
    int nnei, ndescrpt, nloc, nall;
    bool fill_nei_a;

    //private func
    void cum_sum (std::vector<int> & sec, const std::vector<int32> & n_sel) const {
        sec.resize (n_sel.size() + 1);
        sec[0] = 0;
        for (int ii = 1; ii < sec.size(); ++ii) {
            sec[ii] = sec[ii-1] + n_sel[ii-1];
        }
    }

    int magic_number;
    std::string device;
    int *array_int;
    unsigned long long*array_longlong;
    int * ilist = NULL, * jrange = NULL, * jlist = NULL;
    int ilist_size = 0, jrange_size = 0, jlist_size = 0;
    bool init = false;

    void nbor_update(const int * mesh, const int size) {
        int *mesh_host = new int[size], *ilist_host = NULL, *jrange_host = NULL, *jlist_host = NULL;
        cudaErrcheck(cudaMemcpy(mesh_host, mesh, sizeof(int) * size, cudaMemcpyDeviceToHost));
        memcpy (&ilist_host,  4  + mesh_host, sizeof(int *));
	    memcpy (&jrange_host, 8  + mesh_host, sizeof(int *));
	    memcpy (&jlist_host,  12 + mesh_host, sizeof(int *));
        int const ago = mesh_host[0];
        if (!init) {
            ilist_size  = (int)(mesh_host[1] * 1.2);
            jrange_size = (int)(mesh_host[2] * 1.2);
            jlist_size  = (int)(mesh_host[3] * 1.2);
            cudaErrcheck(cudaMalloc((void **)&ilist,     sizeof(int) * ilist_size));
            cudaErrcheck(cudaMalloc((void **)&jrange,    sizeof(int) * jrange_size));
            cudaErrcheck(cudaMalloc((void **)&jlist,     sizeof(int) * jlist_size));
            init = true;
        }
        if (ago == 0) {
            if (ilist_size < mesh_host[1]) {
                ilist_size = (int)(mesh_host[1] * 1.2);
                cudaErrcheck(cudaFree(ilist));
                cudaErrcheck(cudaMalloc((void **)&ilist, sizeof(int) * ilist_size));
            }
            if (jrange_size < mesh_host[2]) {
                jrange_size = (int)(mesh_host[2] * 1.2);
                cudaErrcheck(cudaFree(jrange));
                cudaErrcheck(cudaMalloc((void **)&jrange,sizeof(int) * jrange_size));
            }
            if (jlist_size < mesh_host[3]) {
                jlist_size = (int)(mesh_host[3] * 1.2);
                cudaErrcheck(cudaFree(jlist));
                cudaErrcheck(cudaMalloc((void **)&jlist, sizeof(int) * jlist_size));
            }
            cudaErrcheck(cudaMemcpy(ilist,  ilist_host,  sizeof(int) * mesh_host[1], cudaMemcpyHostToDevice));
            cudaErrcheck(cudaMemcpy(jrange, jrange_host, sizeof(int) * mesh_host[2], cudaMemcpyHostToDevice));
            cudaErrcheck(cudaMemcpy(jlist,  jlist_host,  sizeof(int) * mesh_host[3], cudaMemcpyHostToDevice));
        }
        delete [] mesh_host;
    }

    int get_magic_number(int const nnei) {
        if (nnei <= 256) {
            return 256;
        }
        else if (nnei <= 512) {
            return 512;
        }
        else if (nnei <= 1024) {
            return 1024;
        }
        else if (nnei <= 2048) {
            return 2048;
        }
        else if (nnei <= 4096) {
            return 4096;
        }
    }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("DescrptSeR").Device(DEVICE_CPU).TypeConstraint<T>("T"),                       \
    DescrptSeROp<CPUDevice, T>); 
REGISTER_CPU(float);
REGISTER_CPU(double);
// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU(T)                                                                 \
REGISTER_KERNEL_BUILDER(                                                                \
    Name("DescrptSeR").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("natoms"),  \
    DescrptSeROp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(double);
#endif  // GOOGLE_CUDA