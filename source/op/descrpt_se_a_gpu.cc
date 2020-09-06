#include "common.h"
#include "CustomeOperation.h"

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
    .Output("nlist: int32");    //only sel_a and rcut_r uesd.
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

int get_magic_number(int const nnei) {
    if      (nnei <= 256 ) {return 256; }
    else if (nnei <= 512 ) {return 512; }
    else if (nnei <= 1024) {return 1024;}
    else if (nnei <= 2048) {return 2048;}
    else if (nnei <= 4096) {return 4096;}
}

template <typename Device, typename T> 
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

template <typename Device, typename T>
struct DescrptSeAFunctor {
    void operator()(const CPUDevice& d, const T * coord, const int * type, const int * mesh, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const T * avg, const T * std, T * descrpt, T * descrpt_deriv, T * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ntypes, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
        DescrptSeACPULauncher(coord, type, ilist, jrange, jlist, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ntypes, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, magic_number);
    }

    #if GOOGLE_CUDA
    void operator()(const GPUDevice& d, const T * coord, const int * type, const int * mesh, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const T * avg, const T * std, T * descrpt, T * descrpt_deriv, T * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ntypes, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int magic_number) {
        DescrptSeAGPULauncher(coord, type, ilist, jrange, jlist, array_int, array_longlong, avg, std, descrpt, descrpt_deriv, rij, nlist, nloc, nall, nnei, ndescrpt, rcut_r, rcut_r_smth, sec_a, fill_nei_a, magic_number);
    }
    #endif // GOOGLE_CUDA
};

template <typename Device>
class DescrptSeAOp : public OpKernel {
public:
    explicit DescrptSeAOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("rcut_a", &rcut_a));
        OP_REQUIRES_OK(context, context->GetAttr("rcut_r", &rcut_r));
        OP_REQUIRES_OK(context, context->GetAttr("rcut_r_smth", &rcut_r_smth));
        OP_REQUIRES_OK(context, context->GetAttr("sel_a", &sel_a));
        OP_REQUIRES_OK(context, context->GetAttr("sel_r", &sel_r));
        cum_sum (sec_a, sel_a);
        cum_sum (sec_r, sel_r);
        ndescrpt_a = sec_a.back() * 4;
        ndescrpt_r = sec_r.back() * 1;
        ndescrpt = ndescrpt_a + ndescrpt_r;
        nnei_a = sec_a.back();
        nnei_r = sec_r.back();
        nnei = nnei_a + nnei_r;
        fill_nei_a = (rcut_a < 0);
        magic_number = get_magic_number(nnei);
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

        DeviceFunctor<Device, VALUETYPE>() (
            context->eigen_device<Device>(),
            device
        );

        int const * natoms = natoms_tensor.flat<int>().data();
        int nloc = natoms[0], nall = natoms[1];
        int ntypes = natoms_tensor.shape().dim_size(0) - 2; //nloc and nall mean something.
        int nsamples = coord_tensor.shape().dim_size(0);

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
        OP_REQUIRES (context, (nnei <= 4096),	                errors::InvalidArgument ("Assert failed, max neighbor size of atom(nnei) " + std::to_string(nnei) + " is larger than 4096, which currently is not supported by deepmd-kit."));

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
        
        if(device == "GPU") {
            // allocate temp memory, temp memory must not be used after this operation!
            Tensor int_temp;
            TensorShape int_shape;
            int_shape.AddDim(sec_a.size() + nloc * sec_a.size() + nloc);
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
        
        DescrptSeAFunctor<Device, VALUETYPE>()(
            context->eigen_device<Device>(),            // define actually graph execution device
            coord_tensor.matrix<VALUETYPE>().data(),    // related to the kk argument
            type_tensor.matrix<int>().data(),           // also related to the kk argument
            mesh_tensor.flat<int>().data(),
            ilist,
            jrange,
            jlist,
            array_int,
            array_longlong,
            avg_tensor.matrix<VALUETYPE>().data(),
            std_tensor.matrix<VALUETYPE>().data(),
            descrpt_tensor->matrix<VALUETYPE>().data(),
            descrpt_deriv_tensor->matrix<VALUETYPE>().data(),
            rij_tensor->matrix<VALUETYPE>().data(),
            nlist_tensor->matrix<int>().data(),
            nloc,
            nall,
            nnei,
            ntypes,
            ndescrpt,
            rcut_r,
            rcut_r_smth,
            sec_a,
            fill_nei_a,
            magic_number
        );
    }
    ~DescrptSeAOp() {
        if (init) {
            #if GOOGLE_CUDA
            cudaErrcheck(cudaFree(ilist));
            cudaErrcheck(cudaFree(jrange));
            cudaErrcheck(cudaFree(jlist));
            #endif // GOOGL_CUDA
        }
    }
/////////////////////////////////////////////////////////////////////////////////////////////
private:
    bool fill_nei_a;
    std::string device;
    std::vector<int> sel_r;
    std::vector<int> sel_a;
    std::vector<int> sec_a;
    std::vector<int> sec_r;
    int *array_int;
    unsigned long long*array_longlong;
    float rcut_a, rcut_r, rcut_r_smth;
    int ndescrpt, ndescrpt_a, ndescrpt_r;
    int nnei, nnei_a, nnei_r, nloc, nall, magic_number;

    //private func
    void cum_sum (std::vector<int> & sec, const std::vector<int> & n_sel) const {
        sec.resize (n_sel.size() + 1);
        sec[0] = 0;
        for (int ii = 1; ii < sec.size(); ++ii) {
            sec[ii] = sec[ii-1] + n_sel[ii-1];
        }
    }

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
};

// Register the CPU kernels.
#define REGISTER_CPU()                                           \
REGISTER_KERNEL_BUILDER(                                         \
    Name("DescrptSeA").Device(DEVICE_CPU),                       \
    DescrptSeAOp<CPUDevice>);
REGISTER_CPU();

// Register the GPU kernels.
#if GOOGLE_CUDA
#define REGISTER_GPU()                                           \
REGISTER_KERNEL_BUILDER(                                         \
    Name("DescrptSeA").Device(DEVICE_GPU).HostMemory("natoms"),  \
    DescrptSeAOp<GPUDevice>);
REGISTER_GPU();
#endif  // GOOGLE_CUDA