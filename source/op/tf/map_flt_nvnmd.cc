
// SPDX-License-Identifier: LGPL-3.0-or-later
//
// --------------------------------------------------------------------
/*

# Function
x = xk+dx
y = vk+dvk*dx

build a mapping table V, use the X as index to select value Y

# Parameters
x index
table mapping table
table_grad mapping table of gradient
table_info information of mapping table:
  x0 x1 dx N0 N1
y output
*/
// --------------------------------------------------------------------
//

//- import the library of tensorflow
#include <algorithm>

#include "custom_op.h"
#include "env_mat_nvnmd.h"

using namespace tensorflow;

template <class T>  // float and double
void mul_flt_nvnmd(T& y, T x1, T x2);

template <class T>  // float and double
void add_flt_nvnmd(T& y, T x1, T x2);

//- register the operator
// prec = 2^n, so it doesn't need to match `T`
REGISTER_OP("MapFltNvnmd")
    .Attr("T: {float, double} = DT_DOUBLE")
    .Input("x: T")
    .Input("table: T")
    .Input("table_grad: T")
    .Input("table_info: T")
    .Output("y: T");

//- create the operator class
//* the class must inherit the OpKernel Class
template <typename Device, typename FPTYPE>
class MapFltNvnmdOp : public OpKernel {
 public:
  /// Constructor.
  explicit MapFltNvnmdOp(OpKernelConstruction* context) : OpKernel(context) {}

  /// Compute the descriptor
  /// param: context
  void Compute(OpKernelContext* context) override {
    OP_REQUIRES(context, context->num_inputs() == 4,
                deepmd::tf_compat::InvalidArgument(
                    "MapFltNvnmd expects four input tensors"));

    const Tensor& t_x = context->input(0);
    const Tensor& t_table = context->input(1);
    const Tensor& t_table_grad = context->input(2);
    const Tensor& t_table_info = context->input(3);

    const TensorShape& shX = t_x.shape();
    const TensorShape& shT = t_table.shape();
    const TensorShape& shTG = t_table_grad.shape();
    const TensorShape& shI = t_table_info.shape();

    OP_REQUIRES(context, shX.dims() == 2,
                deepmd::tf_compat::InvalidArgument("Dim of x should be 2"));
    OP_REQUIRES(context, shT.dims() == 2,
                deepmd::tf_compat::InvalidArgument("Dim of table should be 2"));
    OP_REQUIRES(context, shTG == shT,
                deepmd::tf_compat::InvalidArgument(
                    "table_grad shape should match table"));
    OP_REQUIRES(
        context, shI.dims() == 1,
        deepmd::tf_compat::InvalidArgument("Dim of table_info should be 1"));
    OP_REQUIRES(context, shT.dim_size(1) > 0 && shT.dim_size(1) % 4 == 0,
                deepmd::tf_compat::InvalidArgument(
                    "table width should be a positive multiple of 4"));
    OP_REQUIRES(context, shI.dim_size(0) > 0 && shI.dim_size(0) % 5 == 0,
                deepmd::tf_compat::InvalidArgument(
                    "table_info length should be a positive multiple of 5"));

    int N = shX.dim_size(0);
    int D = shX.dim_size(1);
    int M = shT.dim_size(1) / 4;
    int S = shI.dim_size(0) / 5;

    /*
     * Calculate the output
     * 1.create tensor
     * 2.allocate the memory
     * 3.calculate
     */

    //- 1.create tensor
    TensorShape shY;
    shY.AddDim(N);
    shY.AddDim(D);
    shY.AddDim(M);
    Tensor* t_y = NULL;

    //- 2.allocate the memory
    //* allocate memory for the Y tensor which is called output 0
    OP_REQUIRES_OK(context, context->allocate_output(0, shY, &t_y));
    auto x = t_x.flat<FPTYPE>().data();
    auto table = t_table.flat<FPTYPE>().data();
    auto info = t_table_info.flat<FPTYPE>().data();
    auto y = t_y->flat<FPTYPE>().data();

    // Values outside every configured interval intentionally map to zero.
    // Initialize first so skipped entries never expose allocator contents.
    if (t_y->NumElements() > 0) {
      std::fill(y, y + t_y->NumElements(), FPTYPE(0));
    }

    for (int interval = 0; interval < S; ++interval) {
      const FPTYPE x0 = info[interval * 5 + 0];
      const FPTYPE x1 = info[interval * 5 + 1];
      const FPTYPE dx = info[interval * 5 + 2];
      const int n0 = int(info[interval * 5 + 3]);
      const int n1 = int(info[interval * 5 + 4]);
      OP_REQUIRES(
          context,
          x0 <= x1 && dx > 0 && n0 >= 0 && n1 > n0 && n1 <= shT.dim_size(0),
          deepmd::tf_compat::InvalidArgument("invalid interval in table_info"));
    }

    int ss, ii, jj;
    FPTYPE xi, x0, x1, dx;
    FPTYPE xx, id;
    int idx;
    int N0, N1, dN;

    U_Flt64_Int64 ufi;

    FPTYPE ytmp;
    FPTYPE ytmp2;
    for (ss = S - 1; ss >= 0; ss--) {
      x0 = info[ss * 5 + 0];
      x1 = info[ss * 5 + 1];
      dx = info[ss * 5 + 2];
      N0 = int(info[ss * 5 + 3]);
      N1 = int(info[ss * 5 + 4]);
      dN = N1 - N0;
      for (ii = 0; ii < N * D; ii++) {
        // cal idx and xx
        xi = x[ii];
        if ((xi < x0) || (xi > x1)) {
          continue;
        }
        //
        xx = xi - x0;
        id = floor(xx / dx);
        id = (id < 0) ? 0 : id;
        id = (id >= dN) ? (dN - 1) : id;
        xx -= id * dx;
        idx = id + N0;
        //
        ufi.nflt = xx;
        ufi.nint &= 0xfffffff000000000;  // 52 - 16 = 36 = 9 * 4
        xx = ufi.nflt;
        for (jj = 0; jj < M; jj++) {
          FPTYPE a = table[idx * M * 4 + jj * 4 + 0];
          FPTYPE b = table[idx * M * 4 + jj * 4 + 1];
          FPTYPE c = table[idx * M * 4 + jj * 4 + 2];
          FPTYPE d = table[idx * M * 4 + jj * 4 + 3];
          mul_flt_nvnmd(ytmp, a, xx);
          add_flt_nvnmd(ytmp, b, ytmp);
          mul_flt_nvnmd(ytmp, ytmp, xx);
          add_flt_nvnmd(ytmp, c, ytmp);
          mul_flt_nvnmd(ytmp, ytmp, xx);
          add_flt_nvnmd(ytmp, d, ytmp);
          y[ii * M + jj] = ytmp;
        }  // jj
      }  // ii
    }  // ss
  }  // Compute
};  // MapFltNvnmdOp

#define REGISTER_CPU(T)                                              \
  REGISTER_KERNEL_BUILDER(                                           \
      Name("MapFltNvnmd").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      MapFltNvnmdOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(double);
