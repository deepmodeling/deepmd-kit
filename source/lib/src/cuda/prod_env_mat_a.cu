#include "prod_env_mat.h"
#include "gpu_nv.cuh"

template<
    typename FPTYPE,
    int      THREADS_PER_BLOCK>
__global__ void compute_em_mat_a(
    FPTYPE* em,
    FPTYPE* em_deriv,
    FPTYPE* rij,
    const FPTYPE* coord,
    const FPTYPE* avg,
    const FPTYPE* std,
    const int* type,
    const int* nlist,
    const int nnei,
    const float rmin,
    const float rmax)
{   
  // <<<nloc, TPB>>>
  const unsigned int bid = blockIdx.x;
  const unsigned int tid = threadIdx.x;
  if (tid >= nnei) {
    return;
  }
  const int ndescrpt = nnei * 4;
  const int * row_nlist = nlist + bid * nnei;
  FPTYPE * row_rij = rij + bid * nnei * 3;
  FPTYPE * row_descript = em + bid * nnei * 4;
  FPTYPE * row_descript_deriv = em_deriv + bid * nnei * 12;
  for (int ii = tid; ii < nnei; ii += THREADS_PER_BLOCK) {
    const int idx_value = ii * 4;	  // 4 components
    const int idx_deriv = ii * 12;	// 4 components time 3 directions
    if (row_nlist[ii] >= 0) {
      FPTYPE rr[3]  = {0};
      FPTYPE dd[4]  = {0};
      FPTYPE vv[12] = {0};
      const int j_idx = row_nlist[ii];
      for (int kk = 0; kk < 3; kk++) {
        rr[kk] = coord[j_idx * 3 + kk] - coord[bid * 3 + kk];
        row_rij[ii * 3 + kk] = rr[kk];
      }
      // const FPTYPE * rr = &row_rij[ii * 3];
      FPTYPE nr2 = dev_dot(rr, rr);
      FPTYPE inr = 1./sqrt(nr2);
      FPTYPE nr = nr2 * inr;
      FPTYPE inr2 = inr * inr;
      FPTYPE inr4 = inr2 * inr2;
      FPTYPE inr3 = inr4 * nr;
      FPTYPE sw, dsw;
      spline5_switch(sw, dsw, nr, rmin, rmax);
      dd[0] = (1./nr)       ;//* sw;
      dd[1] = (rr[0] / nr2) ;//* sw;
      dd[2] = (rr[1] / nr2) ;//* sw;
      dd[3] = (rr[2] / nr2) ;//* sw;
      vv[0] = (rr[0] * inr3 * sw - dd[0] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 0) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 0) % (ndescrpt * 3)) / 3];
      vv[1] = (rr[1] * inr3 * sw - dd[0] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 1) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 1) % (ndescrpt * 3)) / 3];
      vv[2] = (rr[2] * inr3 * sw - dd[0] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 2) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 2) % (ndescrpt * 3)) / 3];
      // ****deriv of component x/r2
      vv[3] = ((2. * rr[0] * rr[0] * inr4 - inr2) * sw - dd[1] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 3) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 3) % (ndescrpt * 3)) / 3];
      vv[4] = ((2. * rr[0] * rr[1] * inr4	) * sw - dd[1] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 4) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 4) % (ndescrpt * 3)) / 3];
      vv[5] = ((2. * rr[0] * rr[2] * inr4	) * sw - dd[1] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 5) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 5) % (ndescrpt * 3)) / 3];
      // ***deriv of component y/r2
      vv[6] = ((2. * rr[1] * rr[0] * inr4	) * sw - dd[2] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 6) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 6) % (ndescrpt * 3)) / 3];
      vv[7] = ((2. * rr[1] * rr[1] * inr4 - inr2) * sw - dd[2] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 7) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 7) % (ndescrpt * 3)) / 3];
      vv[8] = ((2. * rr[1] * rr[2] * inr4	) * sw - dd[2] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 8) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 8) % (ndescrpt * 3)) / 3];
      // ***deriv of component z/r2 
      vv[9] = ((2. * rr[2] * rr[0] * inr4	) * sw - dd[3] * dsw * rr[0] * inr); // avg[type[(idx_deriv + 9) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 9) % (ndescrpt * 3)) / 3];
      vv[10]= ((2. * rr[2] * rr[1] * inr4	) * sw - dd[3] * dsw * rr[1] * inr); // avg[type[(idx_deriv + 10) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 10) % (ndescrpt * 3)) / 3];
      vv[11]= ((2. * rr[2] * rr[2] * inr4 - inr2) * sw - dd[3] * dsw * rr[2] * inr); // avg[type[(idx_deriv + 11) / (ndescrpt * 3)] * ndescrpt + ((idx_deriv + 11) % (ndescrpt * 3)) / 3];
      // 4 value components
      dd[0] *= sw; // * em[idx * ndescrpt + idx_value + 0]);// - avg[type[idx] * ndescrpt + idx_value + 0]) / std[type[idx] * ndescrpt + idx_value + 0];
      dd[1] *= sw; // * em[idx * ndescrpt + idx_value + 1]);// - avg[type[idx] * ndescrpt + idx_value + 1]) / std[type[idx] * ndescrpt + idx_value + 1];
      dd[2] *= sw; // * em[idx * ndescrpt + idx_value + 2]);// - avg[type[idx] * ndescrpt + idx_value + 2]) / std[type[idx] * ndescrpt + idx_value + 2];
      dd[3] *= sw; // * em[idx * ndescrpt + idx_value + 3]);// - avg[type[idx] * ndescrpt + idx_value + 3]) / std[type[idx] * ndescrpt + idx_value + 3];
      for (int ii = 0; ii < 12; ii++) {
        row_descript_deriv[idx_deriv + ii] = vv[ii] / std[type[bid] * ndescrpt + idx_value + ii / 3];
      }
      for (int ii = 0; ii < 4; ii++) {  
        row_descript[idx_value + ii] = (dd[ii] - avg[type[bid] * ndescrpt + idx_value + ii]) / std[type[bid] * ndescrpt + idx_value + ii];
      }
    }
    else {
      // TODO: move it to the memset.
      row_descript[idx_value] -= avg[type[bid] * ndescrpt + idx_value] / std[type[bid] * ndescrpt + idx_value];
    }
  }
}

template <typename FPTYPE>
void prod_env_mat_a_gpu_nv(    
    FPTYPE * em, 
    FPTYPE * em_deriv, 
    FPTYPE * rij, 
    int * nlist, 
    const FPTYPE * coord, 
    const int * type, 
    const int * ilist, 
    const int * jrange, 
    const int * jlist,
    int * array_int, 
    int_64 * array_longlong,
    const int max_nbor_size,
    const FPTYPE * avg, 
    const FPTYPE * std, 
    const int nloc, 
    const int nall, 
    const float rcut, 
    const float rcut_smth, 
    const std::vector<int> sec)
{
  prod_env_mat_common(
      em, em_deriv, rij, nlist, 
      coord, type, ilist, jrange, jlist, array_int, array_longlong, max_nbor_size, avg, std, nloc, nall, rcut, rcut_smth, sec);

  compute_em_mat_a<FPTYPE, TPB> <<<nloc, TPB>>> (
      em, em_deriv, rij, 
      coord, avg, std, type, nlist, sec.back(), rcut_smth, rcut);
}

template void prod_env_mat_a_gpu_nv<float>(float * em, float * em_deriv, float * rij, int * nlist, const float * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const int max_nbor_size, const float * avg, const float * std, const int nloc, const int nall, const float rcut, const float rcut_smth, const std::vector<int> sec);
template void prod_env_mat_a_gpu_nv<double>(double * em, double * em_deriv, double * rij, int * nlist, const double * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const int max_nbor_size, const double * avg, const double * std, const int nloc, const int nall, const float rcut, const float rcut_smth, const std::vector<int> sec);