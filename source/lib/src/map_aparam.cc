#include "map_aparam.h"

template <typename FPTYPE>
void deepmd::map_aparam_cpu (
    FPTYPE * output,
    const FPTYPE * aparam,
    const int * nlist,
    const int & nloc,
    const int & nnei,
    const int & numb_aparam
    )
//
//	output:	nloc x nnei x numb_aparam
//	aparam:	nall x numb_aparam
//	nlist:	nloc x nnei
//
{
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;
    for (int dd = 0; dd < nnei * numb_aparam; ++dd) {
      output[i_idx * nnei * numb_aparam + dd] = 0.;
    }
  }

  // loop over loc atoms
  for (int ii = 0; ii < nloc; ++ii){
    int i_idx = ii;	
    // loop over neighbor atoms
    for (int jj = 0; jj < nnei; ++jj){
      int j_idx = nlist[i_idx * nnei + jj];
      if (j_idx < 0) continue;
      // loop over elements of aparam
      for (int dd = 0; dd < numb_aparam; ++dd){
	output[ii * nnei * numb_aparam + jj * numb_aparam + dd] = aparam[j_idx * numb_aparam + dd];
      }
    }
  }  
}

template
void deepmd::map_aparam_cpu<double> (
    double * output,
    const double * aparam,
    const int * nlist,
    const int & nloc,
    const int & nnei,
    const int & numb_aparam
    );

template
void deepmd::map_aparam_cpu<float> (
    float * output,
    const float * aparam,
    const int * nlist,
    const int & nloc,
    const int & nnei,
    const int & numb_aparam
    );

