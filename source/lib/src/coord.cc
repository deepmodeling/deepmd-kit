#include "coord.h"
#include "neighbor_list.h"
#include "SimulationRegion.h"
#include <vector>

using namespace deepmd;

// normalize coords
template <typename FPTYPE>
void
deepmd::
normalize_coord_cpu(
    FPTYPE * coord,
    const int natom,
    const Region<FPTYPE> & region)
{
  for(int ii = 0; ii < natom; ++ii){
    FPTYPE ri[3];
    convert_to_inter_cpu(ri, region, coord+3*ii);
    for(int dd = 0; dd < 3; ++dd){
      while(ri[dd] >= 1.) ri[dd] -= 1.;
      while(ri[dd] <  0.) ri[dd] += 1.;
    }
    convert_to_phys_cpu(coord+3*ii, region, ri);
  }
}


template <typename FPTYPE>
int
deepmd::
copy_coord_cpu(
    FPTYPE * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    const FPTYPE * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall_,
    const float & rcut,
    const Region<FPTYPE> & region)
{
  const int mem_nall = mem_nall_;
  std::vector<double> coord(nloc * 3);
  std::vector<int> atype(nloc);
  std::copy(in_c, in_c+nloc*3, coord.begin());
  std::copy(in_t, in_t+nloc, atype.begin());
  SimulationRegion<double> tmpr;
  double tmp_boxt[9];
  std::copy(region.boxt, region.boxt+9, tmp_boxt);
  tmpr.reinitBox(tmp_boxt);
  
  std::vector<double > out_coord;
  std::vector<int> out_atype, out_mapping, ncell, ngcell;
  copy_coord(out_coord, out_atype, out_mapping, ncell, ngcell, coord, atype, rcut, tmpr);
  
  *nall = out_atype.size();
  if(*nall > mem_nall){
    // size of the output arrays is not large enough
    return 1;
  }
  else{
    std::copy(out_coord.begin(), out_coord.end(), out_c);
    std::copy(out_atype.begin(), out_atype.end(), out_t);
    std::copy(out_mapping.begin(), out_mapping.end(), mapping);
  }
  return 0;
}

template <typename FPTYPE>
void
deepmd::
compute_cell_info(
    int * cell_info, //nat_stt,ncell,ext_stt,ext_end,ngcell,cell_shift,cell_iter,loc_cellnum,total_cellnum
    const float & rcut,
    const Region<FPTYPE> & region)
{
  SimulationRegion<double> tmpr;
	double to_face [3];
  double tmp_boxt[9];
  std::copy(region.boxt, region.boxt+9, tmp_boxt);
	tmpr.reinitBox(tmp_boxt);
	tmpr.toFaceDistance (to_face);
  double cell_size [3];
  for (int dd = 0; dd < 3; ++dd){
    cell_info[dd]=0; //nat_stt
    cell_info[3+dd]  = to_face[dd] / rcut; //ncell
    if (cell_info[3+dd] == 0) cell_info[3+dd] = 1;
    cell_size[dd] = to_face[dd] / cell_info[3+dd]; 
    cell_info[12+dd] = int(rcut / cell_size[dd]) + 1; //ngcell
    cell_info[6+dd]=-cell_info[12+dd]; //ext_stt
    cell_info[9+dd]=cell_info[3+dd]+cell_info[12+dd]; //ext_end
    cell_info[15+dd]=cell_info[12+dd]; //cell_shift
    cell_info[18+dd]= rcut / cell_size[dd]; //cell_iter
    if (cell_info[18+dd] * cell_size[dd] < rcut) cell_info[18+dd] += 1;
  }
  cell_info[21] = (cell_info[3+0]) * (cell_info[3+1]) * (cell_info[3+2]); //loc_cellnum
  cell_info[22] = (2 * cell_info[12+0] + cell_info[3+0]) * (2 * cell_info[12+1] + cell_info[3+1]) * (2 * cell_info[12+2] + cell_info[3+2]); //total_cellnum
}

template
void
deepmd::
normalize_coord_cpu<double>(
    double * coord,
    const int natom,
    const deepmd::Region<double> & region);

template
void
deepmd::
normalize_coord_cpu<float>(
    float * coord,
    const int natom,
    const deepmd::Region<float> & region);

template
int
deepmd::
copy_coord_cpu<double>(
    double * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    const double * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall,
    const float & rcut,
    const deepmd::Region<double> & region);

template
int
deepmd::
copy_coord_cpu<float>(
    float * out_c,
    int * out_t,
    int * mapping,
    int * nall,
    const float * in_c,
    const int * in_t,
    const int & nloc,
    const int & mem_nall,
    const float & rcut,
    const deepmd::Region<float> & region);

template
void
deepmd::
compute_cell_info<double>(
    int * cell_info, //nat_stt,ncell,ext_stt,ext_end,ngcell,cell_shift,cell_iter,loc_cellnum,total_cellnum
    const float & rcut,
    const Region<double> & region);

template
void
deepmd::
compute_cell_info<float>(
    int * cell_info, //nat_stt,ncell,ext_stt,ext_end,ngcell,cell_shift,cell_iter,loc_cellnum,total_cellnum
    const float & rcut,
    const Region<float> & region);



