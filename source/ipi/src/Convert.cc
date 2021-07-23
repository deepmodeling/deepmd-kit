#include "Convert.h"

#include <algorithm>
#include <cassert>

template <typename VALUETYPE>
Convert<VALUETYPE>::
Convert(const std::vector<std::string > &  atomname,
	std::map<std::string, int> & name_type_map)
{
  int natoms = atomname.size();
  atype.resize (natoms);
  for (unsigned ii = 0; ii < atype.size(); ++ii){
    atype[ii] = name_type_map[atomname[ii]];
  }
  std::vector<std::pair<int, int > > sorting (natoms);
  for (unsigned ii = 0; ii < sorting.size(); ++ii){
    sorting[ii] = std::pair<int, int > (atype[ii], ii);
  }
  // sort (sorting.begin(), sorting.end());
  idx_map.resize(natoms);
  for (unsigned ii = 0; ii < idx_map.size(); ++ii){
    idx_map[ii] = sorting[ii].second;
    atype[ii] = sorting[ii].first;
  }
}

template <typename VALUETYPE>
void
Convert<VALUETYPE>::
forward (std::vector<VALUETYPE > & out,
	 const std::vector<double > & in, 
	 const int stride) const 
{
  assert (in.size() == stride * idx_map.size());
  int natoms = idx_map.size();
  out.resize (stride * natoms);
  for (int ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd){
      out[ii*stride+dd] = in[gro_i*stride+dd];
    }
  }
}

template <typename VALUETYPE>
void
Convert<VALUETYPE>::
backward (std::vector<VALUETYPE > & out,
	  const std::vector<double > & in,
	  const int stride) const 
{
  int natoms = idx_map.size();
  assert (in.size() == stride * idx_map.size());
  out.resize(stride * natoms);
  for (int ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd){
      out[gro_i*stride+dd] = in[ii*stride+dd];
    }
  }
}

template class Convert<float>;
template class Convert<double>;

