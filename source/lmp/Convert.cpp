#include "Convert.h"

#include <algorithm>
#include <cassert>

Convert::
Convert(const vector<int > & lmp_type)
{
  int natoms = lmp_type.size();  
  atype.resize (natoms);
  for (unsigned ii = 0; ii < atype.size(); ++ii){
    atype[ii] = lmp_type[ii] - 1;
  }
  vector<pair<int, int > > sorting (natoms);
  for (unsigned ii = 0; ii < sorting.size(); ++ii){
    sorting[ii] = pair<int, int > (atype[ii], ii);
  }
  sort (sorting.begin(), sorting.end());
  idx_map.resize(natoms);
  for (unsigned ii = 0; ii < idx_map.size(); ++ii){
    idx_map[ii] = sorting[ii].second;
    atype[ii] = sorting[ii].first;
  }
}

void
Convert::
forward (vector<double > & out,
	 const vector<double > & in, 
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

void
Convert::
backward (vector<double > & out,
	  const vector<double > & in,
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
