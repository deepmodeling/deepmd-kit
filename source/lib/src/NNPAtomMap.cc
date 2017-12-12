#include "NNPAtomMap.h"

#include <algorithm>
#include <cassert>

template <typename VALUETYPE>
NNPAtomMap<VALUETYPE>::
NNPAtomMap(const vector<int > & in_atom_type)
{
  int natoms = in_atom_type.size();
  atype.resize (natoms);
  vector<pair<int, int > > sorting (natoms);
  for (unsigned ii = 0; ii < sorting.size(); ++ii){
    sorting[ii] = pair<int, int > (in_atom_type[ii], ii);
  }
  sort (sorting.begin(), sorting.end());
  idx_map.resize(natoms);
  for (unsigned ii = 0; ii < idx_map.size(); ++ii){
    idx_map[ii] = sorting[ii].second;
    atype[ii] = sorting[ii].first;
  }
}

template <typename VALUETYPE>
void
NNPAtomMap<VALUETYPE>::
forward (vector<VALUETYPE > & out,
	 const vector<VALUETYPE > & in, 
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
NNPAtomMap<VALUETYPE>::
backward (vector<VALUETYPE > & out,
	  const vector<VALUETYPE > & in,
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

template class NNPAtomMap<float>;
template class NNPAtomMap<double>;

