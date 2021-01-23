#include "NNPAtomMap.h"

#include <algorithm>
#include <cassert>

template <typename VALUETYPE>
NNPAtomMap<VALUETYPE>::
NNPAtomMap() {}

template <typename VALUETYPE>
NNPAtomMap<VALUETYPE>::
NNPAtomMap(const vector<int >::const_iterator in_begin, 
	   const vector<int >::const_iterator in_end)
{
  int natoms = in_end - in_begin;
  atype.resize (natoms);
  vector<pair<int, int > > sorting (natoms);
  vector<int >::const_iterator iter = in_begin;
  for (unsigned ii = 0; ii < sorting.size(); ++ii){
    sorting[ii] = pair<int, int > (*(iter++), ii);
  }
  sort (sorting.begin(), sorting.end());
  idx_map.resize(natoms);
  fwd_idx_map.resize(natoms);
  for (unsigned ii = 0; ii < idx_map.size(); ++ii){
    idx_map[ii] = sorting[ii].second;
    fwd_idx_map[sorting[ii].second] = ii;
    atype[ii] = sorting[ii].first;
  }
}

template <typename VALUETYPE>
void
NNPAtomMap<VALUETYPE>::
forward (typename vector<VALUETYPE >::iterator out,
	 const typename vector<VALUETYPE >::const_iterator in, 
	 const int stride) const 
{
  int natoms = idx_map.size();
  for (int ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd){
      // out[ii*stride+dd] = in[gro_i*stride+dd];
      *(out + ii*stride + dd) = *(in + gro_i*stride + dd);
    }
  }
}

template <typename VALUETYPE>
void
NNPAtomMap<VALUETYPE>::
backward (typename vector<VALUETYPE >::iterator out,
	  const typename vector<VALUETYPE >::const_iterator in, 
	  const int stride) const 
{
  int natoms = idx_map.size();
  for (int ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map[ii];
    for (int dd = 0; dd < stride; ++dd){
      // out[gro_i*stride+dd] = in[ii*stride+dd];
      *(out + gro_i*stride + dd) = *(in + ii*stride + dd);
    }
  }
}

template class NNPAtomMap<float>;
template class NNPAtomMap<double>;

