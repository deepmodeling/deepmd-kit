#pragma once

#include <vector>

using namespace std;

template <typename VALUETYPE>
class NNPAtomMap 
{
public:
  NNPAtomMap();
  NNPAtomMap(const vector<int >::const_iterator in_begin, 
	     const vector<int >::const_iterator in_end);
  void forward (typename vector<VALUETYPE >::iterator out,
		const typename vector<VALUETYPE >::const_iterator in, 
		const int stride = 1) const ;
  void backward (typename vector<VALUETYPE >::iterator out,
		 const typename vector<VALUETYPE >::const_iterator in, 
		 const int stride = 1) const ;
  const vector<int > & get_type () const {return atype;}
  const vector<int > & get_fwd_map () const {return fwd_idx_map;}
private:
  vector<int> idx_map;
  vector<int> fwd_idx_map;
  vector<int> atype;
};
