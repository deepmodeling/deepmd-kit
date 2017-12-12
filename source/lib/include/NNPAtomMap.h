#pragma once

#include <vector>

using namespace std;

template <typename VALUETYPE>
class NNPAtomMap 
{
public:
  NNPAtomMap (const vector<int > & in_atom_type);
  void forward (vector<VALUETYPE > & out,
		const vector<VALUETYPE > & in, 
		const int stride = 1) const ;
  void backward (vector<VALUETYPE > & out,
		 const vector<VALUETYPE > & in,
		 const int stride = 1) const ;
  const vector<int > & get_type () const {return atype;}
private:
  vector<int> idx_map;
  vector<int> atype;
};
