#pragma once

#include <vector>

// using namespace std;

namespace deepmd{
template <typename VALUETYPE>
class AtomMap 
{
public:
  AtomMap();
  AtomMap(const std::vector<int >::const_iterator in_begin, 
	     const std::vector<int >::const_iterator in_end);
  void forward (typename std::vector<VALUETYPE >::iterator out,
		const typename std::vector<VALUETYPE >::const_iterator in, 
		const int stride = 1) const ;
  void backward (typename std::vector<VALUETYPE >::iterator out,
		 const typename std::vector<VALUETYPE >::const_iterator in, 
		 const int stride = 1) const ;
  const std::vector<int > & get_type () const {return atype;}
  const std::vector<int > & get_fwd_map () const {return fwd_idx_map;}
  const std::vector<int > & get_bkw_map () const {return idx_map;}
private:
  std::vector<int> idx_map;
  std::vector<int> fwd_idx_map;
  std::vector<int> atype;
};
}
