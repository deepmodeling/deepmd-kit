#pragma once

#include <vector>
#include <string>
#include <map>

// using namespace std;

template <typename VALUETYPE>
class Convert 
{
public:
  Convert(const std::vector<std::string > &  atomname,
	  std::map<std::string, int> & name_type_map);
  void forward (
      std::vector<VALUETYPE > & out,
      const std::vector<double > & in, 
      const int stride = 1) const ;
  void backward (
      std::vector<VALUETYPE > & out,
      const std::vector<double > & in,
      const int stride = 1) const ;
  const std::vector<int > & get_type () const {return atype;}
private:
  std::vector<int> idx_map;
  std::vector<int> atype;
};
