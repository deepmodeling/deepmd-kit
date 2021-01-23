#pragma once

#include <vector>
#include <string>
#include <map>

using namespace std;

template <typename VALUETYPE>
class Convert 
{
public:
  Convert(const vector<string > &  atomname,
	  map<string, int> & name_type_map);
  void forward (
      vector<VALUETYPE > & out,
      const vector<double > & in, 
      const int stride = 1) const ;
  void backward (
      vector<VALUETYPE > & out,
      const vector<double > & in,
      const int stride = 1) const ;
  const vector<int > & get_type () const {return atype;}
private:
  vector<int> idx_map;
  vector<int> atype;
};
