#pragma once

#include <vector>
#include <string>
#include <map>

using namespace std;

class Convert 
{
public:
  Convert(const vector<int > & lmp_type);
  void forward (
      vector<double > & out,
      const vector<double > & in, 
      const int stride = 1) const ;
  void backward (
      vector<double > & out,
      const vector<double > & in,
      const int stride = 1) const ;
  const vector<int > & get_type () const {return atype;}
private:
  vector<int> idx_map;
  vector<int> atype;
};


