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
	  map<string, int> & name_type_map,
	  map<string, VALUETYPE> & name_mass_map,
	  map<string, VALUETYPE> & name_charge_map, 
	  const bool sort = true);
  void gro2nnp (
      vector<VALUETYPE > & coord,
      vector<VALUETYPE > & veloc,
      vector<VALUETYPE > & box,
      const vector<vector<double > > & posi,
      const vector<vector<double > > & velo,
      const vector<double > & box_size) const ;
  void nnp2gro (
      vector<vector<double > > & posi,
      vector<vector<double > > & velo,
      vector<double > & box_size,
      const vector<VALUETYPE > & coord,
      const vector<VALUETYPE > & veloc,
      const vector<VALUETYPE > & box) const ;
  void idx_gro2nnp (
      vector<int > & out,
      const vector<int > & in) const;
  void idx_nnp2gro (
      vector<int > & out,
      const vector<int > & in) const;      
  const vector<int > & get_type () const {return atype;}
  const vector<VALUETYPE > & get_mass () const {return amass;}
  const vector<VALUETYPE > & get_charge () const {return acharge;}
private:
  vector<int> idx_map_nnp2gro;
  vector<int> idx_map_gro2nnp;
  vector<int> atype;
  vector<VALUETYPE> amass;
  vector<VALUETYPE> acharge;
};
