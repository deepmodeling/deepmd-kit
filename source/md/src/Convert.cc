#include "Convert.h"

#include <algorithm>
#include <cassert>
#include <iostream>

template <typename VALUETYPE>
Convert<VALUETYPE>::
Convert(const vector<string > &  atomname,
	map<string, int> & name_type_map,
	map<string, VALUETYPE> & name_mass_map,
	map<string, VALUETYPE> & name_charge_map, 
	const bool sort_)
{
  int natoms = atomname.size();
  atype.resize (natoms);
  amass.resize (natoms);
  vector<VALUETYPE> tmp_charge (natoms);
  for (unsigned ii = 0; ii < atype.size(); ++ii){
    atype[ii] = name_type_map[atomname[ii]];
    amass[ii] = name_mass_map[atomname[ii]];
    tmp_charge[ii] = name_charge_map[atomname[ii]];
  }
  vector<pair<int, pair<int, VALUETYPE> > > sorting (natoms);
  for (unsigned ii = 0; ii < sorting.size(); ++ii){
    sorting[ii] = pair<int, pair<int, VALUETYPE> > (atype[ii], pair<int, VALUETYPE> (ii, amass[ii]));
  }
  if (sort_) sort (sorting.begin(), sorting.end());
  idx_map_nnp2gro.resize(natoms);
  idx_map_gro2nnp.resize(natoms);
  for (unsigned ii = 0; ii < idx_map_nnp2gro.size(); ++ii){
    idx_map_nnp2gro[ii] = sorting[ii].second.first;
    idx_map_gro2nnp[sorting[ii].second.first] = ii;
    atype[ii] = sorting[ii].first;
    amass[ii] = sorting[ii].second.second;
  }
  acharge.resize (natoms);
  for (int ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map_nnp2gro[ii];
    acharge[ii] = tmp_charge[gro_i];
  }
}

template <typename VALUETYPE>
void
Convert<VALUETYPE>::
gro2nnp (vector<VALUETYPE > & coord,
	 vector<VALUETYPE > & veloc,
	 vector<VALUETYPE > & box,
	 const vector<vector<double > > & posi,
	 const vector<vector<double > > & velo,
	 const vector<double > & box_size) const
{
  assert (posi.size() == idx_map_nnp2gro.size());
  assert (velo.size() == idx_map_nnp2gro.size());
  int natoms = idx_map_nnp2gro.size();
  coord.resize (3 * natoms);
  veloc.resize (3 * natoms);
  for (unsigned ii = 0; ii < natoms; ++ii){
    int gro_i = idx_map_nnp2gro[ii];
    for (int dd = 0; dd < 3; ++dd){
      coord[ii*3+dd] = posi[gro_i][dd] * 10;
      veloc[ii*3+dd] = velo[gro_i][dd] * 10;
    }
  }
  box.resize(9);
  for (int dd = 0; dd < 9; ++dd){
    box[dd] = box_size[dd] * 10;
  }
}

template <typename VALUETYPE>
void
Convert<VALUETYPE>::
nnp2gro (vector<vector<double > > & posi,
	 vector<vector<double > > & velo,
	 vector<double > & box_size,
	 const vector<VALUETYPE > & coord,
	 const vector<VALUETYPE > & veloc,
	 const vector<VALUETYPE > & box) const
{
  int natoms = idx_map_nnp2gro.size();
  posi.resize(natoms);
  velo.resize(natoms);
  for (unsigned ii = 0; ii < posi.size(); ++ii){
    posi[ii].resize(3);
    velo[ii].resize(3);
  }
  for (unsigned ii = 0; ii < posi.size(); ++ii){
    int gro_i = idx_map_nnp2gro[ii];
    for (int dd = 0; dd < 3; ++dd){
      posi[gro_i][dd] = coord[ii*3+dd] * 0.1;
      velo[gro_i][dd] = veloc[ii*3+dd] * 0.1;
    }
  }
  box_size.resize(9);
  for (int dd = 0; dd < 9; ++dd){
    box_size[dd] = box[dd] * 0.1;
  }
}

template <typename VALUETYPE>
void
Convert<VALUETYPE>::
idx_gro2nnp (vector<int > & out,
	     const vector<int > & in) const
{
  for (unsigned ii = 0; ii < in.size(); ++ii){
    out[ii] = idx_map_gro2nnp[in[ii]];
  }
}

template <typename VALUETYPE>
void
Convert<VALUETYPE>::
idx_nnp2gro (vector<int > & out,
	     const vector<int > & in) const
{
  for (unsigned ii = 0; ii < in.size(); ++ii){
    out[ii] = idx_map_nnp2gro[in[ii]];
  }
}

template class Convert<float>;
template class Convert<double>;

