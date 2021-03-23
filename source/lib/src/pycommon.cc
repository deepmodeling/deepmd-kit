#include "pycommon.h"
#include "NNPAtomMap.h"
#include "SimulationRegion.h"


using namespace std;


void 
select_by_type(vector<int> & fwd_map,
	       vector<int> & bkw_map,
	       int & nghost_real, 
	       const vector<VALUETYPE> & dcoord_, 
	       const vector<int> & datype_,
	       const int & nghost,
	       const vector<int> & sel_type_)
{
  vector<int> sel_type (sel_type_);
  sort(sel_type.begin(), sel_type.end());  
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  int nloc_real = 0;
  nghost_real = 0;
  fwd_map.resize(nall);
  bkw_map.clear();
  bkw_map.reserve(nall);  
  int cc = 0;
  for (int ii = 0; ii < nall; ++ii){
    // exclude virtual sites
    // select the type with id < ntypes
    if (binary_search(sel_type.begin(), sel_type.end(), datype_[ii])){
      bkw_map.push_back(ii);
      if (ii < nloc) {
	nloc_real += 1;
      }
      else{
	nghost_real += 1;
      }
      fwd_map[ii] = cc;
      cc ++;
    }
    else{
      fwd_map[ii] = -1;
    }
  }  
  assert((nloc_real+nghost_real) == bkw_map.size());  
}	       


void
select_real_atoms(vector<int> & fwd_map,
		  vector<int> & bkw_map,
		  int & nghost_real,
		  const vector<VALUETYPE> & dcoord_, 
		  const vector<int> & datype_,
		  const int & nghost,
		  const int & ntypes)
{
  vector<int > sel_type;
  for (int ii = 0; ii < ntypes; ++ii){
    sel_type.push_back(ii);
  }
  select_by_type(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, sel_type);
}

void
convert_nlist_lmp_internal (InternalNeighborList & list,
			    const LammpsNeighborList & lmp_list) 
{
  list.clear();
  int total_num_nei = 0;
  int inum = lmp_list.inum;
  for (int ii = 0; ii < inum; ++ii){
    total_num_nei += lmp_list.numneigh[ii];
  }
  list.ilist.resize(inum);
  list.jrange.resize(inum+1);
  list.jlist.resize(total_num_nei);
  memcpy(&list.ilist[0], lmp_list.ilist, inum*sizeof(int));
  list.jrange[0] = 0;
  for (int ii = 0; ii < inum; ++ii){
    int jnum = lmp_list.numneigh[ii];
    list.jrange[ii+1] = list.jrange[ii] + jnum;
    const int * jlist = lmp_list.firstneigh[ii];
    memcpy(&(list.jlist[list.jrange[ii]]), jlist, jnum*sizeof(int));
  }
}

void
shuffle_nlist (InternalNeighborList & list, 
	       const NNPAtomMap<VALUETYPE> & map)
{
  const vector<int> & fwd_map = map.get_fwd_map();
  shuffle_nlist(list, fwd_map);
}

void
shuffle_nlist (InternalNeighborList & list, 
	       const vector<int> & fwd_map)
{
  int nloc = fwd_map.size();
  for (unsigned ii = 0; ii < list.ilist.size(); ++ii){
    if (list.ilist[ii] < nloc) {
      list.ilist[ii] = fwd_map[list.ilist[ii]];
    }
  }
  for (unsigned ii = 0; ii < list.jlist.size(); ++ii){
    if (list.jlist[ii] < nloc) {
      list.jlist[ii] = fwd_map[list.jlist[ii]];
    }
  }
}

void
shuffle_nlist_exclude_empty (InternalNeighborList & list, 
			     const vector<int> & fwd_map)
{
  int old_nloc = fwd_map.size();
  shuffle_nlist(list, fwd_map);
  vector<int> new_ilist, new_jrange, new_jlist, new_icount;
  new_ilist.reserve(list.ilist.size());
  new_icount.reserve(list.ilist.size());
  new_jrange.reserve(list.jrange.size());
  new_jlist.reserve(list.jlist.size());
  for(int ii = 0; ii < list.ilist.size(); ++ii){
    if(list.ilist[ii] >= 0){
      new_ilist.push_back(list.ilist[ii]);
    }
  }
  new_jrange.resize(new_ilist.size()+1);
  new_jrange[0] = 0;
  int ci = 0;
  for(int ii = 0; ii < list.ilist.size(); ++ii){
    if (list.ilist[ii] < 0) continue;
    int js = list.jrange[ii];
    int je = list.jrange[ii+1];
    int cc = 0;
    for (int jj = js; jj < je; ++jj){
      if (list.jlist[jj] >= 0) {
	new_jlist.push_back(list.jlist[jj]);
	cc++;
      }      
    }
    new_jrange[ci+1] = new_jrange[ci] + cc;
    ci ++;
  }
  list.ilist = new_ilist;
  list.jrange = new_jrange;
  list.jlist = new_jlist;
}




string
name_prefix(const string & scope)
{
  string prefix = "";
  if (scope != ""){
    prefix = scope + "/";
  }
  return prefix;
}