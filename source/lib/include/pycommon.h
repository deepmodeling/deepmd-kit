#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <memory.h>

#include "NNPAtomMap.h"
#include "version.h"


// #define PROF
#ifdef PROF
#include <fj_tool/fipp.h>
#include <fj_tool/fapp.h>
#endif


#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
#endif

using std::vector;
using std::string;
using std::cout;
using std::endl;

struct LammpsNeighborList 
{
  int inum;
  const int * ilist;
  const int * numneigh;
  const int *const* firstneigh;
  LammpsNeighborList (int inum_, 
		      const int * ilist_,
		      const int * numneigh_, 
		      const int *const* firstneigh_) 
      : inum(inum_), ilist(ilist_), numneigh(numneigh_), firstneigh(firstneigh_)
      {
      }
};

struct InternalNeighborList 
{
  int * pilist;
  int * pjrange;
  int * pjlist;
  vector<int > ilist;
  vector<int > jrange;
  vector<int > jlist;
  void clear () {ilist.clear(); jrange.clear(); jlist.clear();}
  void make_ptrs () {
    pilist = &ilist[0]; pjrange = &jrange[0]; pjlist = &jlist[0];
  }
};

void
convert_nlist_lmp_internal (InternalNeighborList & list,
			    const LammpsNeighborList & lmp_list);

void
shuffle_nlist (InternalNeighborList & list, 
	       const vector<int> & fwd_map);

void
shuffle_nlist (InternalNeighborList & list, 
	       const NNPAtomMap<VALUETYPE> & map);

void
shuffle_nlist_exclude_empty (InternalNeighborList & list, 
			     const vector<int> & fwd_map);


void 
select_by_type(vector<int> & fwd_map,
	       vector<int> & bkw_map,
	       int & nghost_real, 
	       const vector<VALUETYPE> & dcoord_, 
	       const vector<int> & datype_,
	       const int & nghost,
	       const vector<int> & sel_type_);

void
select_real_atoms(vector<int> & fwd_map,
		  vector<int> & bkw_map,
		  int & nghost_real,
		  const vector<VALUETYPE> & dcoord_, 
		  const vector<int> & datype_,
		  const int & nghost,
		  const int & ntypes);

template<typename VT>
void 
select_map(vector<VT> & out,
	   const vector<VT > & in,
	   const vector<int > & fwd_map, 
	   const int & stride);




string name_prefix(const string & name_scope);


template<typename VT>
void 
select_map(vector<VT> & out,
	   const vector<VT > & in,
	   const vector<int > & idx_map, 
	   const int & stride)
{
#ifdef DEBUG
  assert(in.size() / stride * stride == in.size()), "in size should be multiples of stride"
#endif
  for (int ii = 0; ii < in.size() / stride; ++ii){
#ifdef DEBUG
    assert(ii < idx_map.size()), "idx goes over the idx map size";
    assert(idx_map[ii] < out.size()), "mappped idx goes over the out size";
#endif
    if (idx_map[ii] >= 0) {
      int to_ii = idx_map[ii];
      for (int dd = 0; dd < stride; ++dd){
	out[to_ii * stride + dd] = in[ii * stride + dd];
      }
    }
  }
}
