#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "version.h"
#include "neighbor_list.h"
#include "AtomMap.h"
#include "string.h"
#include <numeric>

namespace deepmd{

#if TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION >= 2
typedef tensorflow::tstring STRINGTYPE;
#else
typedef std::string STRINGTYPE;
#endif

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
// typedef float  VALUETYPE;
// typedef float ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
// typedef float ENERGYTYPE;
#endif

struct NeighborListData 
{
  std::vector<int > ilist;
  std::vector<std::vector<int> > jlist;
  std::vector<int > numneigh;
  std::vector<int* > firstneigh;  
public:
  void copy_from_nlist(const InputNlist & inlist);
  void shuffle(const std::vector<int> & fwd_map);
  void shuffle(const deepmd::AtomMap<VALUETYPE> & map);
  void shuffle_exclude_empty(const std::vector<int> & fwd_map);
  void make_inlist(InputNlist & inlist);
};

bool
model_compatable(
    std::string & model_version);

void 
select_by_type(std::vector<int> & fwd_map,
	       std::vector<int> & bkw_map,
	       int & nghost_real, 
	       const std::vector<VALUETYPE> & dcoord_, 
	       const std::vector<int> & datype_,
	       const int & nghost,
	       const std::vector<int> & sel_type_);

void
select_real_atoms(std::vector<int> & fwd_map,
		  std::vector<int> & bkw_map,
		  int & nghost_real,
		  const std::vector<VALUETYPE> & dcoord_, 
		  const std::vector<int> & datype_,
		  const int & nghost,
		  const int & ntypes);

template<typename VT>
void 
select_map(std::vector<VT> & out,
	   const std::vector<VT > & in,
	   const std::vector<int > & fwd_map, 
	   const int & stride);

void
get_env_nthreads(int & num_intra_nthreads,
		 int & num_inter_nthreads);

std::string 
name_prefix(
    const std::string & name_scope);

}

