#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "version.h"
#include "NNPAtomMap.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

using namespace tensorflow;
#if TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION >= 2
typedef tensorflow::tstring STRINGTYPE;
#else
typedef std::string STRINGTYPE;
#endif

#ifdef HIGH_PREC
typedef double VALUETYPE;
typedef double ENERGYTYPE;
#else 
typedef float  VALUETYPE;
typedef double ENERGYTYPE;
#endif

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
  std::vector<int > ilist;
  std::vector<int > jrange;
  std::vector<int > jlist;
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
	       const std::vector<int> & fwd_map);

void
shuffle_nlist (InternalNeighborList & list, 
	       const NNPAtomMap<VALUETYPE> & map);

void
shuffle_nlist_exclude_empty (InternalNeighborList & list, 
			     const std::vector<int> & fwd_map);


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

void
checkStatus(const tensorflow::Status& status);

std::string name_prefix(const std::string & name_scope);

template<typename VT>
VT
session_get_scalar(Session* session, const std::string name, const std::string scope = "");

template<typename VT>
void
session_get_vector(std::vector<VT> & o_vec, Session* session, const std::string name_, const std::string scope = "");

int
session_input_tensors (std::vector<std::pair<std::string, Tensor>> & input_tensors,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox, 
		       const VALUETYPE &		cell_size,
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const NNPAtomMap<VALUETYPE>&	nnpmap,
		       const int			nghost = 0,
		       const std::string			scope = "");

int
session_input_tensors (std::vector<std::pair<std::string, Tensor>> & input_tensors,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox,		    
		       InternalNeighborList &		dlist, 
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const NNPAtomMap<VALUETYPE>&	nnpmap,
		       const int			nghost,
		       const int			ago,
		       const std::string			scope = "");

int
session_input_tensors (std::vector<std::pair<std::string, Tensor>> & input_tensors,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox,		    
		       InternalNeighborList &		dlist, 
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const NNPAtomMap<VALUETYPE>&	nnpmap,
		       const int			nghost,
		       const std::string			scope = "");

int 
session_input_tensors (std::vector<std::pair<std::string, Tensor>>& input_tensors,
		       const std::vector<VALUETYPE>          & dcoord_,
		       const int                        & ntypes,
		       const std::vector<int>                & atype_,
		       const std::vector<VALUETYPE>          & dbox,
		       const int                        * ilist, 
		       const int                        * jrange,
		       const int                        * jlist,
		       const std::vector<VALUETYPE>		& fparam_,
		       const std::vector<VALUETYPE>	        & aparam_,
		       const NNPAtomMap<VALUETYPE>      & nnpmap,
		       const int			& nghost);


template<typename VT>
VT
session_get_scalar(Session* session, const std::string name_, const std::string scope) 
{
  std::string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(std::vector<std::pair<std::string, Tensor>> ({}), 
			    {name.c_str()}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  auto orc = output_rc.flat <VT> ();
  return orc(0);
}

template<typename VT>
void
session_get_vector(std::vector<VT> & o_vec, Session* session, const std::string name_, const std::string scope) 
{
  std::string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(std::vector<std::pair<std::string, Tensor>> ({}), 
			    {name.c_str()}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  assert(1 == output_rc.shape().dims());
  int dof = output_rc.shape().dim_size(0);
  o_vec.resize(dof);
  auto orc = output_rc.flat <VT> ();
  for (int ii = 0; ii < dof; ++ii){
    o_vec[ii] = orc(ii);
  }  
}


template<typename VT>
void 
select_map(std::vector<VT> & out,
	   const std::vector<VT > & in,
	   const std::vector<int > & idx_map, 
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

