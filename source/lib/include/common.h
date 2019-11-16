#pragma once

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using namespace std;

#include "NNPAtomMap.h"
#include <vector>
#include "version.h"

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
	       const NNPAtomMap<VALUETYPE> & map);

void
get_env_nthreads(int & num_intra_nthreads,
		 int & num_inter_nthreads);

void
checkStatus(const tensorflow::Status& status);

template<class VT>
VT
session_get_scalar(Session* session, const string name);

int
session_input_tensors (std::vector<std::pair<string, Tensor>> & input_tensors,
		       const vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const vector<int> &		datype_,
		       const vector<VALUETYPE> &	dbox, 
		       const VALUETYPE &		cell_size,
		       const vector<VALUETYPE> &	fparam_,
		       const vector<VALUETYPE> &	aparam_,
		       const NNPAtomMap<VALUETYPE>&	nnpmap,
		       const int			nghost = 0);

int
session_input_tensors (std::vector<std::pair<string, Tensor>> & input_tensors,
		       const vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const vector<int> &		datype_,
		       const vector<VALUETYPE> &	dbox,		    
		       InternalNeighborList &		dlist, 
		       const vector<VALUETYPE> &	fparam_,
		       const vector<VALUETYPE> &	aparam_,
		       const NNPAtomMap<VALUETYPE>&	nnpmap,
		       const int			nghost);


template<class VT>
VT
session_get_scalar(Session* session, const string name) 
{
  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(std::vector<std::pair<string, Tensor>> ({}), 
			    {name.c_str()}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  auto orc = output_rc.flat <VT> ();
  return orc(0);
}
