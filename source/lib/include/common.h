#pragma once

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>

using namespace tensorflow;
using namespace std;

#if TF_MAJOR_VERSION >= 2 && TF_MINOR_VERSION >= 2
typedef tensorflow::tstring STRINGTYPE;
#else
typedef std::string STRINGTYPE;
#endif

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "pycommon.h"


void
get_env_nthreads(int & num_intra_nthreads,
		 int & num_inter_nthreads);

void
checkStatus(const tensorflow::Status& status);


template<typename VT>
VT
session_get_scalar(Session* session, const string name, const string scope = "");

template<typename VT>
void
session_get_vector(vector<VT> & o_vec, Session* session, const string name_, const string scope = "");

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
		       const int			nghost = 0,
		       const string			scope = "");

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
		       const int			nghost,
		       const int			ago,
		       const string			scope = "");

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
		       const int			nghost,
		       const string			scope = "");

int 
session_input_tensors (vector<std::pair<string, Tensor>>& input_tensors,
		       const vector<VALUETYPE>          & dcoord_,
		       const int                        & ntypes,
		       const vector<int>                & atype_,
		       const vector<VALUETYPE>          & dbox,
		       const int                        * ilist, 
		       const int                        * jrange,
		       const int                        * jlist,
		       const vector<VALUETYPE>		& fparam_,
		       const vector<VALUETYPE>	        & aparam_,
		       const NNPAtomMap<VALUETYPE>      & nnpmap,
		       const int			& nghost);


template<typename VT>
VT
session_get_scalar(Session* session, const string name_, const string scope) 
{
  string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(std::vector<std::pair<string, Tensor>> ({}), 
			    {name.c_str()}, 
			    {}, 
			    &output_tensors));
  Tensor output_rc = output_tensors[0];
  auto orc = output_rc.flat <VT> ();
  return orc(0);
}

template<typename VT>
void
session_get_vector(vector<VT> & o_vec, Session* session, const string name_, const string scope) 
{
  string name = name_;
  if (scope != "") {
    name = scope + "/" + name;
  }
  std::vector<Tensor> output_tensors;
  checkStatus (session->Run(std::vector<std::pair<string, Tensor>> ({}), 
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
