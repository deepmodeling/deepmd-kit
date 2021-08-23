#pragma once

#include <vector>
#include <string>
#include <iostream>
#include "version.h"
#include "neighbor_list.h"
#include "AtomMap.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/version.h"
#include <tensorflow/core/graph/default_device.h>
#include <tensorflow/core/graph/graph_def_builder.h>


namespace deepmd{

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

struct NeighborListData 
{
  /// Array stores the core region atom's index
  std::vector<int > ilist;
  /// Array stores the core region atom's neighbor index
  std::vector<std::vector<int> > jlist;
  /// Array stores the number of neighbors of core region atoms
  std::vector<int > numneigh;
  /// Array stores the the location of the first neighbor of core region atoms
  std::vector<int* > firstneigh;  
public:
  void copy_from_nlist(const InputNlist & inlist);
  void shuffle(const std::vector<int> & fwd_map);
  void shuffle(const deepmd::AtomMap<VALUETYPE> & map);
  void shuffle_exclude_empty(const std::vector<int> & fwd_map);
  void make_inlist(InputNlist & inlist);
};

/** @struct deepmd::InputNlist
 **/

/**
* @brief Check if the model version is supported.
* @param[in] model_version The model version.
* @return Whether the model is supported (true or false).
**/
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

template<typename VT>
void 
select_map(typename std::vector<VT >::iterator out,
	   const typename std::vector<VT >::const_iterator in, 
	   const std::vector<int > & fwd_map, 
	   const int & stride);

template<typename VT>
void 
select_map_inv(std::vector<VT> & out,
	   const std::vector<VT > & in,
	   const std::vector<int > & fwd_map, 
	   const int & stride);

template<typename VT>
void 
select_map_inv(typename std::vector<VT >::iterator out,
	   const typename std::vector<VT >::const_iterator in, 
	   const std::vector<int > & fwd_map, 
	   const int & stride);

/**
* @brief Get the number of threads from the environment variable.
* @param[out] num_intra_nthreads The number of intra threads. Read from TF_INTRA_OP_PARALLELISM_THREADS.
* @param[out] num_inter_nthreads The number of inter threads. Read from TF_INTER_OP_PARALLELISM_THREADS.
**/
void
get_env_nthreads(int & num_intra_nthreads,
		 int & num_inter_nthreads);

struct
tf_exception: public std::exception {
};

/**
* @brief Check TensorFlow status. Exit if not OK.
* @param[in] status TensorFlow status.
**/
void
check_status(
    const tensorflow::Status& status);

std::string 
name_prefix(
    const std::string & name_scope);

template<typename VT>
VT
session_get_scalar(
    tensorflow::Session* session, 
    const std::string name, 
    const std::string scope = "");

template<typename VT>
void
session_get_vector(
    std::vector<VT> & o_vec, 
    tensorflow::Session* session, 
    const std::string name_, 
    const std::string scope = "");

int
session_input_tensors (std::vector<std::pair<std::string, tensorflow::Tensor>> & input_tensors,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox, 
		       const VALUETYPE &		cell_size,
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const deepmd::AtomMap<VALUETYPE>&atommap,
		       const std::string		scope = "");

int
session_input_tensors (std::vector<std::pair<std::string, tensorflow::Tensor>> & input_tensors,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox,		    
		       InputNlist &		dlist, 
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const deepmd::AtomMap<VALUETYPE>&atommap,
		       const int			nghost,
		       const int			ago,
		       const std::string		scope = "");
}

