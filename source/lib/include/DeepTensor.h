#pragma once

#include "NNPInter.h"

class DeepTensor
{
public:
  DeepTensor();
  DeepTensor(const string & model, 
	     const int & gpu_rank = 0, 
	     const string &name_scope = "");
  void init (const string & model, 
	     const int & gpu_rank = 0, 
	     const string &name_scope = "");
  void print_summary(const string &pre) const;
public:
  void compute (vector<VALUETYPE> &		value,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box,
		const int			nghost = 0);
  void compute (vector<VALUETYPE> &		value,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box, 
		const int			nghost,
		const LammpsNeighborList &	lmp_list);
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  int output_dim () const {assert(inited); return odim;};
  const vector<int> & sel_types () const {assert(inited); return sel_type;};
private:
  Session* session;
  string name_scope;
  int num_intra_nthreads, num_inter_nthreads;
  GraphDef graph_def;
  bool inited;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  string model_type;
  int odim;
  vector<int> sel_type;
  template<class VT> VT get_scalar(const string & name) const;
  template<class VT> void get_vector (vector<VT> & vec, const string & name) const;
  void run_model (vector<VALUETYPE> &		d_tensor_,
		  Session *			session, 
		  const std::vector<std::pair<string, Tensor>> & input_tensors,
		  const NNPAtomMap<VALUETYPE> &	nnpmap, 
		  const int			nghost = 0);
  void compute_inner (vector<VALUETYPE> &	value,
		      const vector<VALUETYPE> &	coord,
		      const vector<int> &	atype,
		      const vector<VALUETYPE> &	box,
		      const int			nghost = 0);
  void compute_inner (vector<VALUETYPE> &	value,
		      const vector<VALUETYPE> &	coord,
		      const vector<int> &	atype,
		      const vector<VALUETYPE> &	box, 
		      const int			nghost,
		      const InternalNeighborList&lmp_list);
};

