#pragma once

#include "NNPInter.h"

class DataModifier
{
public:
  DataModifier();
  DataModifier(const string & model, 
	       const int & gpu_rank = 0, 
	       const string & name_scope = "");
  ~DataModifier () {};
  void init (const string & model, 
	     const int & gpu_rank = 0, 
	     const string & name_scope = "");
  void print_summary(const string &pre) const;
public:
  void compute (vector<VALUETYPE> &		dfcorr_,
		vector<VALUETYPE> &		dvcorr_,
		const vector<VALUETYPE> &	dcoord_,
		const vector<int> &		datype_,
		const vector<VALUETYPE> &	dbox, 
		const vector<pair<int,int>> &	pairs,
		const vector<VALUETYPE> &	delef_, 
		const int			nghost,
		const LammpsNeighborList &	lmp_list);
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  int numb_types () const {assert(inited); return ntypes;};
  vector<int> sel_types () const {assert(inited); return sel_type;};
private:
  Session* session;
  string name_scope, name_prefix;
  int num_intra_nthreads, num_inter_nthreads;
  GraphDef graph_def;
  bool inited;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  string model_type;
  vector<int> sel_type;
  template<class VT> VT get_scalar(const string & name) const;
  template<class VT> void get_vector(vector<VT> & vec, const string & name) const;
  void run_model (vector<VALUETYPE> &		dforce,
		  vector<VALUETYPE> &		dvirial,
		  Session *			session,
		  const std::vector<std::pair<string, Tensor>> & input_tensors,
		  const NNPAtomMap<VALUETYPE> &	nnpmap,
		  const int			nghost);
};

