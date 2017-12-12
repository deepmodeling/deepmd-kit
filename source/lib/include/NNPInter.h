#pragma once

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <vector>

using namespace tensorflow;
using namespace std;

#ifdef HIGH_PREC
typedef double VALUETYPE;
#else 
typedef float  VALUETYPE;
#endif

class NNPInter 
{
public:
  NNPInter () ;
  NNPInter  (const string & model);
  void init (const string & model);
public:
  void compute (VALUETYPE &			ener,
		vector<VALUETYPE> &		force,
		vector<VALUETYPE> &		virial,
		const vector<VALUETYPE> &	coord,
		const vector<int> &		atype,
		const vector<VALUETYPE> &	box);
  VALUETYPE cutoff () const {return rcut;};
private:
  Session* session;
  GraphDef graph_def;
  bool inited;
  VALUETYPE get_rcut () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
};


