#pragma once
#include <vector>
#include <string>
#include <iostream>
#include "neighbor_list.h"
#include "AtomMap.h"
#include "common.h"
#include "paddle/include/paddle_inference_api.h"

using namespace std;

namespace deepmd{
  int get_math_lib_num_threads();
  
  int paddle_input_tensors (
    std::shared_ptr<paddle_infer::Predictor> predictor_,
    const std::vector<deepmd::VALUETYPE> &	dcoord_,
    const int &					ntypes,
    const std::vector<int> &			datype_,
    const std::vector<deepmd::VALUETYPE> &	dbox, 
    const deepmd::VALUETYPE &			cell_size,
    const std::vector<deepmd::VALUETYPE> &	fparam_,
    const std::vector<deepmd::VALUETYPE> &	aparam_,
    const deepmd::AtomMap<deepmd::VALUETYPE>&	atommap);

  int paddle_input_tensors (std::shared_ptr<paddle_infer::Predictor> predictor_,
		       const std::vector<VALUETYPE> &	dcoord_,
		       const int &			ntypes,
		       const std::vector<int> &		datype_,
		       const std::vector<VALUETYPE> &	dbox,		    
		       InputNlist &		dlist, 
		       const std::vector<VALUETYPE> &	fparam_,
		       const std::vector<VALUETYPE> &	aparam_,
		       const deepmd::AtomMap<VALUETYPE>&atommap,
		       const int			nghost,
		       int			ago=0);
}
