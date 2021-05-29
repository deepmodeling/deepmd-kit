#pragma once

#include "common.h"
#include "neighbor_list.h"

namespace deepmd{
/**
* @brief Deep Tensor.
**/
class DeepTensor
{
public:
  /**
  * @brief Deep Tensor constructor without initialization.
  **/
  DeepTensor();
  /**
  * @brief Deep Tensor constructor with initialization..
  * @param[in] model The name of the frozen model file.
  * @param[in] gpu_rank The GPU rank. Default is 0.
  * @param[in] file_content The content of the model file. If it is not empty, DP will read from the string instead of the file.
  **/
  DeepTensor(const std::string & model, 
	     const int & gpu_rank = 0, 
	     const std::string &name_scope = "");
  /**
  * @brief Initialize the Deep Tensor.
  * @param[in] model The name of the frozen model file.
  * @param[in] gpu_rank The GPU rank. Default is 0.
  * @param[in] file_content The content of the model file. If it is not empty, DP will read from the string instead of the file.
  **/
  void init (const std::string & model, 
	     const int & gpu_rank = 0, 
	     const std::string &name_scope = "");
  /**
  * @brief Print the DP summary to the screen.
  * @param[in] pre The prefix to each line.
  **/
  void print_summary(const std::string &pre) const;
public:
  /**
  * @brief Evaluate the value by using this model.
  * @param[out] value The value to evalute.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  **/
  void compute (std::vector<VALUETYPE> &	value,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box);
  /**
  * @brief Evaluate the value by using this model.
  * @param[out] value The value to evalute.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  * @param[in] nghost The number of ghost atoms.
  * @param[in] inlist The input neighbour list.
  **/
  void compute (std::vector<VALUETYPE> &	value,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost,
		const InputNlist &	inlist);
  /**
  * @brief Get the cutoff radius.
  * @return The cutoff radius.
  **/
  VALUETYPE cutoff () const {assert(inited); return rcut;};
  /**
  * @brief Get the number of types.
  * @return The number of types.
  **/
  int numb_types () const {assert(inited); return ntypes;};
  /**
  * @brief Get the output dimension.
  * @return The output dimension.
  **/
  int output_dim () const {assert(inited); return odim;};
  const std::vector<int> & sel_types () const {assert(inited); return sel_type;};
private:
  tensorflow::Session* session;
  std::string name_scope;
  int num_intra_nthreads, num_inter_nthreads;
  tensorflow::GraphDef graph_def;
  bool inited;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  std::string model_type;
  std::string model_version;
  int odim;
  std::vector<int> sel_type;
  template<class VT> VT get_scalar(const std::string & name) const;
  template<class VT> void get_vector (std::vector<VT> & vec, const std::string & name) const;
  void run_model (std::vector<VALUETYPE> &		d_tensor_,
		  tensorflow::Session *			session, 
		  const std::vector<std::pair<std::string, tensorflow::Tensor>> & input_tensors,
		  const AtomMap<VALUETYPE> &		atommap, 
		  const int				nghost = 0);
  void compute_inner (std::vector<VALUETYPE> &		value,
		      const std::vector<VALUETYPE> &	coord,
		      const std::vector<int> &		atype,
		      const std::vector<VALUETYPE> &	box);
  void compute_inner (std::vector<VALUETYPE> &		value,
		      const std::vector<VALUETYPE> &	coord,
		      const std::vector<int> &		atype,
		      const std::vector<VALUETYPE> &	box, 
		      const int				nghost,
		      const InputNlist&			inlist);
};
}

