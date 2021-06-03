#pragma once

#include "common.h"
#include "neighbor_list.h"

namespace deepmd{
/**
* @brief Deep Potential.
**/
class DeepPot 
{
public:
  /**
  * @brief DP constructor without initialization.
  **/
  DeepPot () ;
  ~DeepPot() ;
  /**
  * @brief DP constructor with initialization.
  * @param[in] model The name of the frozen model file.
  * @param[in] gpu_rank The GPU rank. Default is 0.
  * @param[in] file_content The content of the model file. If it is not empty, DP will read from the string instead of the file.
  **/
  DeepPot  (const std::string & model, const int & gpu_rank = 0, const std::string & file_content = "");
  /**
  * @brief Initialize the DP.
  * @param[in] model The name of the frozen model file.
  * @param[in] gpu_rank The GPU rank. Default is 0.
  * @param[in] file_content The content of the model file. If it is not empty, DP will read from the string instead of the file.
  **/
  void init (const std::string & model, const int & gpu_rank = 0, const std::string & file_content = "");
  /**
  * @brief Print the DP summary to the screen.
  * @param[in] pre The prefix to each line.
  **/
  void print_summary(const std::string &pre) const;
public:
  /**
  * @brief Evaluate the energy, force and virial by using this DP.
  * @param[out] ener The system energy.
  * @param[out] force The force on each atom.
  * @param[out] virial The virial.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  * @param[in] fparam The frame parameter. The array can be of size :
      * nframes x dim_fparam.
      * dim_fparam. Then all frames are assumed to be provided with the same fparam.
  * @param[in] aparam The atomic parameter The array can be of size :
      * nframes x natoms x dim_aparam.
      * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
      * dim_aparam. Then all frames and atoms are provided with the same aparam.
  **/
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  /**
  * @brief Evaluate the energy, force and virial by using this DP.
  * @param[out] ener The system energy.
  * @param[out] force The force on each atom.
  * @param[out] virial The virial.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  * @param[in] nghost The number of ghost atoms.
  * @param[in] inlist The input neighbour list.
  * @param[in] ago Update the internal neighbour list if ago is 0.
  * @param[in] fparam The frame parameter. The array can be of size :
      * nframes x dim_fparam.
      * dim_fparam. Then all frames are assumed to be provided with the same fparam.
  * @param[in] aparam The atomic parameter The array can be of size :
      * nframes x natoms x dim_aparam.
      * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
      * dim_aparam. Then all frames and atoms are provided with the same aparam.
  **/
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost,
		const InputNlist &		inlist,
		const int&			ago,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  /**
  * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial by using this DP.
  * @param[out] ener The system energy.
  * @param[out] force The force on each atom.
  * @param[out] virial The virial.
  * @param[out] atom_energy The atomic energy.
  * @param[out] atom_virial The atomic virial.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  * @param[in] fparam The frame parameter. The array can be of size :
      * nframes x dim_fparam.
      * dim_fparam. Then all frames are assumed to be provided with the same fparam.
  * @param[in] aparam The atomic parameter The array can be of size :
      * nframes x natoms x dim_aparam.
      * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
      * dim_aparam. Then all frames and atoms are provided with the same aparam.
  **/
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		std::vector<VALUETYPE> &	atom_energy,
		std::vector<VALUETYPE> &	atom_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
  /**
  * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial by using this DP.
  * @param[out] ener The system energy.
  * @param[out] force The force on each atom.
  * @param[out] virial The virial.
  * @param[out] atom_energy The atomic energy.
  * @param[out] atom_virial The atomic virial.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  * @param[in] nghost The number of ghost atoms.
  * @param[in] inlist The input neighbour list.
  * @param[in] ago Update the internal neighbour list if ago is 0.
  * @param[in] fparam The frame parameter. The array can be of size :
      * nframes x dim_fparam.
      * dim_fparam. Then all frames are assumed to be provided with the same fparam.
  * @param[in] aparam The atomic parameter The array can be of size :
      * nframes x natoms x dim_aparam.
      * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
      * dim_aparam. Then all frames and atoms are provided with the same aparam.
  **/
  void compute (ENERGYTYPE &			ener,
		std::vector<VALUETYPE> &	force,
		std::vector<VALUETYPE> &	virial,
		std::vector<VALUETYPE> &	atom_energy,
		std::vector<VALUETYPE> &	atom_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &	atype,
		const std::vector<VALUETYPE> &	box, 
		const int			nghost, 
		const InputNlist &	lmp_list,
		const int&			ago,
		const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());
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
  * @brief Get the dimension of the frame parameter.
  * @return The dimension of the frame parameter.
  **/
  int dim_fparam () const {assert(inited); return dfparam;};
  /**
  * @brief Get the dimension of the atomic parameter.
  * @return The dimension of the atomic parameter.
  **/
  int dim_aparam () const {assert(inited); return daparam;};
  /**
  * @brief Get the type map (element name of the atom types) of this model.
  * @param[out] type_map The type map of this model.
  **/
  void get_type_map (std::string & type_map);
private:
  tensorflow::Session* session;
  int num_intra_nthreads, num_inter_nthreads;
  tensorflow::GraphDef graph_def;
  bool inited;
  template<class VT> VT get_scalar(const std::string & name) const;
  // VALUETYPE get_rcut () const;
  // int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  std::string model_type;
  std::string model_version;
  int ntypes;
  int dfparam;
  int daparam;
  void validate_fparam_aparam(const int & nloc,
			      const std::vector<VALUETYPE> &fparam,
			      const std::vector<VALUETYPE> &aparam)const ;
  void compute_inner (ENERGYTYPE &			ener,
		      std::vector<VALUETYPE> &		force,
		      std::vector<VALUETYPE> &		virial,
		      const std::vector<VALUETYPE> &	coord,
		      const std::vector<int> &		atype,
		      const std::vector<VALUETYPE> &	box, 
		      const int				nghost,
		      const int &			ago,
		      const std::vector<VALUETYPE>&	fparam = std::vector<VALUETYPE>(),
		      const std::vector<VALUETYPE>&	aparam = std::vector<VALUETYPE>());

  // copy neighbor list info from host
  bool init_nbor;
  std::vector<int> sec_a;
  NeighborListData nlist_data;
  InputNlist nlist;
  AtomMap<VALUETYPE> atommap;

  // function used for neighbor list copy
  std::vector<int> get_sel_a() const;
};

class DeepPotModelDevi
{
public:
  /**
  * @brief DP model deviation constructor without initialization.
  **/
  DeepPotModelDevi () ;
  ~DeepPotModelDevi() ;
  /**
  * @brief DP model deviation constructor with initialization.
  * @param[in] model The names of the frozen model files.
  * @param[in] gpu_rank The GPU rank. Default is 0.
  * @param[in] file_content The contents of the model files. If it is not empty, DP will read from the strings instead of the files.
  **/
  DeepPotModelDevi  (const std::vector<std::string> & models, const int & gpu_rank = 0, const std::vector<std::string> & file_contents = std::vector<std::string>());
  /**
  * @brief Initialize the DP model deviation contrcutor.
  * @param[in] model The names of the frozen model files.
  * @param[in] gpu_rank The GPU rank. Default is 0.
  * @param[in] file_content The contents of the model files. If it is not empty, DP will read from the strings instead of the files.
  **/
  void init (const std::vector<std::string> & models, const int & gpu_rank = 0, const std::vector<std::string> & file_contents = std::vector<std::string>());
public:
  /**
  * @brief Evaluate the energy, force and virial by using these DP models.
  * @param[out] all_ener The system energies of all models.
  * @param[out] all_force The forces on each atom of all models.
  * @param[out] all_virial The virials of all models.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  * @param[in] nghost The number of ghost atoms.
  * @param[in] lmp_list The input neighbour list.
  * @param[in] ago Update the internal neighbour list if ago is 0.
  * @param[in] fparam The frame parameter. The array can be of size :
      * nframes x dim_fparam.
      * dim_fparam. Then all frames are assumed to be provided with the same fparam.
  * @param[in] aparam The atomic parameter The array can be of size :
      * nframes x natoms x dim_aparam.
      * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
      * dim_aparam. Then all frames and atoms are provided with the same aparam.
  **/
  void compute (std::vector<ENERGYTYPE> &		all_ener,
		std::vector<std::vector<VALUETYPE> > &	all_force,
		std::vector<std::vector<VALUETYPE> > &	all_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &		atype,
		const std::vector<VALUETYPE> &	box,
		const int			nghost,
		const InputNlist &	lmp_list,
		const int 				&   ago,
		const std::vector<VALUETYPE>	&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>	&	aparam = std::vector<VALUETYPE>());
  /**
  * @brief Evaluate the energy, force, virial, atomic energy, and atomic virial by using these DP models.
  * @param[out] all_ener The system energies of all models.
  * @param[out] all_force The forces on each atom of all models.
  * @param[out] all_virial The virials of all models.
  * @param[out] all_atom_energy The atomic energies of all models.
  * @param[out] all_atom_virial The atomic virials of all models.
  * @param[in] coord The coordinates of atoms. The array should be of size nframes x natoms x 3.
  * @param[in] atype The atom types. The list should contain natoms ints.
  * @param[in] box The cell of the region. The array should be of size nframes x 9.
  * @param[in] nghost The number of ghost atoms.
  * @param[in] lmp_list The input neighbour list.
  * @param[in] ago Update the internal neighbour list if ago is 0.
  * @param[in] fparam The frame parameter. The array can be of size :
      * nframes x dim_fparam.
      * dim_fparam. Then all frames are assumed to be provided with the same fparam.
  * @param[in] aparam The atomic parameter The array can be of size :
      * nframes x natoms x dim_aparam.
      * natoms x dim_aparam. Then all frames are assumed to be provided with the same aparam.
      * dim_aparam. Then all frames and atoms are provided with the same aparam.
  **/
  void compute (std::vector<ENERGYTYPE> &		all_ener,
		std::vector<std::vector<VALUETYPE> > &	all_force,
		std::vector<std::vector<VALUETYPE> > &	all_virial,
		std::vector<std::vector<VALUETYPE> > &	all_atom_energy,
		std::vector<std::vector<VALUETYPE> > &	all_atom_virial,
		const std::vector<VALUETYPE> &	coord,
		const std::vector<int> &		atype,
		const std::vector<VALUETYPE> &	box,
		const int			nghost,
		const InputNlist &	lmp_list,
		const int 				&   ago,
		const std::vector<VALUETYPE>	&	fparam = std::vector<VALUETYPE>(),
		const std::vector<VALUETYPE>	&	aparam = std::vector<VALUETYPE>());
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
  * @brief Get the dimension of the frame parameter.
  * @return The dimension of the frame parameter.
  **/
  int dim_fparam () const {assert(inited); return dfparam;};
  /**
  * @brief Get the dimension of the atomic parameter.
  * @return The dimension of the atomic parameter.
  **/
  int dim_aparam () const {assert(inited); return daparam;};
#ifndef HIGH_PREC
  /**
  * @brief Compute the average energy.
  * @param[out] dener The average energy.
  * @param[in] all_energy The energies of all models.
  **/
  void compute_avg (ENERGYTYPE &		dener,
		    const std::vector<ENERGYTYPE > &	all_energy);
#endif
  /**
  * @brief Compute the average energy.
  * @param[out] dener The average energy.
  * @param[in] all_energy The energies of all models.
  **/
  void compute_avg (VALUETYPE &			dener,
		    const std::vector<VALUETYPE > &	all_energy);
  /**
  * @brief Compute the average of vectors.
  * @param[out] avg The average of vectors.
  * @param[in] xx The vectors of all models.
  **/
  void compute_avg (std::vector<VALUETYPE> &		avg,
		    const std::vector<std::vector<VALUETYPE> > &	xx);
  /**
  * @brief Compute the standard deviation of vectors.
  * @param[out] std The standard deviation of vectors.
  * @param[in] avg The average of vectors.
  * @param[in] xx The vectors of all models.
  * @param[in] stride The stride to compute the deviation.
  **/
  void compute_std (
      std::vector<VALUETYPE> & std,
      const std::vector<VALUETYPE> & avg,
      const std::vector<std::vector<VALUETYPE> >& xx, 
      const int & stride);
  /**
  * @brief Compute the relative standard deviation of vectors.
  * @param[out] std The standard deviation of vectors.
  * @param[in] avg The average of vectors.
  * @param[in] eps The level parameter for computing the deviation.
  * @param[in] stride The stride to compute the deviation.
  **/
  void compute_relative_std (
      std::vector<VALUETYPE> & std,
      const std::vector<VALUETYPE> & avg,
      const VALUETYPE eps, 
      const int & stride);
  /**
  * @brief Compute the standard deviation of atomic energies.
  * @param[out] std The standard deviation of atomic energies.
  * @param[in] avg The average of atomic energies.
  * @param[in] xx The vectors of all atomic energies.
  **/
  void compute_std_e (std::vector<VALUETYPE> &		std,
		      const std::vector<VALUETYPE> &		avg,
		      const std::vector<std::vector<VALUETYPE> >&	xx);
  /**
  * @brief Compute the standard deviation of forces.
  * @param[out] std The standard deviation of forces.
  * @param[in] avg The average of forces.
  * @param[in] xx The vectors of all forces.
  **/
  void compute_std_f (std::vector<VALUETYPE> &		std,
		      const std::vector<VALUETYPE> &		avg,
		      const std::vector<std::vector<VALUETYPE> >& xx);
  /**
  * @brief Compute the relative standard deviation of forces.
  * @param[out] std The relative standard deviation of forces.
  * @param[in] avg The relative average of forces.
  * @param[in] eps The level parameter for computing the deviation.
  **/
  void compute_relative_std_f (std::vector<VALUETYPE> &		std,
		      const std::vector<VALUETYPE> &		avg,
		      const VALUETYPE eps);
private:
  unsigned numb_models;
  std::vector<tensorflow::Session*> sessions;
  int num_intra_nthreads, num_inter_nthreads;
  std::vector<tensorflow::GraphDef> graph_defs;
  bool inited;
  template<class VT> VT get_scalar(const std::string name) const;
  // VALUETYPE get_rcut () const;
  // int get_ntypes () const;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  std::string model_type;
  std::string model_version;
  int ntypes;
  int dfparam;
  int daparam;
  void validate_fparam_aparam(const int & nloc,
			      const std::vector<VALUETYPE> &fparam,
			      const std::vector<VALUETYPE> &aparam)const ;

  // copy neighbor list info from host
  bool init_nbor;
  std::vector<std::vector<int> > sec;
  deepmd::AtomMap<VALUETYPE> atommap;
  NeighborListData nlist_data;
  InputNlist nlist;

  // function used for nborlist copy
  std::vector<std::vector<int> > get_sel() const;
  void cum_sum(const std::vector<std::vector<tensorflow::int32> > n_sel);
};
}


