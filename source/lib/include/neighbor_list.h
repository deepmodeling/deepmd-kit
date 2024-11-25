// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <vector>

#include "SimulationRegion.h"
#include "region.h"
#include "utilities.h"

namespace deepmd {

/**
 * @brief             Construct InputNlist with the input LAMMPS nbor list info.
 *
 * @struct            InputNlist
 */
struct InputNlist {
  /// Number of core region atoms
  int inum;
  /// Array stores the core region atom's index
  int* ilist;
  /// Array stores the core region atom's neighbor atom number
  int* numneigh;
  /// Array stores the core region atom's neighbor index
  int** firstneigh;
  ///  # of swaps to perform = sum of maxneed
  int nswap;
  /// # of atoms to send in each swap
  int* sendnum;
  /// # of atoms to recv in each swap
  int* recvnum;
  /// where to put 1st recv atom in each swap
  int* firstrecv;
  /// list of atoms to send in each swap
  int** sendlist;
  /// proc to send to at each swap
  int* sendproc;
  /// proc to recv from at each swap
  int* recvproc;
  /// MPI_comm data in lmp
  void* world;
  /// mask to the neighbor index
  int mask = 0xFFFFFFFF;
  /// mapping from all atoms to real atoms, in the size of nall
  int* mapping = nullptr;
  InputNlist()
      : inum(0),
        ilist(NULL),
        numneigh(NULL),
        firstneigh(NULL),
        nswap(0),
        sendnum(nullptr),
        recvnum(nullptr),
        firstrecv(nullptr),
        sendlist(nullptr),
        sendproc(nullptr),
        recvproc(nullptr),
        world(0) {};
  InputNlist(int inum_, int* ilist_, int* numneigh_, int** firstneigh_)
      : inum(inum_),
        ilist(ilist_),
        numneigh(numneigh_),
        firstneigh(firstneigh_),
        nswap(0),
        sendnum(nullptr),
        recvnum(nullptr),
        firstrecv(nullptr),
        sendlist(nullptr),
        sendproc(nullptr),
        recvproc(nullptr),
        world(0) {};
  InputNlist(int inum_,
             int* ilist_,
             int* numneigh_,
             int** firstneigh_,
             int nswap,
             int* sendnum,
             int* recvnum,
             int* firstrecv,
             int** sendlist,
             int* sendproc,
             int* recvproc,
             void* world)
      : inum(inum_),
        ilist(ilist_),
        numneigh(numneigh_),
        firstneigh(firstneigh_),
        nswap(nswap),
        sendnum(sendnum),
        recvnum(recvnum),
        firstrecv(firstrecv),
        sendlist(sendlist),
        sendproc(sendproc),
        recvproc(recvproc),
        world(world) {};
  ~InputNlist() {};
  /**
   * @brief Set mask for this neighbor list.
   */
  void set_mask(int mask_) { mask = mask_; };
  /**
   * @brief Set mapping for this neighbor list.
   */
  void set_mapping(int* mapping_) { mapping = mapping_; };
};

/**
 *@brief              Construct the InputNlist with a two-dimensional vector.
 *
 *@param              to_nlist:   InputNlist struct which stores the neighbor
 *information of the core region atoms.
 *@param              from_nlist: Vector which stores the neighbor information
 *of the core region atoms.
 */
void convert_nlist(InputNlist& to_nlist,
                   std::vector<std::vector<int> >& from_nlist);

/**
 *@brief              Compute the max number of neighbors within the core region
 *atoms
 *
 *@param              to_nlist:   InputNlist struct which stores the neighbor
 *information of the core region atoms.
 *
 *@return             integer
 *@retval             max number of neighbors
 */
int max_numneigh(const InputNlist& to_nlist);

// build neighbor list.
// outputs
//	nlist, max_list_size
//	max_list_size is the maximal size of jlist.
// inputs
//	c_cpy, nloc, nall, mem_size, rcut, region
//	mem_size is the size of allocated memory for jlist.
// returns
//	0: successful
//	1: the memory is not large enough to hold all neighbors.
//	   i.e. max_list_size > mem_nall
template <typename FPTYPE>
int build_nlist_cpu(InputNlist& nlist,
                    int* max_list_size,
                    const FPTYPE* c_cpy,
                    const int& nloc,
                    const int& nall,
                    const int& mem_size,
                    const float& rcut);

void use_nei_info_cpu(int* nlist,
                      int* ntype,
                      bool* nmask,
                      const int* type,
                      const int* nlist_map,
                      const int nloc,
                      const int nnei,
                      const int ntypes,
                      const bool b_nlist_map);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
/**
 *@brief              Convert the a host memory InputNlist to a device memory
 *InputNlist
 *
 *@param              cpu_nlist:    Host memory InputNlist struct which stores
 *the neighbor information of the core region atoms
 *@param              gpu_nlist:    Device memory InputNlist struct which stores
 *the neighbor information of the core region atoms
 *@param              gpu_memory:   Device array which stores the elements of
 *gpu_nlist
 *@param              max_nbor_size
 */
void convert_nlist_gpu_device(InputNlist& gpu_nlist,
                              InputNlist& cpu_nlist,
                              int*& gpu_memory,
                              const int& max_nbor_size);

/**
 *@brief              Reclaim the allocated device memory of struct InputNlist
 *
 *@param              gpu_nlist:    Device memory InputNlist struct which stores
 *the neighbor information of the core region atoms
 */
void free_nlist_gpu_device(InputNlist& gpu_nlist);

void use_nlist_map(int* nlist,
                   const int* nlist_map,
                   const int nloc,
                   const int nnei);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// build neighbor list.
// outputs
//	nlist, max_list_size
//	max_list_size is the maximal size of jlist.
// inputs
//	c_cpy, nloc, nall, mem_size, rcut, region
//	mem_size is the size of allocated memory for jlist.
// returns
//	0: successful
//	1: the memory is not large enough to hold all neighbors.
//	   i.e. max_list_size > mem_nall
template <typename FPTYPE>
int build_nlist_gpu(InputNlist& nlist,
                    int* max_list_size,
                    int* nlist_data,
                    const FPTYPE* c_cpy,
                    const int& nloc,
                    const int& nall,
                    const int& mem_size,
                    const float& rcut);

/**
 * @brief Filter the fake atom type.
 * @details If >=0, set to 0; if <0, set to -1.
 * @param ftype_out The output filtered atom type.
 * @param ftype_in The input atom type.
 * @param nloc The number of atoms.
 */
void filter_ftype_gpu(int* ftype_out, const int* ftype_in, const int nloc);

void use_nei_info_gpu(int* nlist,
                      int* ntype,
                      bool* nmask,
                      const int* type,
                      const int* nlist_map,
                      const int nloc,
                      const int nnei,
                      const int ntypes,
                      const bool b_nlist_map);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace deepmd

////////////////////////////////////////////////////////
// legacy code
////////////////////////////////////////////////////////

// build nlist by an extended grid
void build_nlist(std::vector<std::vector<int> >& nlist0,
                 std::vector<std::vector<int> >& nlist1,
                 const std::vector<double>& coord,
                 const int& nloc,
                 const double& rc0,
                 const double& rc1,
                 const std::vector<int>& nat_stt_,
                 const std::vector<int>& nat_end_,
                 const std::vector<int>& ext_stt_,
                 const std::vector<int>& ext_end_,
                 const SimulationRegion<double>& region,
                 const std::vector<int>& global_grid);

// build nlist by a grid for a periodic region
void build_nlist(std::vector<std::vector<int> >& nlist0,
                 std::vector<std::vector<int> >& nlist1,
                 const std::vector<double>& coord,
                 const double& rc0,
                 const double& rc1,
                 const std::vector<int>& grid,
                 const SimulationRegion<double>& region);

// build nlist by a grid for a periodic region, atoms selected by sel0 and sel1
void build_nlist(std::vector<std::vector<int> >& nlist0,
                 std::vector<std::vector<int> >& nlist1,
                 const std::vector<double>& coord,
                 const std::vector<int>& sel0,
                 const std::vector<int>& sel1,
                 const double& rc0,
                 const double& rc1,
                 const std::vector<int>& grid,
                 const SimulationRegion<double>& region);

// brute force (all-to-all distance computation) neighbor list building
// if region is NULL, open boundary is assumed,
// otherwise, periodic boundary condition is defined by region
void build_nlist(std::vector<std::vector<int> >& nlist0,
                 std::vector<std::vector<int> >& nlist1,
                 const std::vector<double>& coord,
                 const double& rc0_,
                 const double& rc1_,
                 const SimulationRegion<double>* region = NULL);

// copy periodic images for the system
void copy_coord(std::vector<double>& out_c,
                std::vector<int>& out_t,
                std::vector<int>& mapping,
                std::vector<int>& ncell,
                std::vector<int>& ngcell,
                const std::vector<double>& in_c,
                const std::vector<int>& in_t,
                const double& rc,
                const SimulationRegion<double>& region);
