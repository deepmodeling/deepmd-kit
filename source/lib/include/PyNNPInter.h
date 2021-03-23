#pragma once

#include "PyCaller.h"

typedef double compute_t;

class PyNNPInter
{
public:
  PyNNPInter();
  ~PyNNPInter();
  PyNNPInter(const string &model, const int &gpu_rank = 0);
  void init(const string &model, const int &gpu_rank = 0);
  void print_summary(const string &pre) const;

public:
  void compute(ENERGYTYPE &ener,
               vector<VALUETYPE> &force,
               vector<VALUETYPE> &virial,
               const vector<VALUETYPE> &coord,
               const vector<int> &atype,
               const vector<VALUETYPE> &box,
               const int nghost,
               const LammpsNeighborList &lmp_list,
               const int &ago,
               const vector<VALUETYPE> &fparam = vector<VALUETYPE>(),
               const vector<VALUETYPE> &aparam = vector<VALUETYPE>());
  VALUETYPE cutoff() const
  {
    assert(inited);
    return rcut;
  };
  int numb_types() const
  {
    assert(inited);
    return ntypes;
  };
  int dim_fparam() const
  {
    assert(inited);
    return dfparam;
  };
  int dim_aparam() const
  {
    assert(inited);
    return daparam;
  };

private:
  int num_intra_nthreads, num_inter_nthreads;

  bool inited;
  VALUETYPE rcut;
  VALUETYPE cell_size;
  int ntypes;
  int dfparam;
  int daparam;
  void validate_fparam_aparam(const int &nloc,
                              const vector<VALUETYPE> &fparam,
                              const vector<VALUETYPE> &aparam) const;

  void compute_inner(ENERGYTYPE &ener,
                     vector<VALUETYPE> &force,
                     vector<VALUETYPE> &virial,
                     const vector<VALUETYPE> &coord,
                     const vector<int> &atype,
                     const vector<VALUETYPE> &box,
                     const int nghost,
                     const int &ago,
                     const vector<VALUETYPE> &fparam = vector<VALUETYPE>(),
                     const vector<VALUETYPE> &aparam = vector<VALUETYPE>());

  void run_model_ndarray(ENERGYTYPE &dener,
                         vector<VALUETYPE> &dforce_,
                         vector<VALUETYPE> &dvirial,
                         const PyArrayObject *coord_ndarry,
                         const PyArrayObject *type_ndarry,
                         const PyArrayObject *box_ndarry,
                         const PyArrayObject *mesh_ndarry,
                         const PyArrayObject *natoms_ndarry,
                         const PyArrayObject *fparam_ndarry,
                         const PyArrayObject *aparam_ndarry,
                         const NNPAtomMap<VALUETYPE> &nnpmap,
                         const int nghost = 0);

  // copy neighbor list info from host
  bool init_nbor;
  std::vector<int> sec_a;
  compute_t *array_double;
  InternalNeighborList nlist;
  NNPAtomMap<VALUETYPE> nnpmap;
  int *ilist, *jrange, *jlist;
  int ilist_size, jrange_size, jlist_size;

  PyCaller pyCaller;
  PyObject *pymodel;
  const string model_path;
};
