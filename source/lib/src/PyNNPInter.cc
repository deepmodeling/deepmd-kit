#include "PyNNPInter.h"
#include "NNPAtomMap.h"
#include "SimulationRegion.h"
#include <stdexcept>

#include "PyCaller.h"

void PyNNPInter::run_model_ndarray(ENERGYTYPE &dener,
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
                                   const int nghost)
{
  // cout << "in run_model_ndarray 1 -------------------------\n";
  unsigned nloc = nnpmap.get_type().size();
  unsigned nall = nloc + nghost;
  if (nloc == 0)
  {
    dener = 0;
    // no backward map needed
    // dforce of size nall * 3
    dforce_.resize(nall * 3);
    fill(dforce_.begin(), dforce_.end(), 0.0);
    // dvirial of size 9
    dvirial.resize(9);
    fill(dvirial.begin(), dvirial.end(), 0.0);
    return;
  }
  // cout << "1--------------------------------------\n";

  PyArrayObject *energy_ndarry;
  PyArrayObject *force_ndarry;
  PyArrayObject *virial_ndarry;

  pyCaller.infer(pymodel, coord_ndarry, type_ndarry, box_ndarry, mesh_ndarry, natoms_ndarry, fparam_ndarry, aparam_ndarry, &energy_ndarry, &force_ndarry, &virial_ndarry);

  // cout << "2--------------------------------------\n";
  // Tensor output_e = output_tensors[0];
  // Tensor output_f = output_tensors[1];
  // Tensor output_av = output_tensors[2];

  // auto oe = output_e.flat<ENERGYTYPE>();
  // auto of = output_f.flat<VALUETYPE>();
  // auto oav = output_av.flat<VALUETYPE>();

  dener = *(ENERGYTYPE *)PyArray_GETPTR1(energy_ndarry, 0);
  // cout << "2.1------------------------------------\n";
  
  vector<VALUETYPE> dforce(3 * nall);
  dvirial.resize(9);

  VALUETYPE *force_data = (VALUETYPE*)PyArray_DATA(force_ndarry);

  for (unsigned ii = 0; ii < nall * 3; ++ii)
  {
    dforce[ii] = force_data[ii];
  }
  // cout << "2.2------------------------------------\n";

  // for (int ii = 0; ii < nall; ++ii)
  // {
    // dvirial[0] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 0));
    // dvirial[1] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 1));
    // dvirial[2] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 2));
    // dvirial[3] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 3));
    // dvirial[4] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 4));
    // dvirial[5] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 5));
    // dvirial[6] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 6));
    // dvirial[7] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 7));
    // dvirial[8] += 1.0 * (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 9 * ii + 8));
  // } 
  dvirial[0] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 0));
  dvirial[1] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 1));
  dvirial[2] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 2));
  dvirial[3] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 3));
  dvirial[4] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 4));
  dvirial[5] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 5));
  dvirial[6] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 6));
  dvirial[7] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 7));
  dvirial[8] = (*(VALUETYPE *)PyArray_GETPTR1(virial_ndarry, 8));
  // cout << "3--------------------------------------\n";

  dforce_ = dforce;
  // cout << "4--------------------------------------\n";
  nnpmap.backward(dforce_.begin(), dforce.begin(), 3);
  // cout << "5--------------------------------------\n";
}

PyNNPInter::PyNNPInter() : inited(false), init_nbor(false)
{
}

PyNNPInter::PyNNPInter(const string &model_path, const int &gpu_rank)
    : inited(false), init_nbor(false), model_path(model_path)
{
  init(model_path, gpu_rank);
}

PyNNPInter::~PyNNPInter() {}

void PyNNPInter::init(const string &model_path, const int &gpu_rank)
{
  // cout << "PyNNPInter::init -------------------------------\n";
  assert(!inited);

  pyCaller.init_python();

  // TODO
  // SessionOptions options;
  // options.config.set_inter_op_parallelism_threads(num_inter_nthreads);
  // options.config.set_intra_op_parallelism_threads(num_intra_nthreads);
  // checkStatus(ReadBinaryProto(Env::Default(), model, &graph_def));
  // checkStatus(NewSession(options, &session));
  // checkStatus(session->Create(graph_def));
  pymodel = pyCaller.init_model(model_path);

  rcut = pyCaller.get_scalar<VALUETYPE>(pymodel, "descrpt_attr/rcut");
  cell_size = rcut;
  ntypes = pyCaller.get_scalar<int>(pymodel, "descrpt_attr/ntypes");
  dfparam = pyCaller.get_scalar<int>(pymodel, "fitting_attr/dfparam");
  daparam = pyCaller.get_scalar<int>(pymodel, "fitting_attr/daparam");
  if (dfparam < 0)
    dfparam = 0;
  if (daparam < 0)
    daparam = 0;
  inited = true;

  init_nbor = false;
  ilist = NULL;
  jrange = NULL;
  jlist = NULL;
  ilist_size = 0;
  jrange_size = 0;
  jlist_size = 0;
}

void PyNNPInter::print_summary(const string &pre) const
{
  cout << pre << "installed to:       " + global_install_prefix << endl;
  cout << pre << "source:             " + global_git_summ << endl;
  cout << pre << "source brach:       " + global_git_branch << endl;
  cout << pre << "source commit:      " + global_git_hash << endl;
  cout << pre << "source commit at:   " + global_git_date << endl;
  cout << pre << "build float prec:   " + global_float_prec << endl;
  cout << pre << "build with tf inc:  " + global_tf_include_dir << endl;
  cout << pre << "build with tf lib:  " + global_tf_lib << endl;
  cout << pre << "set tf intra_op_parallelism_threads: " << num_intra_nthreads << endl;
  cout << pre << "set tf inter_op_parallelism_threads: " << num_inter_nthreads << endl;
}

void PyNNPInter::validate_fparam_aparam(const int &nloc,
                                        const vector<VALUETYPE> &fparam,
                                        const vector<VALUETYPE> &aparam) const
{
  if (fparam.size() != dfparam)
  {
    throw std::runtime_error("the dim of frame parameter provided is not consistent with what the model uses");
  }
  if (aparam.size() != daparam * nloc)
  {
    throw std::runtime_error("the dim of atom parameter provided is not consistent with what the model uses");
  }
}

void PyNNPInter::compute(ENERGYTYPE &dener,
                         vector<VALUETYPE> &dforce_,
                         vector<VALUETYPE> &dvirial,
                         const vector<VALUETYPE> &dcoord_,
                         const vector<int> &datype_,
                         const vector<VALUETYPE> &dbox,
                         const int nghost,
                         const LammpsNeighborList &lmp_list,
                         const int &ago,
                         const vector<VALUETYPE> &fparam,
                         const vector<VALUETYPE> &aparam_)
{
# ifdef PROF
  fapp_start("PyNNPInter::compute",1,0);
# endif

  // cout << " in PyNNPInter::compute 2 ------------------------\n";
  vector<VALUETYPE> dcoord, dforce, aparam;
  vector<int> datype, fwd_map, bkw_map;
  int nghost_real;
  select_real_atoms(fwd_map, bkw_map, nghost_real, dcoord_, datype_, nghost, ntypes);
  // resize to nall_real
  dcoord.resize(bkw_map.size() * 3);
  datype.resize(bkw_map.size());
  // fwd map
  select_map<VALUETYPE>(dcoord, dcoord_, fwd_map, 3);
  select_map<int>(datype, datype_, fwd_map, 1);
  // aparam
  if (daparam > 0)
  {
    aparam.resize(bkw_map.size());
    select_map<VALUETYPE>(aparam, aparam_, fwd_map, daparam);
  }
  // internal nlist
  if (ago == 0)
  {
    convert_nlist_lmp_internal(nlist, lmp_list);
    shuffle_nlist_exclude_empty(nlist, fwd_map);
  }

# ifdef PROF
  fapp_stop("PyNNPInter::compute",1,0);
# endif

  compute_inner(dener, dforce, dvirial, dcoord, datype, dbox, nghost_real, ago, fparam, aparam);
  // bkw map
  select_map<VALUETYPE>(dforce_, dforce, bkw_map, 3);



}

void PyNNPInter::compute_inner(ENERGYTYPE &dener,
                               vector<VALUETYPE> &dforce_,
                               vector<VALUETYPE> &dvirial,
                               const vector<VALUETYPE> &dcoord_,
                               const vector<int> &datype_,
                               const vector<VALUETYPE> &dbox,
                               const int nghost,
                               const int &ago,
                               const vector<VALUETYPE> &fparam,
                               const vector<VALUETYPE> &aparam)
{
  // cout << " in PyNNPInter::compute_inner 1 ------------------------\n";

  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;

  validate_fparam_aparam(nloc, fparam, aparam);

  // agp == 0 means that the LAMMPS nbor list has been updated
  if (ago == 0)
  {
    nnpmap = NNPAtomMap<VALUETYPE>(datype_.begin(), datype_.begin() + nloc);
    assert(nloc == nnpmap.get_type().size());

    shuffle_nlist(nlist, nnpmap);
  }

  // cout << "1------------------------\n";
  // std::vector<std::pair<string, Tensor> > input_tensors;
  // int ret = session_input_tensors(input_tensors, dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, nnpmap, nghost, ago);
  // assert(nloc == ret);
  // run_model(dener,dforce_,dvirial,session,input_tensors,nnpmap,nghost);

  PyArrayObject *coord_ndarry;
  PyArrayObject *type_ndarry;
  PyArrayObject *box_ndarry;
  PyArrayObject *mesh_ndarry;
  PyArrayObject *natoms_ndarry;
  PyArrayObject *fparam_ndarry;
  PyArrayObject *aparam_ndarry;
  int ret = pyCaller.session_input_ndarrays(&coord_ndarry, &type_ndarry, &box_ndarry, &mesh_ndarry, &natoms_ndarry, &fparam_ndarry, &aparam_ndarry,
                                            dcoord_, ntypes, datype_, dbox, nlist, fparam, aparam, nnpmap, nghost, ago);
  assert(nloc == ret);
  run_model_ndarray(dener, dforce_, dvirial,
                    coord_ndarry, type_ndarry, box_ndarry, mesh_ndarry, natoms_ndarry, fparam_ndarry, aparam_ndarry,
                    nnpmap, nghost);
}
