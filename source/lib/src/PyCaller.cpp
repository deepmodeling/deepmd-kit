#include "PyCaller.h"


void PyCaller::check(void *ptr, const string& message)
{
  if (ptr == nullptr)
  {
    std::cout << message << std::endl;
    exit(1);
  }
}

int PyCaller::init_python()
{
  if (!Py_IsInitialized())
  {
    Py_Initialize();
    import_array();
    if (!Py_IsInitialized())
    {
      cout << "python initialize error !!!" << endl;
      exit(-1);
    }
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append('/vol0004/hp200266/u01036/gzq/deepmd-kit/_skbuild/linux-aarch64-3.8/cmake-install')");

    inferModule = PyImport_Import(PyUnicode_FromString(inferModuleName.c_str()));
    check(inferModule, "module load error !!!");

    initFunc = PyObject_GetAttrString(inferModule, initFuncName.c_str());
    check(initFunc, "func load error !!!");

    inferFunc = PyObject_GetAttrString(inferModule, inferFuncName.c_str());
    check(inferFunc, "func load error !!!");

    getScalarFunc = PyObject_GetAttrString(inferModule, getScalarFuncName.c_str());
    check(getScalarFunc, "func load error !!!");
  }

  return 0;
}

int PyCaller::session_input_ndarrays(PyArrayObject **coord_ndarry_p,
                                     PyArrayObject **type_ndarry_p,
                                     PyArrayObject **box_ndarry_p,
                                     PyArrayObject **mesh_ndarry_p,
                                     PyArrayObject **natoms_ndarry_p,
                                     PyArrayObject **fparam_ndarry_p,
                                     PyArrayObject **aparam_ndarry_p,
                                     const vector<VALUETYPE> &dcoord_,
                                     const int &ntypes,
                                     const vector<int> &datype_,
                                     const vector<VALUETYPE> &dbox,
                                     InternalNeighborList &dlist,
                                     const vector<VALUETYPE> &fparam_,
                                     const vector<VALUETYPE> &aparam_,
                                     const NNPAtomMap<VALUETYPE> &nnpmap,
                                     const int nghost,
                                     const int ago,
                                     const string scope)
{
  // std::cout << "in session_input_ndarrays 2 -------------" << endl;
  assert(dbox.size() == 9);
  int nframes = 1;
  int nall = dcoord_.size() / 3;
  int nloc = nall - nghost;
  assert(nall == datype_.size());

  vector<int> datype = nnpmap.get_type();
  vector<int> type_count(ntypes, 0);
  for (unsigned ii = 0; ii < datype.size(); ++ii)
  {
    type_count[datype[ii]]++;
  }
  datype.insert(datype.end(), datype_.begin() + nloc, datype_.end());

  vector<npy_intp> coord_shape{nframes, nall * 3};
  vector<npy_intp> type_shape{nframes, nall};
  vector<npy_intp> box_shape{nframes, 9};
  vector<npy_intp> mesh_shape{16};
  vector<npy_intp> natoms_shape{2 + ntypes};
  vector<npy_intp> fparam_shape{nframes, static_cast<npy_intp>(fparam_.size())};
  vector<npy_intp> aparam_shape{nframes, static_cast<npy_intp>(aparam_.size())};

#ifdef HIGH_PREC
  PyArrayObject *coord_ndarry = (PyArrayObject *)PyArray_SimpleNew(coord_shape.size(), coord_shape.data(), NPY_DOUBLE);
  PyArrayObject *box_ndarry = (PyArrayObject *)PyArray_SimpleNew(box_shape.size(), box_shape.data(), NPY_DOUBLE);
  PyArrayObject *fparam_ndarry = (PyArrayObject *)PyArray_SimpleNew(fparam_shape.size(), fparam_shape.data(), NPY_DOUBLE);
  PyArrayObject *aparam_ndarry = (PyArrayObject *)PyArray_SimpleNew(aparam_shape.size(), aparam_shape.data(), NPY_DOUBLE);
#else
  PyArrayObject *coord_ndarry = (PyArrayObject *)PyArray_SimpleNew(coord_shape.size(), coord_shape.data(), NPY_FLOAT);
  PyArrayObject *box_ndarry = (PyArrayObject *)PyArray_SimpleNew(box_shape.size(), box_shape.data(), NPY_FLOAT);
  PyArrayObject *fparam_ndarry = (PyArrayObject *)PyArray_SimpleNew(fparam_shape.size(), fparam_shape.data(), NPY_FLOAT);
  PyArrayObject *aparam_ndarry = (PyArrayObject *)PyArray_SimpleNew(aparam_shape.size(), aparam_shape.data(), NPY_FLOAT);
#endif

  PyArrayObject *type_ndarry = (PyArrayObject *)PyArray_SimpleNew(type_shape.size(), type_shape.data(), NPY_INT32);
  PyArrayObject *mesh_ndarry = (PyArrayObject *)PyArray_SimpleNew(mesh_shape.size(), mesh_shape.data(), NPY_INT32);
  PyArrayObject *natoms_ndarry = (PyArrayObject *)PyArray_SimpleNew(natoms_shape.size(), natoms_shape.data(), NPY_INT32);

  vector<VALUETYPE> dcoord(dcoord_);
  nnpmap.forward(dcoord.begin(), dcoord_.begin(), 3);

  for (int ii = 0; ii < nframes; ++ii)
  {
    for (int jj = 0; jj < nall * 3; ++jj)
    {
      *(VALUETYPE *)PyArray_GETPTR2(coord_ndarry, ii, jj) = dcoord[jj];
    }
    for (int jj = 0; jj < 9; ++jj)
    {
      *(VALUETYPE *)PyArray_GETPTR2(box_ndarry, ii, jj) = dbox[jj];
    }
    for (int jj = 0; jj < nall; ++jj)
    {
      *(int *)PyArray_GETPTR2(type_ndarry, ii, jj) = datype[jj];
    }
    for (int jj = 0; jj < fparam_.size(); ++jj)
    {
      *(VALUETYPE *)PyArray_GETPTR2(fparam_ndarry, ii, jj) = fparam_[jj];
    }
    for (int jj = 0; jj < aparam_.size(); ++jj)
    {
      *(VALUETYPE *)PyArray_GETPTR2(aparam_ndarry, ii, jj) = aparam_[jj];
    }
  }
  for (int ii = 0; ii < 16; ++ii)
  {
    *(int *)PyArray_GETPTR1(mesh_ndarry, ii) = 0;
  }

  const int stride = sizeof(int *) / sizeof(int);

  *(int *)PyArray_GETPTR1(mesh_ndarry, 0) = ago;
  *(int *)PyArray_GETPTR1(mesh_ndarry, 1) = dlist.ilist.size();
  *(int *)PyArray_GETPTR1(mesh_ndarry, 2) = dlist.jrange.size();
  *(int *)PyArray_GETPTR1(mesh_ndarry, 3) = dlist.jlist.size();
  dlist.make_ptrs();

  memcpy(PyArray_GETPTR1(mesh_ndarry, 4), &(dlist.pilist), sizeof(int *));
  memcpy(PyArray_GETPTR1(mesh_ndarry, 8), &(dlist.pjrange), sizeof(int *));
  memcpy(PyArray_GETPTR1(mesh_ndarry, 12), &(dlist.pjlist), sizeof(int *));

  *(int *)PyArray_GETPTR1(natoms_ndarry, 0) = nloc;
  *(int *)PyArray_GETPTR1(natoms_ndarry, 1) = nall;
  for (int ii = 0; ii < ntypes; ++ii)
  {
    *(int *)PyArray_GETPTR1(natoms_ndarry, ii + 2) = type_count[ii];
  }

  string prefix = "";
  if (scope != "")
  {
    prefix = scope + "/";
  }

  *coord_ndarry_p = coord_ndarry;
  *type_ndarry_p = type_ndarry;
  *box_ndarry_p = box_ndarry;
  *mesh_ndarry_p = mesh_ndarry;
  *natoms_ndarry_p = natoms_ndarry;
  *fparam_ndarry_p = fparam_ndarry;
  *aparam_ndarry_p = aparam_ndarry;

  return nloc;
}

void PyCaller::infer(const PyObject *model,
                     const PyArrayObject *coord_ndarry,
                     const PyArrayObject *type_ndarry,
                     const PyArrayObject *box_ndarry,
                     const PyArrayObject *mesh_ndarry,
                     const PyArrayObject *natoms_ndarry,
                     const PyArrayObject *fparam_ndarry,
                     const PyArrayObject *aparam_ndarry,
                     PyArrayObject **energy_ndarry,
                     PyArrayObject **force_ndarry,
                     PyArrayObject **virial_ndarry)
{
  PyObject *args = PyTuple_Pack(6, model, coord_ndarry, type_ndarry, box_ndarry, mesh_ndarry, natoms_ndarry);

  PyObject *ret = PyObject_CallObject(inferFunc, args);
  check(ret, "call function error !!! ");

  *energy_ndarry = (PyArrayObject *)PyTuple_GetItem(ret, 0);
  *force_ndarry = (PyArrayObject *)PyTuple_GetItem(ret, 1);
  *virial_ndarry = (PyArrayObject *)PyTuple_GetItem(ret, 2);
}

PyObject *PyCaller::init_model(const string &model_path)
{
  PyObject *model_path_obj = PyUnicode_FromString(model_path.c_str());
  PyObject *args = PyTuple_Pack(1, model_path_obj);

  PyObject *ret = PyObject_CallObject(initFunc, args);
  check(ret, "call function error !!! ");

  return ret;
}
