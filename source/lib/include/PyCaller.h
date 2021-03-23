#pragma once

#include <Python.h>
#include <pythonrun.h>
#include <numpy/ndarrayobject.h>
#include "pycommon.h"

class PyCaller
{

public:
    PyCaller(){};
    ~PyCaller(){};

    int init_python();

    int session_input_ndarrays(PyArrayObject **coord_ndarry_p,
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
                               const string scope = "");

    void infer(const PyObject *pymodel,
               const PyArrayObject *coord_ndarry,
               const PyArrayObject *type_ndarry,
               const PyArrayObject *box_ndarry,
               const PyArrayObject *mesh_ndarry,
               const PyArrayObject *natoms_ndarry,
               const PyArrayObject *fparam_ndarry,
               const PyArrayObject *aparam_ndarry,
               PyArrayObject **energy_ndarry,
               PyArrayObject **force_ndarry,
               PyArrayObject **virial_ndarry);

    PyObject *init_model(const string &model_path);

    template <class T>
    T get_scalar(const PyObject *pymodel, const string &scalar_name)
    {

        PyObject *scalarNameObj = PyUnicode_FromString(scalar_name.c_str());
        PyObject *args = PyTuple_Pack(2, pymodel, scalarNameObj);
        check(args, "getScalarFunc args error !!! ");


        PyObject *ret = PyObject_CallObject(getScalarFunc, args);
        check(ret, "call getScalarFunc error !!! ");

        double double_val;
        PyArg_Parse(ret, "d", &double_val);
        T retval = static_cast<T>(double_val);

        // cout << "in get scalar ret : " << double_val << "," << retval << endl;
        return retval;
    }

private:
    void check(void *ptr, const string &message);

    string inferModuleName = "deepmd.infer";
    string initFuncName = "init_model";
    string inferFuncName = "infer";
    string getScalarFuncName = "get_scalar";

    PyObject *inferModule = nullptr;
    PyObject *initFunc = nullptr;
    PyObject *inferFunc = nullptr;
    PyObject *getScalarFunc = nullptr;
};
