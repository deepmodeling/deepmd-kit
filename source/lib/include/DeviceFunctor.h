#pragma once
#include <vector>
#include <climits>
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>

typedef unsigned long long int_64;
#define SQRT_2_PI 0.7978845608028654 

#define cudaErrcheck(res) {cudaAssert((res), __FILE__, __LINE__);}
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"cuda assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

template<typename T>
struct DescrptSeAGPUExecuteFunctor {
    void operator()(const T * coord, const int * type, const int * ilist, const int * jrange, const int * jlist, int * array_int, unsigned long long * array_longlong, const T * avg, const T * std, T * descript, T * descript_deriv, T * rij, int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const float rcut_r, const float rcut_r_smth, const std::vector<int> sec_a, const bool fill_nei_a, const int MAGIC_NUMBER);
};

template<typename T>
struct ProdForceSeAGPUExecuteFunctor {
    void operator()(T * force, const T * net_derive, const T * in_deriv, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift);
};

template<typename T>
struct ProdVirialSeAGPUExecuteFunctor {
    void operator()(T * virial, T * atom_virial, const T * net_deriv, const T * in_deriv, const T * rij, const int * nlist, const int nloc, const int nall, const int nnei, const int ndescrpt, const int n_a_sel, const int n_a_shift);
};

template<typename T>
struct GeluGPUExecuteFunctor {
    void operator()(const T * in, T * out, const int size);
};

template<typename T>
struct GeluGradGPUExecuteFunctor {
    void operator()(const T * dy, const T * in, T * out, const int size);
};

template<typename T>
struct GeluGradGradGPUExecuteFunctor {
    void operator()(const T * dy, const T * dy_, const T * in, T * out, const int size);
};