# C/C++ interface

:::{note}
See [Environment variables](../env.md) for the runtime environment variables.
:::

## C++ interface

The C++ interface of DeePMD-kit is also available for the model interface, which is considered faster than the Python interface. An example `infer_water.cpp` is given below:

```cpp
#include "deepmd/DeepPot.h"

int main(){
  deepmd::DeepPot dp ("graph.pb");
  std::vector<double > coord = {1., 0., 0., 0., 0., 1.5, 1. ,0. ,3.};
  std::vector<double > cell = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  std::vector<int > atype = {1, 0, 1};
  double e;
  std::vector<double > f, v;
  dp.compute (e, f, v, coord, atype, cell);
}
```

where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.
See {cpp:class}`deepmd::DeepPot` for details.

You can compile `infer_water.cpp` using `gcc`:

```sh
gcc infer_water.cpp -L $deepmd_root/lib -L $tensorflow_root/lib -I $deepmd_root/include -Wl,--no-as-needed -ldeepmd_cc -lstdc++ -ltensorflow_cc -Wl,-rpath=$deepmd_root/lib -Wl,-rpath=$tensorflow_root/lib -o infer_water
```

and then run the program:

```sh
./infer_water
```

## C interface

Although C is harder to write, the C library will not be affected by different versions of C++ compilers.

An example `infer_water.c` is given below:

```cpp
#include <stdio.h>
#include <stdlib.h>
#include "deepmd/c_api.h"

int main(){
  const char* model = "graph.pb";
  double coord[] = {1., 0., 0., 0., 0., 1.5, 1. ,0. ,3.};
  double cell[] = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  int atype[] = {1, 0, 1};
  // init C pointers with given memory
  double* e = malloc(sizeof(*e));
  double* f = malloc(sizeof(*f) * 9); // natoms * 3
  double* v = malloc(sizeof(*v) * 9);
  double* ae = malloc(sizeof(*ae) * 9); // natoms
  double* av = malloc(sizeof(*av) * 27); // natoms * 9
  // DP model
  DP_DeepPot* dp = DP_NewDeepPot(model);
  DP_DeepPotCompute (dp, 3, coord, atype, cell, e, f, v, ae, av);
  // print results
  printf("energy: %f\n", *e);
  for (int ii = 0; ii < 9; ++ii)
    printf("force[%d]: %f\n", ii, f[ii]);
  for (int ii = 0; ii < 9; ++ii)
    printf("force[%d]: %f\n", ii, v[ii]);
  // free memory
  free(e);
  free(f);
  free(v);
  free(ae);
  free(av);
  DP_DeleteDeepPot(dp);
}
```

where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.
`ae` and `av` are atomic energy and atomic virials, respectively.
See {cpp:func}`DP_DeepPotCompute` for details.

You can compile `infer_water.c` using `gcc`:

```sh
gcc infer_water.c -L $deepmd_root/lib -L $tensorflow_root/lib -I $deepmd_root/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=$deepmd_root/lib -Wl,-rpath=$tensorflow_root/lib -o infer_water
```

and then run the program:

```sh
./infer_water
```

## Header-only C++ library interface (recommended)

The header-only C++ library is built based on the C library.
Thus, it has the same ABI compatibility as the C library but provides a powerful C++ interface.
To use it, include `deepmd/deepmd.hpp`.

```cpp
#include "deepmd/deepmd.hpp"

int main(){
  deepmd::hpp::DeepPot dp ("graph.pb");
  std::vector<double > coord = {1., 0., 0., 0., 0., 1.5, 1. ,0. ,3.};
  std::vector<double > cell = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  std::vector<int > atype = {1, 0, 1};
  double e;
  std::vector<double > f, v;
  dp.compute (e, f, v, coord, atype, cell);
}
```

Note that the feature of the header-only C++ library is still limited compared to the original C++ library.
See {cpp:class}`deepmd::hpp::DeepPot` for details.

You can compile `infer_water_hpp.cpp` using `gcc`:

```sh
gcc infer_water_hpp.cpp -L $deepmd_root/lib -L $tensorflow_root/lib -I $deepmd_root/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=$deepmd_root/lib -Wl,-rpath=$tensorflow_root/lib -o infer_water_hpp
```

and then run the program:

```sh
./infer_water_hpp
```

In some cases, one may want to pass the custom neighbor list instead of the native neighbor list. The above code can be revised as follows:

```cpp
  // neighbor list
  std::vector<std::vector<int >> nlist_vec = {
    {1, 2},
    {0, 2},
    {0, 1}
    };
  std::vector<int> ilist(3), numneigh(3);
  std::vector<int*> firstneigh(3);
  InputNlist nlist(3, &ilist[0], &numneigh[0], &firstneigh[0]);
  convert_nlist(nlist, nlist_vec);
  dp.compute (e, f, v, coord, atype, cell, 0, nlist, 0);
```

Here, `nlist_vec` means the neighbors of atom 0 are atom 1 and atom 2, the neighbors of atom 1 are atom 0 and atom 2, and the neighbors of atom 2 are atom 0 and atom 1.
