# C/C++ interface
## C++ interface
The C++ interface of DeePMD-kit is also avaiable for model interface, which is considered faster than Python interface. An example `infer_water.cpp` is given below:
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

Although C is harder to write, C library will not be affected by different versions of C++ compilers.

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
  free(dp);
}
```

where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.
`ae` and `av` are atomic energy and atomic virial, respectively.
See {cpp:func}`DP_DeepPotCompute` for details.

You can compile `infer_water.c` using `gcc`:
```sh
gcc infer_water.c -L $deepmd_root/lib -L $tensorflow_root/lib -I $deepmd_root/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=$deepmd_root/lib -Wl,-rpath=$tensorflow_root/lib -o infer_water
```
and then run the program:
```sh
./infer_water
```
