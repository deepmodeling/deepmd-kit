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

You can compile `infer_water.cpp` using `gcc`:
```sh
gcc infer_water.cpp -D HIGH_PREC -L $deepmd_root/lib -L $tensorflow_root/lib -I $deepmd_root/include -I $tensorflow_root/include -Wl,--no-as-needed -ldeepmd_cc -lstdc++ -Wl,-rpath=$deepmd_root/lib -Wl,-rpath=$tensorflow_root/lib -o infer_water
```
and then run the program:
```sh
./infer_water
```

## C interface
An example `infer_water.c` is given below:
```cpp
#include "deepmd/c_api.h"

int main(){
  char* model = "graph.pb";
  DP_DeepPot* dp = DP_NewDeepPot(model);
  double coord[] = {1., 0., 0., 0., 0., 1.5, 1. ,0. ,3.};
  double cell[] = {10., 0., 0., 0., 10., 0., 0., 0., 10.};
  int atype[] = {1, 0, 1};
  DP_ComputeResult result = DP_DeepPotCompute (dp, 3, &coord[0], &atype[0], &cell[0]);
  const double e = result.energy, *f = (double*)result.force, *v = (double*)result.virial;
}
```

where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.

You can compile `infer_water.c` using `gcc`:
```sh
gcc infer_water.cpp -D HIGH_PREC -L $deepmd_root/lib -L $tensorflow_root/lib -I $deepmd_root/include -Wl,--no-as-needed -ldeepmd_c -Wl,-rpath=$deepmd_root/lib -Wl,-rpath=$tensorflow_root/lib -o infer_water
```
and then run the program:
```sh
./infer_water
```