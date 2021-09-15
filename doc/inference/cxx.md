# C++ interface
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