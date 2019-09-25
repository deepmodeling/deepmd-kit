# Multi-GPU support
Recently, we updated code for multi-GPU support. The main changes are focused on the interface part of LAMMPS and TensorFlow,
while the LAMMPS package was also minorly modified. Here, we'll give you a brief overview of this upgrade and provide a simple test. First, you need to install the GPU support environment for DeePMD-kit.
## Install GPU support environments for DeePMD-kit
First, you need a CUDA environment, and CUDA-10.0 is required. If you have a higher version of CUDA, such as CUDA-10.1, you can use it when compiling or running the DeePMD-kit's c-plus-plus interface. However, when you use the DeePMD-kit's python interface, the CUDA-10.0 environment is required. Lower versions of CUDAs are not recommended for use.

For a successful installation, we strongly recommend that you use Bazel-0.24.1, TensorFlow-1.14.0-GPU, as well as higher versions of CMake and git. Sometimes you may also report an error due to a low python version or a GCC version issue. When you have a compilation problem, it may be helpful to try to upgrade the software version.

Detailed installation process can be referred to [tf-1.14-gpu](doc/install-tf.1.14-gpu.md).
## Code upgrade
We'll briefly describe this upgrade in three parts.
### Allocate GPU resources to TensorFlow
Tensorflow uses all available GPU resources by default. So in the original code, when we run parallel programs, multiple processors apply for memory resources will conflict and report errors. So we use the TensorFlow graph API to assign a specific GPU to each TensorFlow graph based on the device ranks while limiting the default memory usage of TensorFlow in multiple GPU cases. The code can be viewed [here](source/lib/src/NNPInter.cc), focusing mainly on the init function.
### Get the processor's node rank
When working on platforms across nodes, we need to consider how to get the device rank mentioned in the previous section. If you're using Open MPI, it comes with a node-rank API, but if you use an Intel impi, you may need to use another method to specify node-rank. At present, we think it is a good way to be compatible with multi-platforms by dividing the MPI communicator based on the processor name. The code can be viewed [here](source/lmp/pair_nnp.cpp), focusing mainly on the get_node_rank function.
### Cmake conditional compilation
We introduced the USE-CUDA-TOOLKIT parameter as a control variable for whether to compile using the CUDA environment. If you want to build DeePMD-kit with CUDA-toolkit support, then execute cmake
```bash
cmake -DTF_GOOGLE_BIN=true -DUSE_CUDA_TOOLKIT=true -DTENSORFLOW_ROOT=$tensorflow_root \
-DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```
## Simple test for multi-GPU support
We tested the water sample provided by DeePMD-kit on up to 24 NVIDIA GV100 devices, as follows:
### Signal processor with signal GPU with 12288 atoms
```bash
Loop time of 230.028 on 1 procs for 1000 steps with 12288 atoms

Performance: 0.188 ns/day, 127.793 hours/ns, 4.347 timesteps/s
218.9% CPU use with 1 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 222.29     | 222.29     | 222.29     |   0.0 | 96.64
Neigh   | 7.1514     | 7.1514     | 7.1514     |   0.0 |  3.11
Comm    | 0.15155    | 0.15155    | 0.15155    |   0.0 |  0.07
Output  | 0.15792    | 0.15792    | 0.15792    |   0.0 |  0.07
Modify  | 0.21998    | 0.21998    | 0.21998    |   0.0 |  0.10
Other   |            | 0.05425    |            |       |  0.02
```
### Two processors with two GPUs with 12288 atoms
```bash
Loop time of 103.86 on 2 procs for 1000 steps with 12288 atoms

Performance: 0.416 ns/day, 57.700 hours/ns, 9.628 timesteps/s
184.9% CPU use with 2 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 99.374     | 99.479     | 99.584     |   1.1 | 95.78
Neigh   | 3.5141     | 3.5171     | 3.5201     |   0.2 |  3.39
Comm    | 0.50469    | 0.61397    | 0.72326    |  13.9 |  0.59
Output  | 0.083435   | 0.083471   | 0.083507   |   0.0 |  0.08
Modify  | 0.12354    | 0.12436    | 0.12519    |   0.2 |  0.12
Other   |            | 0.04167    |            |       |  0.04
```
### Four processors with four GPUs with 12288 atoms
```bash
Loop time of 63.6919 on 4 procs for 1000 steps with 12288 atoms

Performance: 0.678 ns/day, 35.384 hours/ns, 15.701 timesteps/s
157.1% CPU use with 4 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 60.436     | 60.917     | 61.278     |   4.6 | 95.64
Neigh   | 1.8222     | 1.8335     | 1.8443     |   0.7 |  2.88
Comm    | 0.42573    | 0.79821    | 1.2909     |  41.1 |  1.25
Output  | 0.048915   | 0.048949   | 0.049043   |   0.0 |  0.08
Modify  | 0.071305   | 0.071748   | 0.072062   |   0.1 |  0.11
Other   |            | 0.02293    |            |       |  0.04
```
### Eight processors with Eight GPUs with 12288 atoms
```bash
Loop time of 32.2646 on 8 procs for 1000 steps with 12288 atoms

Performance: 1.339 ns/day, 17.925 hours/ns, 30.994 timesteps/s
163.6% CPU use with 8 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 30.148     | 30.552     | 30.796     |   3.6 | 94.69
Neigh   | 0.89673    | 0.90676    | 0.91457    |   0.6 |  2.81
Comm    | 0.4564     | 0.70866    | 1.1179     |  24.1 |  2.20
Output  | 0.029983   | 0.03001    | 0.03012    |   0.0 |  0.09
Modify  | 0.053134   | 0.055055   | 0.057796   |   0.6 |  0.17
Other   |            | 0.01217    |            |       |  0.04
```
### Sixteen processors with sixteen GPUs with 12288 atoms
```bash
Loop time of 17.583 on 16 procs for 1000 steps with 12288 atoms

Performance: 2.457 ns/day, 9.768 hours/ns, 56.873 timesteps/s
164.8% CPU use with 16 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 16.082     | 16.367     | 16.524     |   3.1 | 93.09
Neigh   | 0.44881    | 0.45388    | 0.46047    |   0.5 |  2.58
Comm    | 0.47882    | 0.64386    | 0.93547    |  16.0 |  3.66
Output  | 0.02269    | 0.022706   | 0.022812   |   0.0 |  0.13
Modify  | 0.077578   | 0.086789   | 0.096221   |   1.8 |  0.49
Other   |            | 0.008571   |            |       |  0.05
```
### 24 processors with 24 GPUs with 12288 atoms
```bash
Loop time of 12.4446 on 24 procs for 1000 steps with 12288 atoms

Performance: 3.471 ns/day, 6.914 hours/ns, 80.356 timesteps/s
165.9% CPU use with 24 MPI tasks x no OpenMP threads

MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 11.055     | 11.404     | 11.609     |   3.6 | 91.64
Neigh   | 0.29447    | 0.30082    | 0.31717    |   0.8 |  2.42
Comm    | 0.42714    | 0.61319    | 0.96846    |  15.4 |  4.93
Output  | 0.024538   | 0.024569   | 0.0247     |   0.0 |  0.20
Modify  | 0.071891   | 0.095517   | 0.12622    |   5.8 |  0.77
Other   |            | 0.00627    |            |       |  0.05
```