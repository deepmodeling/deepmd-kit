# Getting Started
In this text, we will call the deep neural network that is used to represent the interatomic interactions (Deep Potential) the **model**. The typical procedure of using DeePMD-kit is 

1. [Prepare data](#prepare-data)
2. [Train a model](#train-a-model)
    - [Write the input script](#write-the-input-script)
    - [Training](#training)
    - [Training analysis with Tensorboard](#training-analysis-with-tensorboard)
3. [Freeze a model](#freeze-a-model)
4. [Test a model](#test-a-model)
5. [Compress a model](#compress-a-model)
6. [Model inference](#model-inference)
    - [Python interface](#python-interface)
    - [C++ interface](#c-interface)
7. [Run MD](#run-md)
    - [Run MD with LAMMPS](#run-md-with-lammps)
    - [Run path-integral MD with i-PI](#run-path-integral-md-with-i-pi)
    - [Use deep potential with ASE](#use-deep-potential-with-ase)
8. [Known limitations](#known-limitations)


## Prepare data
One needs to provide the following information to train a model: the atom type, the simulation box, the atom coordinate, the atom force, system energy and virial. A snapshot of a system that contains these information is called a **frame**. We use the following convention of units:


Property | Unit 
---|---
Time     | ps   
Length   | Å    
Energy   | eV   
Force    | eV/Å 
Virial   | eV   
Pressure | Bar  


The frames of the system are stored in two formats. A raw file is a plain text file with each information item written in one file and one frame written on one line. The default files that provide box, coordinate, force, energy and virial are `box.raw`, `coord.raw`, `force.raw`, `energy.raw` and `virial.raw`, respectively. *We recommend you use these file names*. Here is an example of force.raw:
```bash
$ cat force.raw
-0.724  2.039 -0.951  0.841 -0.464  0.363
 6.737  1.554 -5.587 -2.803  0.062  2.222
-1.968 -0.163  1.020 -0.225 -0.789  0.343
```
This `force.raw` contains 3 frames with each frame having the forces of 2 atoms, thus it has 3 lines and 6 columns. Each line provides all the 3 force components of 2 atoms in 1 frame. The first three numbers are the 3 force components of the first atom, while the second three numbers are the 3 force components of the second atom. The coordinate file `coord.raw` is organized similarly. In `box.raw`, the 9 components of the box vectors should be provided on each line. In `virial.raw`, the 9 components of the virial tensor should be provided on each line in the order `XX XY XZ YX YY YZ ZX ZY ZZ`. The number of lines of all raw files should be identical.

We assume that the atom types do not change in all frames. It is provided by `type.raw`, which has one line with the types of atoms written one by one. The atom types should be integers. For example the `type.raw` of a system that has 2 atoms with 0 and 1:
```bash
$ cat type.raw
0 1
```

Sometimes one needs to map the integer types to atom name. The mapping can be given by the file `type_map.raw`. For example
```bash
$ cat type_map.raw
O H
```
The type `0` is named by `"O"` and the type `1` is named by `"H"`.

The second format is the data sets of `numpy` binary data that are directly used by the training program. User can use the script `$deepmd_source_dir/data/raw/raw_to_set.sh` to convert the prepared raw files to data sets. For example, if we have a raw file that contains 6000 frames, 
```bash
$ ls 
box.raw  coord.raw  energy.raw  force.raw  type.raw  virial.raw
$ $deepmd_source_dir/data/raw/raw_to_set.sh 2000
nframe is 6000
nline per set is 2000
will make 3 sets
making set 0 ...
making set 1 ...
making set 2 ...
$ ls 
box.raw  coord.raw  energy.raw  force.raw  set.000  set.001  set.002  type.raw  virial.raw
```
It generates three sets `set.000`, `set.001` and `set.002`, with each set contains 2000 frames. One do not need to take care of the binary data files in each of the `set.*` directories. The path containing `set.*` and `type.raw` is called a *system*. 

### Data preparation with dpdata

One can use the a convenient tool `dpdata` to convert data directly from the output of first priciple packages to the DeePMD-kit format. One may follow the [example](data-conv.md) of using `dpdata` to find out how to use it.

## Train a model

### Write the input script

A model has two parts, a descriptor that maps atomic configuration to a set of symmetry invariant features, and a fitting net that takes descriptor as input and predicts the atomic contribution to the target physical property.

DeePMD-kit implements the following descriptors:
1. [`se_e2_a`](train-se-e2-a.md#descriptor): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes the distance between atoms as input.
2. [`se_e2_r`](train-se-e2-r.md): DeepPot-SE constructed from radial information of atomic configurations. The embedding takes the distance between atoms as input.
3. [`se_e3`](train-se-e3.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes angles between two neighboring atoms as input.
4. `loc_frame`: Defines a local frame at each atom, and the compute the descriptor as local coordinates under this frame.
5. [`hybrid`](train-hybrid.md): Concate a list of descriptors to form a new descriptor.

The fitting of the following physical properties are supported
1. [`ener`](train-se-e2-a.md#fitting): Fitting the energy of the system. The force (derivative with atom positions) and the virial (derivative with the box tensor) can also be trained. See [the example](train-se-e2-a.md#loss).
2. `dipole`: The dipole moment.
3. `polar`: The polarizability.


### Training

The training can be invoked by
```bash
$ dp train input.json
```
where `input.json` is the name of the input script. See [the example](train-se-e2-a.md#train-a-deep-potential-model) for more details.

During the training, checkpoints will be written to files with prefix `save_ckpt` every `save_freq` training steps. 

Several command line options can be passed to `dp train`, which can be checked with
```bash
$ dp train --help
```
An explanation will be provided
```
positional arguments:
  INPUT                 the input json database

optional arguments:
  -h, --help            show this help message and exit
  --init-model INIT_MODEL
                        Initialize a model by the provided checkpoint
  --restart RESTART     Restart the training from the provided checkpoint
```

**`--init-model model.ckpt`**, initializes the model training with an existing model that is stored in the checkpoint `model.ckpt`, the network architectures should match.

**`--restart model.ckpt`**, continues the training from the checkpoint `model.ckpt`.

On some resources limited machines, one may want to control the number of threads used by DeePMD-kit. This is achieved by three environmental variables: `OMP_NUM_THREADS`, `TF_INTRA_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS`. `OMP_NUM_THREADS` controls the multithreading of DeePMD-kit implemented operations. `TF_INTRA_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS` controls `intra_op_parallelism_threads` and `inter_op_parallelism_threads`, which are  Tensorflow configurations for multithreading. An explanation is found [here](https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads).

For example if you wish to use 3 cores of 2 CPUs on one node, you may set the environmental variables and run DeePMD-kit as follows:
```bash
export OMP_NUM_THREADS=6
export TF_INTRA_OP_PARALLELISM_THREADS=3
export TF_INTER_OP_PARALLELISM_THREADS=2
dp train input.json
```

### Training analysis with Tensorboard

If enbled in json/yaml input file DeePMD-kit will create log files which can be
used to analyze training procedure with Tensorboard. For a short tutorial
please read this [document](tensorboard.md).

## Freeze a model

The trained neural network is extracted from a checkpoint and dumped into a database. This process is called "freezing" a model. The idea and part of our code are from [Morgan](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc). To freeze a model, typically one does
```bash
$ dp freeze -o graph.pb
```
in the folder where the model is trained. The output database is called `graph.pb`.


## Test a model

The frozen model can be used in many ways. The most straightforward test can be performed using `dp test`. A typical usage of `dp test` is 
```bash
dp test -m graph.pb -s /path/to/system -n 30
```
where `-m` gives the tested model, `-s` the path to the tested system and `-n` the number of tested frames. Several other command line options can be passed to `dp test`, which can be checked with
```bash
$ dp test --help
```
An explanation will be provided
```
usage: dp test [-h] [-m MODEL] [-s SYSTEM] [-S SET_PREFIX] [-n NUMB_TEST]
               [-r RAND_SEED] [--shuffle-test] [-d DETAIL_FILE]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Frozen model file to import
  -s SYSTEM, --system SYSTEM
                        The system dir
  -S SET_PREFIX, --set-prefix SET_PREFIX
                        The set prefix
  -n NUMB_TEST, --numb-test NUMB_TEST
                        The number of data for test
  -r RAND_SEED, --rand-seed RAND_SEED
                        The random seed
  --shuffle-test        Shuffle test data
  -d DETAIL_FILE, --detail-file DETAIL_FILE
                        The file containing details of energy force and virial
                        accuracy
```

### Calculate Model Deviation

One can also use a subcommand to calculate deviation of prediced forces or virials for a bunch of models in the following way:
```bash
dp model-devi -m graph.000.pb graph.001.pb graph.002.pb graph.003.pb -s ./data -o model_devi.out
```
where `-m` specifies graph files to be calculated, `-s` gives the data to be evaluated, `-o` the file to which model deviation results is dumped. Here is more information on this sub-command:
```bash
usage: dp model-devi [-h] [-v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}]
                     [-l LOG_PATH] [-m MODELS [MODELS ...]] [-s SYSTEM]
                     [-S SET_PREFIX] [-o OUTPUT] [-f FREQUENCY] [-i ITEMS]

optional arguments:
  -h, --help            show this help message and exit
  -v {DEBUG,3,INFO,2,WARNING,1,ERROR,0}, --log-level {DEBUG,3,INFO,2,WARNING,1,ERROR,0}
                        set verbosity level by string or number, 0=ERROR,
                        1=WARNING, 2=INFO and 3=DEBUG (default: INFO)
  -l LOG_PATH, --log-path LOG_PATH
                        set log file to log messages to disk, if not
                        specified, the logs will only be output to console
                        (default: None)
  -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                        Frozen models file to import (default:
                        ['graph.000.pb', 'graph.001.pb', 'graph.002.pb',
                        'graph.003.pb'])
  -s SYSTEM, --system SYSTEM
                        The system directory, not support recursive detection.
                        (default: .)
  -S SET_PREFIX, --set-prefix SET_PREFIX
                        The set prefix (default: set)
  -o OUTPUT, --output OUTPUT
                        The output file for results of model deviation
                        (default: model_devi.out)
  -f FREQUENCY, --frequency FREQUENCY
                        The trajectory frequency of the system (default: 1)
```

For more details with respect to definition of model deviation and its application, please refer to `Yuzhi Zhang, Haidi Wang, Weijie Chen, Jinzhe Zeng, Linfeng Zhang, Han Wang, and Weinan E, DP-GEN: A concurrent learning platform for the generation of reliable deep learning based potential energy models, Computer Physics Communications, 2020, 253, 107206.`

## Compress a model

Once the frozen model is obtained from deepmd-kit, we can get the neural network structure and its parameters (weights, biases, etc.) from the trained model, and compress it in the following way:
```bash
dp compress input.json -i graph.pb -o graph-compress.pb
```
where input.json denotes the original training input script, `-i` gives the original frozen model, `-o` gives the compressed model. Several other command line options can be passed to `dp compress`, which can be checked with
```bash
$ dp compress --help
```
An explanation will be provided
```
usage: dp compress [-h] [-i INPUT] [-o OUTPUT] [-e EXTRAPOLATE] [-s STRIDE]
                   [-f FREQUENCY] [-d FOLDER]
                   INPUT

positional arguments:
  INPUT                 The input parameter file in json or yaml format, which
                        should be consistent with the original model parameter
                        file

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The original frozen model, which will be compressed by
                        the deepmd-kit
  -o OUTPUT, --output OUTPUT
                        The compressed model
  -e EXTRAPOLATE, --extrapolate EXTRAPOLATE
                        The scale of model extrapolation
  -s STRIDE, --stride STRIDE
                        The uniform stride of tabulation's first table, the
                        second table will use 10 * stride as it's uniform
                        stride
  -f FREQUENCY, --frequency FREQUENCY
                        The frequency of tabulation overflow check(If the
                        input environment matrix overflow the first or second
                        table range). By default do not check the overflow
  -d FOLDER, --folder FOLDER
                        path to checkpoint folder
```
**Parameter explanation**

Model compression, which including tabulating the embedding-net.
The table is composed of fifth-order polynomial coefficients and is assembled from two sub-tables. The first sub-table takes the stride(parameter) as it's uniform stride, while the second sub-table takes 10 * stride as it's uniform stride.
The range of the first table is automatically detected by deepmd-kit, while the second table ranges from the first table's upper boundary(upper) to the extrapolate(parameter) * upper.
Finally, we added a check frequency parameter. It indicates how often the program checks for overflow(if the input environment matrix overflow the first or second table range) during the MD inference.

**Justification of model compression**

Model compression, with little loss of accuracy, can greatly speed up MD inference time. According to different simulation systems and training parameters, the speedup can reach more than 10 times at both CPU and GPU devices. At the same time, model compression can greatly change the memory usage, reducing as much as 20 times under the same hardware conditions.

**Acceptable original model version**

The model compression method requires that the version of DeePMD-kit used in original model generation should be 1.3 or above. If one has a frozen 1.2 model, one can first use the convenient conversion interface of DeePMD-kit-v1.2.4 to get a 1.3 executable model.(eg: ```dp convert-to-1.3 -i frozen_1.2.pb -o frozen_1.3.pb```) 

## Model inference 

### Python interface
One may use the python interface of DeePMD-kit for model inference, an example is given as follows
```python
from deepmd.infer import DeepPot
import numpy as np
dp = DeepPot('graph.pb')
coord = np.array([[1,0,0], [0,0,1.5], [1,0,3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1,0,1]
e, f, v = dp.eval(coord, cell, atype)
```
where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.

Furthermore, one can use the python interface to calulate model deviation.
```python
from deepmd.infer import calc_model_devi
from deepmd.infer import DeepPot as DP
import numpy as np

coord = np.array([[1,0,0], [0,0,1.5], [1,0,3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1,0,1]
graphs = [DP("graph.000.pb"), DP("graph.001.pb")]
model_devi = calc_model_devi(coord, cell, atype, graphs)
```

### C++ interface
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
gcc infer_water.cpp -D HIGH_PREC -L $deepmd_root/lib -L $tensorflow_root/lib -I $deepmd_root/include -I $tensorflow_root/lib -Wl,--no-as-needed -ldeepmd_op -ldeepmd -ldeepmd_cc -ltensorflow_cc -ltensorflow_framework -lstdc++ -Wl,-rpath=$deepmd_root/lib -Wl,-rpath=$tensorflow_root/lib -o infer_water
```
and then run the program:
```sh
./infer_water
```

## Run MD

### Run MD with LAMMPS
Include deepmd in the pair_style

#### Syntax
```
pair_style deepmd models ... keyword value ...
```
- deepmd = style of this pair_style
- models = frozen model(s) to compute the interaction. If multiple models are provided, then the model deviation will be computed
- keyword = *out_file* or *out_freq* or *fparam* or *atomic* or *relative*
<pre>
    <i>out_file</i> value = filename
        filename = The file name for the model deviation output. Default is model_devi.out
    <i>out_freq</i> value = freq
        freq = Frequency for the model deviation output. Default is 100.
    <i>fparam</i> value = parameters
        parameters = one or more frame parameters required for model evaluation.
    <i>atomic</i> = no value is required. 
        If this keyword is set, the model deviation of each atom will be output.
    <i>relative</i> value = level
        level = The level parameter for computing the relative model deviation
</pre>

#### Examples
```
pair_style deepmd graph.pb
pair_style deepmd graph.pb fparam 1.2
pair_style deepmd graph_0.pb graph_1.pb graph_2.pb out_file md.out out_freq 10 atomic relative 1.0
```

#### Description
Evaluate the interaction of the system by using [Deep Potential][DP] or [Deep Potential Smooth Edition][DP-SE]. It is noticed that deep potential is not a "pairwise" interaction, but a multi-body interaction. 

This pair style takes the deep potential defined in a model file that usually has the .pb extension. The model can be trained and frozen by package [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit).

The model deviation evalulate the consistency of the force predictions from multiple models. By default, only the maximal, minimal and averge model deviations are output. If the key `atomic` is set, then the model deviation of force prediction of each atom will be output.

By default, the model deviation is output in absolute value. If the keyword `relative` is set, then the relative model deviation will be output. The relative model deviation of the force on atom `i` is defined by
```math
           |Df_i|
Ef_i = -------------
       |f_i| + level
```
where `Df_i` is the absolute model deviation of the force on atom `i`, `|f_i|` is the norm of the the force and `level` is provided as the parameter of the keyword `relative`.


#### Restrictions
- The `deepmd` pair style is provided in the USER-DEEPMD package, which is compiled from the DeePMD-kit, visit the [DeePMD-kit website](https://github.com/deepmodeling/deepmd-kit) for more information.


#### Long-range interaction
The reciprocal space part of the long-range interaction can be calculated by LAMMPS command `kspace_style`. To use it with DeePMD-kit, one writes 
```bash
pair_style	deepmd graph.pb
pair_coeff
kspace_style	pppm 1.0e-5
kspace_modify	gewald 0.45
```
Please notice that the DeePMD does nothing to the direct space part of the electrostatic interaction, because this part is assumed to be fitted in the DeePMD model (the direct space cut-off is thus the cut-off of the DeePMD model). The splitting parameter `gewald` is modified by the `kspace_modify` command.

### Run path-integral MD with i-PI
The i-PI works in a client-server model. The i-PI provides the server for integrating the replica positions of atoms, while the DeePMD-kit provides a client named `dp_ipi` that computes the interactions (including energy, force and virial). The server and client communicates via the Unix domain socket or the Internet socket. Installation instructions of i-PI can be found [here](install.md#install-i-pi). The client can be started by
```bash
i-pi input.xml &
dp_ipi water.json
```
It is noted that multiple instances of the client is allow for computing, in parallel, the interactions of multiple replica of the path-integral MD.

`water.json` is the parameter file for the client `dp_ipi`, and an example is provided:
```json
{
    "verbose":		false,
    "use_unix":		true,
    "port":		31415,
    "host":		"localhost",
    "graph_file":	"graph.pb",
    "coord_file":	"conf.xyz",
    "atom_type" : {
	"OW":		0, 
	"HW1":		1,
	"HW2":		1
    }
}
```
The option **`use_unix`** is set to `true` to activate the Unix domain socket, otherwise, the Internet socket is used.

The option **`port`** should be the same as that in input.xml:
```xml
<port>31415</port>
```

The option **`graph_file`** provides the file name of the frozen model.

The `dp_ipi` gets the atom names from an [XYZ file](https://en.wikipedia.org/wiki/XYZ_file_format) provided by **`coord_file`** (meanwhile ignores all coordinates in it), and translates the names to atom types by rules provided by **`atom_type`**.

### Use deep potential with ASE

Deep potential can be set up as a calculator with ASE to obtain potential energies and forces.
```python
from ase import Atoms
from deepmd.calculator import DP

water = Atoms('H2O',
              positions=[(0.7601, 1.9270, 1),
                         (1.9575, 1, 1),
                         (1., 1., 1.)],
              cell=[100, 100, 100],
              calculator=DP(model="frozen_model.pb"))
print(water.get_potential_energy())
print(water.get_forces())
```

Optimization is also available:
```python
from ase.optimize import BFGS
dyn = BFGS(water)
dyn.run(fmax=1e-6)
print(water.get_positions())
```

## Known limitations
If you use deepmd-kit in a GPU environment, the acceptable value range of some variables are additionally restricted compared to the CPU environment due to the software's GPU implementations: 
1. The number of atom type of a given system must be less than 128.
2. The maximum distance between an atom and it's neighbors must be less than 128. It can be controlled by setting the rcut value of training parameters.
3. Theoretically, the maximum number of atoms that a single GPU can accept is about 10,000,000. However, this value is actually limited by the GPU memory size currently, usually within 1000,000 atoms even at the model compression mode.
4. The total sel value of training parameters(in model/descriptor section) must be less than 4096.

[DP]:https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[DP-SE]:https://dl.acm.org/doi/10.5555/3327345.3327356
