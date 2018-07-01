# Table of contents

- [Install DeePMD-kit](#install-deepmd-kit)
	- [Install tensorflow's Python interface](#install-tensorflows-python-interface)
	- [Install tensorflow's C++ interface](#install-tensorflows-c-interface)
	- [Install xdrfile](#install-xdrfile)
	- [Install DeePMD-kit](#install-deepmd-kit)
	- [Install Lammps' DeePMD-kit module](#install-lammps-deepmd-kit-module)
- [Use DeePMD-kit](#use-deepmd-kit)
	- [Prepare data](#prepare-data)
	- [Train a model](#train-a-model)
	- [Freeze the model](#freeze-the-model)
	- [Run MD with Lammps](#run-md-with-lammps)
	- [Run path-integral MD with i-PI](#run-path-integral-md-with-i-pi)
	- [Run MD with native code](#run-md-with-native-code)
- [Code structure](#code-structure)
- [License](#license)

# Install DeePMD-kit
The installation of the DeePMD-kit is lengthy, but do not be panic. Just follow step by step. Wish you good luck.. 

A docker for installing the DeePMD-kit on CentOS 7 is available [here](https://github.com/frankhan91/deepmd-kit_docker).

## Install tensorflow's Python interface 
There are two ways of installing the Python interface of tensorflow, either [using google's binary](https://www.tensorflow.org/install/install_linux), or [installing from sources](https://www.tensorflow.org/install/install_sources). When you are using google's binary, do not forget to add the option `-DTF_GOOGLE_BIN=true` when building DeePMD-kit.

## Install tensorflow's C++ interface
Firstly get the source code of the tensorflow
```bash
cd /some/workspace
git clone https://github.com/tensorflow/tensorflow tensorflow
```
The DeePMD-kit works with tensorflow r1.4 -- r1.6. Now taking r1.4 for example:
```bash
cd tensorflow
git checkout r1.4
```
Please make sure you have the Bazel higher than version 0.5.4, otherwise, please [install it](https://docs.bazel.build/versions/master/install.html).

DeePMD-kit is compiled by cmake, so we need to compile and integrate tensorflow with cmake projects. The rest of this section basically follows [the instruction provided by Tuatini](http://tuatini.me/building-tensorflow-as-a-standalone-project/). Now execute
```bash
./configure
```
You will answer a list of questions that help configure the building of tensorflow. It is recommended to build for Python3. You may want to answer the question like this:
```bash
Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3
```
The library path for Python should be set accordingly.

Now build the shared library of tensorflow:
```bash
bazel build -c opt --verbose_failures //tensorflow:libtensorflow_cc.so
```
You may want to add options `--copt=-msse4.2`,  `--copt=-mavx`, `--copt=-mavx2` and `--copt=-mfma` to enable SSE4.2, AVX, AVX2 and FMA SIMD accelerations, respectively. It is noted that these options should be chosen according to the CPU architecture. If the RAM becomes an issue of your machine, you may limit the RAM usage by using `--local_resources 2048,.5,1.0`. 

Now I assume you want to install tensorflow in directory `$tensorflow_root`. Create the directory if it does not exists
```bash
mkdir -p $tensorflow_root
```
Before moving on, we need to compile the dependencies of tensorflow, including Protobuf, Eigen and nsync. Firstly, protobuf
```bash
mkdir /tmp/proto
tensorflow/contrib/makefile/download_dependencies.sh
cd tensorflow/contrib/makefile/downloads/protobuf/
./autogen.sh
./configure --prefix=/tmp/proto/
make
make install
```
Then Eigen
```bash
mkdir /tmp/eigen
cd ../eigen
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/eigen/ ../
make install
```
And nsync
```bash
mkdir /tmp/nsync
cd ../../nsync
mkdir build_dir
cd build_dir
cmake -DCMAKE_INSTALL_PREFIX=/tmp/nsync/ ../
make
make install
cd ../../../../../..
```
Now, copy the libraries to the tensorflow's installation directory:
```bash
mkdir $tensorflow_root/lib
cp bazel-bin/tensorflow/libtensorflow_cc.so $tensorflow_root/lib/
cp bazel-bin/tensorflow/libtensorflow_framework.so $tensorflow_root/lib/
cp /tmp/proto/lib/libprotobuf.a $tensorflow_root/lib/
cp /tmp/nsync/lib/libnsync.a $tensorflow_root/lib/
```
Then copy the headers
```bash
mkdir -p $tensorflow_root/include/tensorflow
cp -r bazel-genfiles/* $tensorflow_root/include/
cp -r tensorflow/cc $tensorflow_root/include/tensorflow
cp -r tensorflow/core $tensorflow_root/include/tensorflow
cp -r third_party $tensorflow_root/include
cp -r /tmp/proto/include/* $tensorflow_root/include
cp -r /tmp/eigen/include/eigen3/* $tensorflow_root/include
cp -r /tmp/nsync/include/*h $tensorflow_root/include
```
Now clean up the source files in the header directories:
```bash
cd $tensorflow_root/include
find . -name "*.cc" -type f -delete
```
The temporary installation directories for the dependencies can be removed:
```bash
rm -fr /tmp/proto /tmp/eigen /tmp/nsync
```

## Install xdrfile
xdrfile is a lib that read, compress and write the MD trajectories. Firstly get the source:
```bash
cd /some/workspace
wget ftp://ftp.gromacs.org/pub/contrib/xdrfile-1.1.4.tar.gz
```
I assume you want to install it in `$xdrfile_root`, then you will probably do
```bash
tar xvf xdrfile-1.1.4.tar.gz
cd xdrfile-1.1.4
./configure --prefix=$xdrfile_root
make 
make install
```

## Install DeePMD-kit
The DeePMD-kit was tested with compiler gcc >= 4.9.

Firstly clone the DeePMD-kit source code
```bash
cd /some/workspace
git clone https://github.com/deepmodeling/deepmd-kit.git deepmd-kit
```
If one downloads the .zip file from the github, then the default folder of source code would be `deepmd-kit-master` rather than `deepmd-kit`. For convenience, you may want to record the location of source to a variable, saying `deepmd_source_dir` by
```bash
cd deepmd-kit
deepmd_source_dir=`pwd`
```
Then goto the source code directory and make a build directory.
```bash
cd $deepmd_source_dir/source
mkdir build 
cd build
```
I assume you want to install DeePMD-kit into path `$deepmd_root`, then execute cmake
```bash
cmake -DXDRFILE_ROOT=$xdrfile_root -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```
If you are using google binary for tensorflow python interface, then you need to specify
```bash
cmake -DXDRFILE_ROOT=$xdrfile_root -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root -DTF_GOOGLE_BIN=true ..
```
If the cmake has executed successfully, then 
```bash
make
make install
```
If everything works fine, you will have the following executables installed in `$deepmd_root/bin`
```bash
$ ls $deepmd_root/bin
dp_frz  dp_ipi  dp_mdnn  dp_test  dp_train
```

## Install Lammps' DeePMD-kit module
DeePMD-kit provide module for running MD simulation with Lammps. Now make the DeePMD-kit module for lammps.
```bash
cd $deepmd_source_dir/source/build
make lammps
```
DeePMD-kit will generate a module called `USER-DEEPMD` in the `build` directory. Now download your favorite Lammps code, and uncompress it (I assume that you have downloaded the tar `lammps-stable.tar.gz`)
```bash
cd /some/workspace
tar xf lammps-stable.tar.gz
```
The source code of Lammps is store in directory, for example `lammps-31Mar17`. Now go into the lammps code and copy the DeePMD-kit module like this
```bash
cd lammps-31Mar17/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
```
Now build Lammps
```bash
make yes-user-deepmd
make mpi -j4
```
The option `-j4` means using 4 processes in parallel. You may want to be use a different number according to your hardware. 

If everything works fine, you will end up with an executable `lmp_mpi`.

The DeePMD-kit module can be removed from Lammps source code by 
```bash
make no-user-deepmd
```

# Use DeePMD-kit
In this text, we will call the deep neural network that is used to represent the interatomic interactions (Deep Potential) the **model**. The typical procedure of using DeePMD-kit is 

1. Prepare data
2. Train a model
3. Freeze the model
4. MD runs with the model (Native MD code or Lammps)

## Prepare data
One needs to provide the following information to train a model: the atom type, the simulation box, the atom coordinate, the atom force, system energy and virial. A snapshot of a system that contains these information is called a **frame**. We use the following convention of units:

Property| Unit
---	| :---:
Time	| ps
Length	| A
Energy	| eV
Force	| eV/A
Pressure| Bar

The frames of the system are stored in two formats. A raw file is a plain text file with each information item written in one file and one frame written on one line. The default files that provide box, coordinate, force, energy and virial are `box.raw`, `coord.raw`, `force.raw`, `energy.raw` and `virial.raw`, respectively. *We recommend you use these file names*. Here is an example of force.raw:
```bash
$ cat force.raw
-0.724  2.039 -0.951  0.841 -0.464  0.363
 6.737  1.554 -5.587 -2.803  0.062  2.222
-1.968 -0.163  1.020 -0.225 -0.789  0.343
```
This `force.raw` contains 3 frames with each frame having the forces of 2 atoms, thus it has 3 lines and 6 columns. Each line provides all the 3 force components of 2 atoms in 1 frame. The first three numbers are the 3 force components of the first atom, while the second three numbers are the 3 force components of the second atom. The coordinate file `coord.raw` is organized similarly. In `box.raw`, the 9 components of the box vectors should be provided on each line. In `virial.raw`, the 9 components of the virial tensor should be provided on each line. The number of lines of all raw files should be identical.

We assume that the atom types do not change in all frames. It is provide by `type.raw`, which has one line with the types of atoms written one by one. The atom types should be integers.

The second format is the data sets of `numpy` binary data that are directly used by the training program. User can use the script `$deepmd_source_dir/data/raw/raw_to_set.sh` to convert the prepared raw files to data sets. For example, if we have raw files that contains 6000 frames, 
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
It generates two sets `set.000`, `set.001` and `set.002`, with each set contains 2000 frames. The last set (`set.002`) is used as testing set, while the rest sets (`set.000` and `set.001`) are used as training sets. One do not need to take care the binary data files in each of the `set.*` directories. The path containing `set.*` and `type.raw` is called a *system*. 

## Train a model
### The standard DeePMD model
The method of training is explained in our [DeePMD paper][1]. With the source code we provide a small training dataset taken from 400 frames generated by NVT ab-initio water MD trajectory with 300 frames for training and 100 for testing. [An example training parameter file](./examples/train/water.json) is provided. One can try with the training by
```bash
$ cd $deepmd_source_dir/examples/train/
$ $deepmd_root/bin/dp_train water.json
```
`$deepmd_root/bin/dp_train` is the training program, and `water.json` is the `json` format parameter file that controls the training. The components of the `water.json` are
```json
{
    "_comment": " model parameters",
    "use_smooth":	false,
    "sel_a":		[16, 32],
    "sel_r":		[30, 60],
    "rcut":		6.00,
    "axis_rule":	[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    "_comment":	" default rule: []",
    "_comment":	" user defined rule: for each type provides two axes, ",
    "_comment":	"                    for each axis: (a_or_r, type, idx)",
    "_comment":	"                    if type < 0, exclude type -(type+1)",
    "_comment": "                    for water (O:0, H:1) it can be",
    "_comment": "                    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0]",
    "n_neuron":		[240, 120, 60, 30, 10],

    "_comment": " training controls",
    "systems":		["../data/water/"],
    "set_prefix":	"set",    
    "stop_batch":	1000000,
    "batch_size":	4,
    "start_lr":		0.001,
    "decay_steps":	5000,
    "decay_rate":	0.95,

    "start_pref_e":	0.02,
    "limit_pref_e":	8,
    "start_pref_f":	1000,
    "limit_pref_f":	1,
    "start_pref_v":	0,
    "limit_pref_v":	0,

    "seed":		1,

    "_comment": " display and restart",
    "_comment": " frequencies counted in batch",
    "disp_file":	"lcurve.out",
    "disp_freq":	100,
    "numb_test":	100,
    "save_freq":	100,
    "save_ckpt":	"model.ckpt",
    "load_ckpt":	"model.ckpt",
    "disp_training":	true,
    "time_training":	true,

    "_comment":		"that's all"
}
```

The option **`rcut`** is the cut-off radius for neighbor searching. The `sel_a` and `sel_r` are the maximum selected numbers of fully-local-coordinate and radial-only-coordinate atoms from the neighbor list, respectively. `sel_a + sel_r` should larger than the maximum possible number of neighbors in the cut-off radius. `sel_a` and `sel_r` are vectors, the length of the vectors are same as the number of atom types in the system. `sel_a[i]` and `sel_r[i]` denote the selected number of neighbors of type `i`.

The option **`axis_rule`** specifies how to make the axis for the local coordinate of each atom. For each atom type, 6 integers should be provided. The first three for the first axis, while the last three for the second axis. Within the three integers, the first one specifies if the axis atom is fully-local-coordinated (`0`) or radial-only-coordinated (`1`). The second integer specifies the type of the axis atom. If this number is less than 0, saying `t < 0`, then this axis exclude atom of type `-(t+1)`. If the third integer is, saying `s`, then the axis atom is the `s`th nearest neighbor satisfying the previous two conditions. 

The option **`n_neuron`** is an integer vector that determines the shape the neural network. The size of the vector is identical to the number of hidden layers of the network. From left to right the members denotes the size of each hidden layers from input end to the output end, respectively.

The option **`systems`** provide location of the systems (path to `set.*` and `type.raw`). It is a vector, thus DeePMD-kit allows you provide multiple systems. DeePMD-kit will train the model with the systems in the vector one by one in a cyclic manner.

The option **`batch_size`** specifies the number of frames in each batch. 
The option **`stop_batch`** specifies the total number of batches will be used in the training.
The option **`start_lr`**, **`decay_rate`** and **`decay_steps`** specify how the learning rate changes. For example, the `t`th batch will be trained with learning rate:
```math
lr(t) = start_lr * decay_rate ^ ( t / decay_steps )
```

The options **`start_pref_e`**, **`limit_pref_e`**, **`start_pref_f`**, **`limit_pref_f`**, **`start_pref_v`** and **`limit_pref_v`** determine how the prefactors of energy error, force error and virial error changes in the loss function (see the appendix of the [DeePMD paper][1] for details). Taking the prefactor of force error for example, the prefactor at batch `t` is
```math
w_f(t) = start_pref_f * ( lr(t) / start_lr ) + limit_pref_f * ( 1 - lr(t) / start_lr )
```
Since we do not have virial data, the virial prefactors `start_pref_v` and `limit_pref_v` are set to 0.

The option **`seed`** specifies the random seed for neural network initialization. If not provided, the `seed` will be initialized with `None`.

During the training, the error of the model is tested every **`disp_freq`** batches with **`numb_test`** frames from the last set in the **`systems`** directory on the fly, and the results are output to **`disp_file`**. 

Checkpoints will be written to files with prefix **`save_ckpt`** every **`save_freq`** batches. If **`restart`** is set to `true`, then the training will start from the checkpoint named **`load_ckpt`**, rather than from scratch.

Several command line options can be passed to `dp_train`, this can be checked with
```bash
$ $deepmd_root/bin/dp_train --help
```
An explanation will be provided
```
positional arguments:
  INPUT                 the input json database

optional arguments:
  -h, --help            show this help message and exit
  -t INTER_THREADS, --inter-threads INTER_THREADS
                        With default value 0. Setting the "inter_op_parallelism_threads" key for the tensorflow, the "intra_op_parallelism_threads" will be set by the env variable OMP_NUM_THREADS
  --init-model INIT_MODEL
                        Initialize a model by the provided checkpoint
  --restart RESTART     Restart the training from the provided checkpoint
```
The keys `intra_op_parallelism_threads` and `inter_op_parallelism_threads` are Tensorflow configurations for multithreading, which are explained [here](https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu). Skipping `-t` and `OMP_NUM_THREADS` leads to the default setting of these keys in the Tensorflow.

**`--init-model model.ckpt`**, for example, initializes the model training with an existing model that is stored in the checkpoint `model.ckpt`, the network architectures should match.

**`--restart model.ckpt`**, continues the training from the checkpoint `model.ckpt`.

### The smooth DeePMD model
The smooth version of DeePMD can be trained by the DeePMD-kit. [An example training parameter file](./examples/train/water_smth.json) is provided. One can try with the training by
```bash
$ cd $deepmd_source_dir/examples/train/
$ $deepmd_root/bin/dp_train water_smth.json
```
The difference between the standard and smooth DeePMD models lies in the model parameters:
```json
    "use_smooth":	true,
    "sel_a":		[46, 92],
    "rcut_smth":	5.80,
    "rcut":		6.00,
    "filter_neuron":	[25, 50, 100],
    "filter_resnet_dt":	false,
    "n_axis_neuron":	16,
    "n_neuron":		[240, 240, 240],
    "resnet_dt":	true,
```
The `sel_r` option is skipped by the smooth version and the model use fully-local-coordinate for all neighboring atoms. The `sel_a` should larger than the maximum possible number of neighbors in the cut-off radius `rcut`. 

The descriptors will decay smoothly from **`rcut_smth`** to the cutoff radius `rcut`.

**`filter_neuron`** provides the size of the filter network (also called local-embedding network). If the size of the next layer is the same or twice as the previous layer, then a skip connection is build (ResNet). **`filter_resnet_dt`** tells if a timestep is used in the skip connection. By default it is `false`. **`n_axis_neuron`** specifies the number of axis filter, which should be much smaller than the size of the last layer of the filter network.

**`n_neuron`** specifies the fitting network. If the size of the next layer is the same as the previous layer, then a skip connection is build (ResNet). **`resnet_dt`** tells if a timestep is used in the skip connection. By default it is `true`. 


## Freeze the model
The trained neural network is extracted from a checkpoint and dumped into a database. This process is called "freeze" a model. Typically one does
```bash
$ $deepmd_root/bin/dp_frz -o graph.pb
```
in the folder where the model is trained. The output database is called `graph.pb`.

## Run MD with Lammps
Run an MD simulation with Lammps is simpler. In the Lammps input file, one needs to specify the pair style as follows
```bash
pair_style     deepmd graph.pb
pair_coeff     
```
where `graph.pb` is the file name of the frozen model. The `pair_coeff` should be left blank. It should be noted that Lammps counts atom types starting from 1, therefore, all Lammps atom type will be firstly subtracted by 1, and then passed into the DeePMD-kit engine to compute the interactions.

### With long-range interaction
The reciprocal space part of the long-range interaction can be calculated by lammps command `kspace_style`. To use it with DeePMD-kit, one writes 
```bash
pair_style	deepmd graph.pb
pair_coeff
kspace_style	pppm 1.0e-5
kspace_modify	gewald 0.45
```
Please notice that the DeePMD does nothing to the direct space part of the electrostatic interaction, because this part is assumed to be fitted in the DeePMD model (the direct space cut-off is thus the cut-off of the DeePMD model). The splitting parameter `gewald` is modified by the `kspace_modify` command.

## Run path-integral MD with i-PI
The i-PI works in a client-server model. The i-PI provides the server for integrating the replica positions of atoms, while the DeePMD-kit provides a client named `dp_ipi` that computes the interactions (including energy, force and virial). The server and client communicates via the Unix domain socket or the Internet socket. The client can be started by
```bash
$ dp_ipi water.json
```
It is noted that multiple instances of the client is allow for computing, in parallel, the interactions of multiple replica of the path-integral MD.

`water.json` is the parameter file for the client `dp_ipi`, and [an example](./examples/ipi/water.json) is provided:
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

The option **`graph_file`** provides the file name of the frozen model.

The `dp_ipi` gets the atom names from an [XYZ file](https://en.wikipedia.org/wiki/XYZ_file_format) provided by **`coord_file`** (meanwhile ignores all coordinates in it), and translates the names to atom types by rules provided by **`atom_type`**.


## Run MD with native code
DeePMD-kit provides a simple MD implementation that runs under either NVE or NVT ensemble. One needs to provide the following input files
```bash
$ ls
conf.gro  graph.pb  water.json
```
`conf.gro` is the file that provides the initial coordinates and/or velocities of all atoms in the system. It is of Gromacs `gro` format. Details of this format can be find in [this website](http://manual.gromacs.org/current/online/gro.html). It should be notice that the length unit of the `gro` format is **nm** rather than A.

`graph.pb` is the frozen model.

`water.json` is the parameter file that specifies how the MD runs. [An example parameter file](./examples/md/water.json) for water NVT simulation is provided. 
```json
{
    "conf_file":	"conf.gro",
    "conf_format":	"gro",
    "graph_file":	"graph.pb",
    "nsteps":		500000,
    "dt": 		5e-4,
    "ener_freq":	20,
    "ener_file":	"energy.out",
    "xtc_freq":		20,
    "xtc_file":		"traj.xtc",
    "trr_freq":		20,
    "trr_file":		"traj.trr",
    "print_force":	false,
    "T":		300,
    "tau_T":		0.1,
    "rand_seed":	2017,
    "atom_type" : {
	"OW":		0, 
	"HW1":		1,
	"HW2":		1
    },
    "atom_mass" : {
	"OW":		16, 
	"HW1":		1,
	"HW2":		1
    }
}
```
The options **`conf_file`**, **`conf_format`** and **`graph_file`** are self-explanatory. It should be noticed, again, the length unit is nm in the `gro` format file.

The option **`nsteps`** specifies the number of time steps of the MD simulation. The option **`dt`** specifies the timestep of the simulation. 

The options **`ener_file`** and **`ener_freq`** specify the energy output file and frequency. 

The options **`xtc_file`**, **`xtc_freq`**, **`trr_file`** and **`trr_freq`** are similar options that specify the output files and frequencies of the xtc and trr trajectory, respectively. When the frequencies are set to 0, the corresponding file will not be output. The instructions of the xtc and trr formats can be found in [xtc manual](http://manual.gromacs.org/online/xtc.html) and [trr manual](http://manual.gromacs.org/online/trr.html). It is noticed that the length unit in the xtc and trr files is **nm**.

If the option **`print_force`** is set to `true`, then the atomic force will be output.

The option **`T`** specifies the temperature of the simulation, and the option **`tau_T`** specifies the timescale of the thermostat. We implement the Langevin thermostat for the NVT simulation. **`rand_seed`** set the random seed of the random generator in the thermostat.

The **`atom_type`** set the type for the atoms in the system. The names of the atoms are those provided in the `conf_file` file. The **`atom_mass`** set the mass for the atoms. Again, the name of the atoms are those provided in the `conf_file`.


# Code structure
The code is organized as follows:
* `data/raw`: tools manipulating the raw data files.
* `examples`: example json parameter files.
* `source/3rdparty`: third-party packages used by DeePMD-kit.
* `source/cmake`: cmake scripts for building.
* `source/ipi`: source code of i-PI client.
* `source/lib`: source code of DeePMD-kit library.
* `source/lmp`: source code of Lammps module.
* `source/md`: source code of native MD.
* `source/op`: tensorflow op implementation. working with library.
* `source/scripts`: Python script for model freezing.
* `source/train`: Python modules and scripts for training and testing.


# License
The project DeePMD-kit is licensed under [GNU LGPLv3.0](./LICENSE)


[1]: https://arxiv.org/pdf/1707.09571.pdf




