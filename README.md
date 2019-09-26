<span style="font-size:larger;">DeePMD-kit Manual</span>
========


# Table of contents
- [About DeePMD-kit](#about-deepmd-kit)
 	- [Highlighted features](#highlighted-features)
 	- [Code structure](#code-structure)
 	- [License and credits](#license-and-credits)
 	- [Deep Potential in a nutshell](#deep-potential-in-a-nutshell)
- [Download and install](#download-and-install)
    - [Easy installation methods](#easy-installation-methods)
    - [Install the python interaction](#install-the-python-interface)
	    - [Install the Tensorflow's python interface](#install-the-tensorflows-python-interface)
	    - [Install the DeePMD-kit's python interface](#install-the-deepmd-kits-python-interface)
    - [Install the C++ interaction](#install-the-c-interface)
	    - [Install the Tensorflow's C++ interface](#install-the-tensorflows-c-interface)    
	    - [Install the DeePMD-kit's C++ interface](#install-the-deepmd-kits-c-interface)
	    - [Install LAMMPS's DeePMD-kit module](#install-lammpss-deepmd-kit-module)
	    - [Build DeePMD-kit with GPU support](#build-deepmd-kit-with-gpu-support)
- [Use DeePMD-kit](#use-deepmd-kit)
	- [Prepare data](#prepare-data)
	- [Train a model](#train-a-model)
	    - [The DeePMD model](#the-deepmd-model)
	    - [The DeepPot-SE model](#the-deeppot-se-model)
	- [Freeze and test a model](#freeze-and-test-a-model)
	- [Model inference](#model-inference)
	- [Run MD with Lammps](#run-md-with-lammps)
	    - [Include deepmd in the pair style](#include-deepmd-in-the-pair-style)
	    - [Long-range interaction](#long-range-interaction)
	- [Run path-integral MD with i-PI](#run-path-integral-md-with-i-pi)
- [Troubleshooting](#troubleshooting)

# About DeePMD-kit
DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics (MD). This brings new hopes to addressing the accuracy-versus-efficiency dilemma in molecular simulations. Applications of DeePMD-kit span from finite molecules to extended systems and from metallic systems to chemically bonded systems. 

## Highlighted features
* **interfaced with TensorFlow**, one of the most popular deep learning frameworks, making the training process highly automatic and efficient.
* **interfaced with high-performance classical MD and quantum (path-integral) MD packages**, i.e., LAMMPS and i-PI, respectively. 
* **implements the Deep Potential series models**, which have been successfully applied to  finite and extended systems including organic molecules, metals, semiconductors, and insulators, etc.
* **implements MPI and GPU supports**, makes it highly efficient for high performance parallel and distributed computing.
* **highly modularized**, easy to adapt to different descriptors for deep learning based potential energy models.

## Code structure
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


## License and credits
The project DeePMD-kit is licensed under [GNU LGPLv3.0](./LICENSE).
If you use this code in any future publications, please cite this using 
``Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. "DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics." Computer Physics Communications 228 (2018): 178-184.``

## Deep Potential in a nutshell
The goal of Deep Potential is to employ deep learning techniques and realize an inter-atomic potential energy model that is general, accurate, computationally efficient and scalable. The key component is to respect the extensive and symmetry-invariant properties of a potential energy model by assigning a local reference frame and a local environment to each atom. Each environment contains a finite number of atoms, whose local coordinates are arranged in a symmetry preserving way. These local coordinates are then transformed, through a sub-network, to a so-called *atomic energy*. Summing up all the atomic energies gives the potential energy of the system.

The initial proof of concept is in the [Deep Potential][1] paper, which employed an approach that was devised to train the neural network model with the potential energy only. With typical *ab initio* molecular dynamics (AIMD) datasets this is insufficient to reproduce the trajectories. The Deep Potential Molecular Dynamics ([DeePMD][2]) model overcomes this limitation. In addition, the learning process in DeePMD improves significantly over the Deep Potential method thanks to the introduction of a flexible family of loss functions. The NN potential constructed in this way reproduces accurately the AIMD trajectories, both classical and quantum (path integral), in extended and finite systems, at a cost that scales linearly with system size and is always several orders of magnitude lower than that of equivalent AIMD simulations.

Although being highly efficient, the original Deep Potential model satisfies the extensive and symmetry-invariant properties of a potential energy model at the price of introducing discontinuities in the model. This has negligible influence on a trajectory from canonical sampling but might not be sufficient for calculations of dynamical and mechanical properties. These points motivated us to develop the Deep Potential-Smooth Edition ([DeepPot-SE][3]) model, which replaces the non-smooth local frame with a smooth and adaptive embedding network. DeepPot-SE shows great ability in modelling many kinds of systems that are of interests in the fields of physics, chemistry, biology, and materials science.

In addition to building up potential energy models, DeePMD-kit can also be used to build up coarse-grained models. In these models, the quantity that we want to parametrize is the free energy, or the coarse-grained potential, of the coarse-grained particles. See the [DeePCG paper][4] for more details.

# Download and install
Please follow our [github](https://github.com/deepmodeling/deepmd-kit) webpage to see the latest released version and development version.
## Easy installation methods
A docker for installing the DeePMD-kit on CentOS 7 is available [here](https://github.com/frankhan91/deepmd-kit_docker). We are currently working on installation methods using the `conda` package management system and `pip` tools. Hope these will come out soon.

## Install the python interface 

### Install the Tensorflow's python interface
We follow the virtual environment approach to install the tensorflow's Python interface. The full instruction can be found on [the tensorflow's official website](https://www.tensorflow.org/install/pip). Now we assume that the Python interface will be installed to virtual environment directory `$tensorflow_venv`
```bash
virtualenv -p python3 $tensorflow_venv
source $tensorflow_venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow==1.8.0
```
If one needs the GPU support of deepmd-kit, the GPU version of tensorflow should be installed by
```bash
pip install --upgrade tensorflow-gpu==1.8.0
```
To verify the installation, 
```bash
python -c "import tensorflow as tf; sess=tf.Session(); print(sess.run(tf.reduce_sum(tf.random_normal([1000, 1000]))))"
```
One should remember to activate the virtual environment every time he/she uses deepmd-kit.

### Install the DeePMD-kit's python interface
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
Then execute
```bash
pip install .
```
To test the installation, one may execute
```bash
python -m deepmd -h
```
It will print the help information like
```text
usage: __main__.py [-h] {config,train,freeze,test} ...

deepmd-kit

optional arguments:
  -h, --help            show this help message and exit

Valid subcommands:
  {config,train,freeze,test}
    config              fast configuration of parameter file for smooth model
    train               train a model
    freeze              freeze the model
    test                test the model
```

## Install the C++ interface 

### Install the Tensorflow's C++ interface

If one does not need to use DeePMD-kit with Lammps or I-Pi, then the python interface does everything and he/she can safely skip this section. 

The C++ interface of DeePMD-kit was tested with compiler gcc >= 4.9.

Firstly the C++ interface of Tensorflow should be installed. It is noted that the version of Tensorflow C++ interface should be in consistent with that of the python interface. We assume that you have followed our instruction and installed tensorflow python interface 1.8.0, i.e.
```bash
pip install --upgrade tensorflow==1.8.0
```
then you may follow [the instruction here](doc/install-tf.1.8.md) to install the corresponding C++ interface. 

Or you have installed Tensorflow's python interface 1.12.0, you may follow [here](doc/install-tf.1.12.md) to install the corresponding C++ interface.

### Install the DeePMD-kit's C++ interface
Now goto the source code directory of DeePMD-kit and make a build place.
```bash
cd $deepmd_source_dir/source
mkdir build 
cd build
```
I assume you want to install DeePMD-kit into path `$deepmd_root`, then execute cmake
```bash
cmake -DTENSORFLOW_ROOT=$tensorflow_root -DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```
If you want to build deepmd-kit with cuda-toolkit support, then execute cmake
```bash
cmake -DTF_GOOGLE_BIN=true -DUSE_CUDA_TOOLKIT=true -DTENSORFLOW_ROOT=$tensorflow_root \
-DCMAKE_INSTALL_PREFIX=$deepmd_root ..
```
If you built the tensorflow's Python interface by gcc>=5.0, then remove the option `-DTF_GOOGLE_BIN=true`. If the cmake has executed successfully, then 
```bash
make
make install
```
If everything works fine, you will have the following executable and libraries installed in `$deepmd_root/bin` and `$deepmd_root/lib`
```bash
$ ls $deepmd_root/bin
dp_ipi
$ ls $deepmd_root/lib
libdeepmd_ipi.so  libdeepmd_op.so  libdeepmd.so
```

### Install LAMMPS's DeePMD-kit module
DeePMD-kit provide module for running MD simulation with LAMMPS. Now make the DeePMD-kit module for LAMMPS.
```bash
cd $deepmd_source_dir/source/build
make lammps
```
DeePMD-kit will generate a module called `USER-DEEPMD` in the `build` directory. Now download your favorite LAMMPS code, and uncompress it (I assume that you have downloaded the tar `lammps-stable.tar.gz`)
```bash
cd /some/workspace
tar xf lammps-stable.tar.gz
```
The source code of LAMMPS is stored in directory, for example `lammps-31Mar17`. Now go into the LAMMPS code and copy the DeePMD-kit module like this
```bash
cd lammps-31Mar17/src/
cp -r $deepmd_source_dir/source/build/USER-DEEPMD .
```
Now build LAMMPS
```bash
make yes-user-deepmd
make mpi -j4
```
The option `-j4` means using 4 processes in parallel. You may want to use a different number according to your hardware. 

If everything works fine, you will end up with an executable `lmp_mpi`.

The DeePMD-kit module can be removed from LAMMPS source code by 
```bash
make no-user-deepmd
```

### Build DeePMD-kit with GPU support
If your system has a NVIDIA GPU, you can build TensorFlow with GPU support, which will be inherited by DeePMD-kit and LAMMPS. To achieve this, please carefully check the webpage [Install TensorFlow from Source](https://www.tensorflow.org/install/install_sources) and look for the GPU version. In particular, you have to make sure that the required NVIDIA softwares, namely [CUDA Toolkit](https://developer.nvidia.com/cuda-zone), [GPU drivers](https://www.nvidia.com/driver), and [cuDNN SDK](https://developer.nvidia.com/cudnn), must be installed on your system.

To install TensorFlow with GPU support, all the installation steps will be the same as the non-GPU version, except that one may allow the GPU option when doing `configure`, e.g.,
```bash
Do you wish to build TensorFlow with CUDA support? [y/N] Y
CUDA support will be enabled for TensorFlow
Do you want to use clang as CUDA compiler? [y/N]
nvcc will be used as CUDA compiler
Please specify the CUDA SDK version you want to use. [Leave empty to default to CUDA 9.0]: 9.0
Please specify the location where CUDA 9.0 toolkit is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
Please specify the cuDNN version you want to use. [Leave empty to default to cuDNN 7.0]: 7
Please specify the location where cuDNN 7 library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]:
Please specify a list of comma-separated CUDA compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
```

After sucessfully installing TensorFlow with GPU support, you should install DeePMD, LAMMPS, etc., in the same way of the non-GPU version. Sometimes you may need to explicitly tell the compiler the place of the CUDA Toolkit and cuDNN libraries, i.e.,
```bash
 $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuda_toolkit/lib64
 $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cudnn/lib64
```

# Use DeePMD-kit
In this text, we will call the deep neural network that is used to represent the interatomic interactions (Deep Potential) the **model**. The typical procedure of using DeePMD-kit is 

1. Prepare data
2. Train a model
3. Freeze the model
4. MD runs with the model (Native MD code or LAMMPS)

## Prepare data
One needs to provide the following information to train a model: the atom type, the simulation box, the atom coordinate, the atom force, system energy and virial. A snapshot of a system that contains these information is called a **frame**. We use the following convention of units:

Property| Unit
---	| :---:
Time	| ps
Length	| Å
Energy	| eV
Force	| eV/Å
Pressure| Bar

The frames of the system are stored in two formats. A raw file is a plain text file with each information item written in one file and one frame written on one line. The default files that provide box, coordinate, force, energy and virial are `box.raw`, `coord.raw`, `force.raw`, `energy.raw` and `virial.raw`, respectively. *We recommend you use these file names*. Here is an example of force.raw:
```bash
$ cat force.raw
-0.724  2.039 -0.951  0.841 -0.464  0.363
 6.737  1.554 -5.587 -2.803  0.062  2.222
-1.968 -0.163  1.020 -0.225 -0.789  0.343
```
This `force.raw` contains 3 frames with each frame having the forces of 2 atoms, thus it has 3 lines and 6 columns. Each line provides all the 3 force components of 2 atoms in 1 frame. The first three numbers are the 3 force components of the first atom, while the second three numbers are the 3 force components of the second atom. The coordinate file `coord.raw` is organized similarly. In `box.raw`, the 9 components of the box vectors should be provided on each line. In `virial.raw`, the 9 components of the virial tensor should be provided on each line. The number of lines of all raw files should be identical.

We assume that the atom types do not change in all frames. It is provided by `type.raw`, which has one line with the types of atoms written one by one. The atom types should be integers. For example the `tyep.raw` of a system that has 2 atoms with 0 and 1:
```bash
$ cat type.raw
0 1
```

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
It generates three sets `set.000`, `set.001` and `set.002`, with each set contains 2000 frames. The last set (`set.002`) is used as testing set, while the rest sets (`set.000` and `set.001`) are used as training sets. One do not need to take care of the binary data files in each of the `set.*` directories. The path containing `set.*` and `type.raw` is called a *system*. 

## Train a model
### The DeePMD model
The method of training is explained in our [DeePMD paper][2]. With the source code we provide a small training dataset taken from 400 frames generated by NVT ab-initio water MD trajectory with 300 frames for training and 100 for testing. [An example training parameter file](./examples/train/water.json) is provided. One can try with the training by
```bash
$ cd $deepmd_source_dir/examples/train/
$ python -m deepmd train water.json
```
`water.json` is the `json` format parameter file that controls the training. The components of the `water.json` are
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
    "fitting_neuron":	[240, 120, 60, 30, 10],

    "_comment": " training controls",
    "systems":		["../data/water/"],
    "set_prefix":	"set",    
    "stop_batch":	1000000,
    "batch_size":	4,
    "start_lr":	0.001,
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

    "_comment":	"that's all"
}
```

The option **`rcut`** is the cut-off radius for neighbor searching. The `sel_a` and `sel_r` are the maximum selected numbers of fully-local-coordinate and radial-only-coordinate atoms from the neighbor list, respectively. `sel_a + sel_r` should be larger than the maximum possible number of neighbors in the cut-off radius. `sel_a` and `sel_r` are vectors, the length of the vectors are same as the number of atom types in the system. `sel_a[i]` and `sel_r[i]` denote the selected number of neighbors of type `i`.

The option **`axis_rule`** specifies how to make the axis for the local coordinate of each atom. For each atom type, 6 integers should be provided. The first three for the first axis, while the last three for the second axis. Within the three integers, the first one specifies if the axis atom is fully-local-coordinated (`0`) or radial-only-coordinated (`1`). The second integer specifies the type of the axis atom. If this number is less than 0, saying `t < 0`, then this axis exclude atom of type `-(t+1)`. If the third integer is, saying `s`, then the axis atom is the `s`th nearest neighbor satisfying the previous two conditions. 

The option **`fitting_neuron`** (deprecated name **`n_neuron`**) is an integer vector that determines the shape the neural network. The size of the vector is identical to the number of hidden layers of the network. From left to right the members denote the sizes of each hidden layers from input end to the output end, respectively. If two neighboring layers are of the same size, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is build between them. If the option **`fitting_resnet_dt`** is set `true`, then a timestep is used in the ResNet.

The option **`systems`** provide location of the systems (path to `set.*` and `type.raw`). It is a vector, thus DeePMD-kit allows you to provide multiple systems. DeePMD-kit will train the model with the systems in the vector one by one in a cyclic manner. **It is warned that the example water data (in folder `examples/data/water`) is of very limited amount, is provided only for testing purpose, and should not be used to train a productive model.**

The option **`batch_size`** specifies the number of frames in each batch. 
The option **`stop_batch`** specifies the total number of batches will be used in the training.
The option **`start_lr`**, **`decay_rate`** and **`decay_steps`** specify how the learning rate changes. For example, the `t`th batch will be trained with learning rate:
```math
lr(t) = start_lr * decay_rate ^ ( t / decay_steps )
```

The options **`start_pref_e`**, **`limit_pref_e`**, **`start_pref_f`**, **`limit_pref_f`**, **`start_pref_v`** and **`limit_pref_v`** determine how the prefactors of energy error, force error and virial error changes in the loss function (see the appendix of the [DeePMD paper][2] for details). Taking the prefactor of force error for example, the prefactor at batch `t` is
```math
w_f(t) = start_pref_f * ( lr(t) / start_lr ) + limit_pref_f * ( 1 - lr(t) / start_lr )
```
Since we do not have virial data, the virial prefactors `start_pref_v` and `limit_pref_v` are set to 0.

The option **`seed`** specifies the random seed for neural network initialization. If not provided, the `seed` will be initialized with `None`.

During the training, the error of the model is tested every **`disp_freq`** batches with **`numb_test`** frames from the last set in the **`systems`** directory on the fly, and the results are output to **`disp_file`**. A typical `disp_file` looks like
```bash
# batch      l2_tst    l2_trn    l2_e_tst  l2_e_trn    l2_f_tst  l2_f_trn         lr
      0    2.67e+01  2.57e+01    2.21e-01  2.22e-01    8.44e-01  8.12e-01    1.0e-03
    100    6.14e+00  5.40e+00    3.01e-01  2.99e-01    1.93e-01  1.70e-01    1.0e-03
    200    5.02e+00  4.49e+00    1.53e-01  1.53e-01    1.58e-01  1.42e-01    1.0e-03
    300    4.36e+00  3.71e+00    7.32e-02  7.27e-02    1.38e-01  1.17e-01    1.0e-03
    400    4.04e+00  3.29e+00    3.16e-02  3.22e-02    1.28e-01  1.04e-01    1.0e-03
```
The first column displays the number of batches. The second and third columns display the loss function evaluated by `numb_test` frames randomly chosen from the test set and that evaluated by the current training batch, respectively. The fourth and fifth columns display the RMS energy error (normalized by number of atoms) evaluated by `numb_test` frames randomly chosen from the test set and that evaluated by the current training batch, respectively. The sixth and seventh columns display the RMS force error (component-wise) evaluated by `numb_test` frames randomly chosen from the test set and that evaluated by the current training batch, respectively. The last column displays the current learning rate.

Checkpoints will be written to files with prefix **`save_ckpt`** every **`save_freq`** batches. If **`restart`** is set to `true`, then the training will start from the checkpoint named **`load_ckpt`**, rather than from scratch.

Several command line options can be passed to `python -m deepmd train`, which can be checked with
```bash
$ python -m deepmd train --help
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

### The DeepPot-SE model
The smooth version of DeePMD, or the [DeepPot-SE model][3], can also be trained by DeePMD-kit. [An example training parameter file](./examples/train/water_smth.json) is provided. One can try with the training by
```bash
$ cd $deepmd_source_dir/examples/train/
$ python -m deepmd train water_smth.json
```
The difference between the standard and smooth DeePMD models lies in the model parameters:
```json
{
    "use_smooth":        true,
    "sel_a":             [46, 92],
    "rcut_smth":         5.80,
    "rcut":              6.00,
    "filter_neuron":     [25, 50, 100],
    "filter_resnet_dt":  false,
    "axis_neuron":       16,
    "fitting_neuron":    [240, 240, 240],
    "fitting_resnet_dt": true,
    "_comment":          "that's all"
}
```
The `sel_r` option is skipped by the smooth version and the model use fully-local-coordinate for all neighboring atoms. The `sel_a` should larger than the maximum possible number of neighbors in the cut-off radius `rcut`. 

The descriptors will decay smoothly from **`rcut_smth`** to the cutoff radius `rcut`.

The **`filter_neuron`** provides the size of the filter network (also called local-embedding network). If the size of the next layer is the same or twice as the previous layer, then a skip connection is build (ResNet). The **`filter_resnet_dt`** tells if a timestep is used in the skip connection. By default it is `false`. **`axis_neuron`** (deprecated name `n_axis_neuron`) specifies the number of axis filter, which should be much smaller than the size of the last layer of the filter network.

**`fitting_neuron`** (deprecated name `n_neuron`) specifies the fitting network. If the size of the next layer is the same as the previous layer, then a skip connection is build (ResNet). **`fitting_resnet_dt`** (deprecated name `resnet_dt`) tells if a timestep is used in the skip connection. By default it is `true`. 


## Freeze and test a model
The trained neural network is extracted from a checkpoint and dumped into a database. This process is called "freezing" a model. The idea and part of our code are from [Morgan](https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc). To freeze a model, typically one does
```bash
$ python -m deepmd freeze -o graph.pb
```
in the folder where the model is trained. The output database is called `graph.pb`.

The frozen model can be used in many ways. The most straightforward test can be performed using `python -m deepmd test`. Several command line options can be passed to `python -m deepmd test`, which can be checked with
```bash
$ python -m deepmd test --help
```
An explanation will be provided
```
usage: __main__.py test [-h] [-m MODEL] [-s SYSTEM] [-S SET_PREFIX]
                        [-n NUMB_TEST] [-r RAND_SEED] [--shuffle-test]
                        [-d DETAIL_FILE]

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

## Model inference 
One may use the python interface of DeePMD-kit for model inference, an example is given as follows
```python
import deepmd.DeepPot as DP
import numpy as np
dp = DP('graph.pb')
coord = np.array([[1,0,0], [0,0,1.5], [1,0,3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1,0,1]
e, f, v = dp.eval(coord, cell, atype)
```
where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.


## Run MD with LAMMPS
### Include deepmd in the pair style
Running an MD simulation with LAMMPS is simpler. In the LAMMPS input file, one needs to specify the pair style as follows
```bash
pair_style     deepmd graph.pb
pair_coeff     
```
where `graph.pb` is the file name of the frozen model. The `pair_coeff` should be left blank. It should be noted that LAMMPS counts atom types starting from 1, therefore, all LAMMPS atom type will be firstly subtracted by 1, and then passed into the DeePMD-kit engine to compute the interactions.

### Long-range interaction
The reciprocal space part of the long-range interaction can be calculated by LAMMPS command `kspace_style`. To use it with DeePMD-kit, one writes 
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


# Troubleshooting
In consequence of various differences of computers or systems, problems may occur. Some common circumstances are listed as follows. 
If other unexpected problems occur, you're welcome to contact us for help.

## Installation: inadequate versions of gcc/g++
Sometimes you may use a gcc/g++ of version <4.9. If you have a gcc/g++ of version > 4.9, say, 7.2.0, you may choose to use it by doing 
```bash
export CC=/path/to/gcc-7.2.0/bin/gcc
export CXX=/path/to/gcc-7.2.0/bin/g++
```

If, for any reason, for example, you only have a gcc/g++ of version 4.8.5, you can still compile all the parts of TensorFlow and most of the parts of DeePMD-kit. Native MD and i-Pi will be disabled automatically.

## Installation: build files left in DeePMD-kit
When you try to build a second time when installing DeePMD-kit, files produced before may contribute to failure. Thus, you may clear them by
```bash
cd build
rm -r *
```
and redo the `cmake` process.

## Training: TensorFlow abi binary cannot be found when doing training
If you confront such kind of error: 

```
$deepmd_root/lib/deepmd/libop_abi.so: undefined symbol:
_ZN10tensorflow8internal21CheckOpMessageBuilder9NewStringB5cxx11Ev
```

you may set `-DTF_GOOGLE_BIN=true` in the process of `cmake`.

Another possible reason might be the large gap between the python version of TensorFlow and the TensorFlow c++ interface.


## MD: cannot run LAMMPS after installing a new version of DeePMD-kit
This typically happens when you install a new version of DeePMD-kit and copy directly the generated `USER-DEEPMD` to a LAMMPS source code folder and re-install LAMMPS.

To solve this problem, it suffices to first remove `USER-DEEPMD` from LAMMPS source code by 
```bash
make no-user-deepmd
```
and then install the new `USER-DEEPMD`.

If this does not solve your problem, try to decompress the LAMMPS source tarball and install LAMMPS from scratch again, which typically should be very fast.


[1]: http://www.global-sci.com/galley/CiCP-2017-0213.pdf
[2]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[3]:https://arxiv.org/abs/1805.09003
[4]:https://aip.scitation.org/doi/full/10.1063/1.5027645
