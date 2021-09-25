<span style="font-size:larger;">DeePMD-kit Manual</span>
========
[![GitHub release](https://img.shields.io/github/release/deepmodeling/deepmd-kit.svg?maxAge=86400)](https://github.com/deepmodeling/deepmd-kit/releases)
[![doi:10.1016/j.cpc.2018.03.016](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2018.03.016-blue)](https://doi.org/10.1016/j.cpc.2020.107206)
[![offline packages](https://img.shields.io/github/downloads/deepmodeling/deepmd-kit/total?label=offline%20packages)](https://github.com/deepmodeling/deepmd-kit/releases)
[![conda install](https://img.shields.io/badge/downloads-9k%20total-green.svg?style=round-square&label=conda%20install)](https://anaconda.org/deepmodeling/deepmd-kit)
[![pip install](https://img.shields.io/pypi/dm/deepmd-kit?label=pip%20install)](https://pypi.org/project/deepmd-kit)
[![docker pull](https://img.shields.io/docker/pulls/deepmodeling/deepmd-kit)](https://hub.docker.com/r/deepmodeling/deepmd-kit)
[![Documentation Status](https://readthedocs.org/projects/deepmd/badge/)](https://deepmd.readthedocs.io/)

# Table of contents
- [About DeePMD-kit](#about-deepmd-kit)
 	- [Highlights in v2.0](#highlights-in-deepmd-kit-v2.0)
 	- [Highlighted features](#highlighted-features)
 	- [License and credits](#license-and-credits)
 	- [Deep Potential in a nutshell](#deep-potential-in-a-nutshell)
- [Download and install](#download-and-install)
- [Use DeePMD-kit](#use-deepmd-kit)
- [Code structure](#code-structure)
- [Troubleshooting](#troubleshooting)

# About DeePMD-kit
DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics (MD). This brings new hopes to addressing the accuracy-versus-efficiency dilemma in molecular simulations. Applications of DeePMD-kit span from finite molecules to extended systems and from metallic systems to chemically bonded systems. 

For more information, check the [documentation](https://deepmd.readthedocs.io/).

# Highlights in DeePMD-kit v2.0
* [Model compression](doc/freeze/compress.md). Accelerate the efficiency of model inference for 4-15 times.
* [New descriptors](doc/model/overall.md). Including [`se_e2_r`](doc/model/train-se-e2-r.md) and [`se_e3`](doc/model/train-se-e3.md).
* [Hybridization of descriptors](doc/model/train-hybrid.md). Hybrid descriptor constructed from concatenation of several descriptors.
* [Atom type embedding](doc/model/train-se-e2-a-tebd.md). Enable atom type embedding to decline training complexity and refine performance.
* Training and inference the dipole (vector) and polarizability (matrix).
* Split of training and validation dataset.
* Optimized training on GPUs. 

## Highlighted features
* **interfaced with TensorFlow**, one of the most popular deep learning frameworks, making the training process highly automatic and efficient, in addition Tensorboard can be used to visualize training procedure.
* **interfaced with high-performance classical MD and quantum (path-integral) MD packages**, i.e., LAMMPS and i-PI, respectively. 
* **implements the Deep Potential series models**, which have been successfully applied to  finite and extended systems including organic molecules, metals, semiconductors, and insulators, etc.
* **implements MPI and GPU supports**, makes it highly efficient for high performance parallel and distributed computing.
* **highly modularized**, easy to adapt to different descriptors for deep learning based potential energy models.

## License and credits
The project DeePMD-kit is licensed under [GNU LGPLv3.0](./LICENSE).
If you use this code in any future publications, please cite this using 
``Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. "DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics." Computer Physics Communications 228 (2018): 178-184.``

## Deep Potential in a nutshell
The goal of Deep Potential is to employ deep learning techniques and realize an inter-atomic potential energy model that is general, accurate, computationally efficient and scalable. The key component is to respect the extensive and symmetry-invariant properties of a potential energy model by assigning a local reference frame and a local environment to each atom. Each environment contains a finite number of atoms, whose local coordinates are arranged in a symmetry preserving way. These local coordinates are then transformed, through a sub-network, to a so-called *atomic energy*. Summing up all the atomic energies gives the potential energy of the system.

The initial proof of concept is in the [Deep Potential][1] paper, which employed an approach that was devised to train the neural network model with the potential energy only. With typical *ab initio* molecular dynamics (AIMD) datasets this is insufficient to reproduce the trajectories. The Deep Potential Molecular Dynamics ([DeePMD][2]) model overcomes this limitation. In addition, the learning process in DeePMD improves significantly over the Deep Potential method thanks to the introduction of a flexible family of loss functions. The NN potential constructed in this way reproduces accurately the AIMD trajectories, both classical and quantum (path integral), in extended and finite systems, at a cost that scales linearly with system size and is always several orders of magnitude lower than that of equivalent AIMD simulations.

Although being highly efficient, the original Deep Potential model satisfies the extensive and symmetry-invariant properties of a potential energy model at the price of introducing discontinuities in the model. This has negligible influence on a trajectory from canonical sampling but might not be sufficient for calculations of dynamical and mechanical properties. These points motivated us to develop the Deep Potential-Smooth Edition ([DeepPot-SE][3]) model, which replaces the non-smooth local frame with a smooth and adaptive embedding network. DeepPot-SE shows great ability in modeling many kinds of systems that are of interests in the fields of physics, chemistry, biology, and materials science.

In addition to building up potential energy models, DeePMD-kit can also be used to build up coarse-grained models. In these models, the quantity that we want to parameterize is the free energy, or the coarse-grained potential, of the coarse-grained particles. See the [DeePCG paper][4] for more details.

# Download and install

Please follow our [GitHub](https://github.com/deepmodeling/deepmd-kit) webpage to download the [latest released version](https://github.com/deepmodeling/deepmd-kit/tree/master) and [development version](https://github.com/deepmodeling/deepmd-kit/tree/devel).

DeePMD-kit offers multiple installation methods. It is recommend using easily methods like [offline packages](doc/install/easy-install.md#offline-packages), [conda](doc/install/easy-install.md#with-conda) and [docker](doc/install/easy-install.md#with-docker). 

One may manually install DeePMD-kit by following the instuctions on [installing the Python interface](doc/install/install-from-source.md#install-the-python-interface) and [installing the C++ interface](doc/install/install-from-source.md#install-the-c-interface). The C++ interface is necessary when using DeePMD-kit with LAMMPS, i-PI or GROMACS.


# Use DeePMD-kit

A quick-start on using DeePMD-kit can be found as follows:

- [Prepare data with dpdata](doc/data/dpdata.md)
- [Training a model](doc/train/training.md)
- [Freeze a model](doc/freeze/freeze.md)
- [Test a model](doc/test/test.md)
- [Run MD with LAMMPS](doc/third-party/lammps.md)

A full [document](doc/train/train-input-auto.rst) on options in the training input script is available.

# Advanced

- [Installation](doc/install/index.md)
    - [Easy install](doc/install/easy-install.md)
    - [Install from source code](doc/install/install-from-source.md)
    - [Install LAMMPS](doc/install/install-lammps.md)
    - [Install i-PI](doc/install/install-ipi.md)
    - [Install GROMACS](doc/install/install-gromacs.md)
    - [Building conda packages](doc/install/build-conda.md)
- [Data](doc/data/index.md)
    - [Data conversion](doc/data/data-conv.md)
    - [Prepare data with dpdata](doc/data/dpdata.md)
- [Model](doc/model/index.md)
    - [Overall](doc/model/overall.md)
    - [Descriptor `"se_e2_a"`](doc/model/train-se-e2-a.md)
    - [Descriptor `"se_e2_r"`](doc/model/train-se-e2-r.md)
    - [Descriptor `"se_e3"`](doc/model/train-se-e3.md)
    - [Descriptor `"hybrid"`](doc/model/train-hybrid.md)
    - [Fit energy](doc/model/train-energy.md)
    - [Fit `tensor` like `Dipole` and `Polarizability`](doc/model/train-fitting-tensor.md)
    - [Train a Deep Potential model using `type embedding` approach](doc/model/train-se-e2-a-tebd.md)
- [Training](doc/train/index.md)
    - [Training a model](doc/train/training.md)
    - [Advanced options](doc/train/training-advanced.md)
    - [Parallel training](doc/train/parallel-training.md)
    - [TensorBoard Usage](doc/train/tensorboard.md)
    - [Known limitations of using GPUs](doc/train/gpu-limitations.md)
    - [Training Parameters](doc/train/train-input-auto.rst)
- [Freeze and Compress](doc/freeze/index.rst)
    - [Freeze a model](doc/freeze/freeze.md)
    - [Compress a model](doc/freeze/compress.md)
- [Test](doc/test/index.rst)
    - [Test a model](doc/test/test.md)
    - [Calculate Model Deviation](doc/test/model-deviation.md)
- [Inference](doc/inference/index.rst)
    - [Python interface](doc/inference/python.md)
    - [C++ interface](doc/inference/cxx.md)
- [Integrate with third-party packages](doc/third-party/index.rst)
    - [Use deep potential with ASE](doc/third-party/ase.md)
    - [Run MD with LAMMPS](doc/third-party/lammps.md)
    - [LAMMPS commands](doc/third-party/lammps-command.md)
    - [Run path-integral MD with i-PI](doc/third-party/ipi.md)
    - [Run MD with GROMACS](doc/third-party/gromacs.md)

# Code structure
The code is organized as follows:

* `data/raw`: tools manipulating the raw data files.

* `examples`: examples.

* `deepmd`: DeePMD-kit python modules.

* `source/api_cc`: source code of DeePMD-kit C++ API.

* `source/ipi`: source code of i-PI client.

* `source/lib`: source code of DeePMD-kit library.

* `source/lmp`: source code of Lammps module.

* `source/gmx`: source code of Gromacs plugin.

* `source/op`: tensorflow op implementation. working with library.


# Troubleshooting

- [Model compatibility](doc/troubleshooting/model-compatability.md)
- [Installation](doc/troubleshooting/installation.md)
- [The temperature undulates violently during early stages of MD](doc/troubleshooting/md-energy-undulation.md)
- [MD: cannot run LAMMPS after installing a new version of DeePMD-kit](doc/troubleshooting/md-version-compatibility.md)
- [Do we need to set rcut < half boxsize?](doc/troubleshooting/howtoset-rcut.md)
- [How to set sel?](doc/troubleshooting/howtoset-sel.md)
- [How to control the number of nodes used by a job?](doc/troubleshooting/howtoset_num_nodes.md)
- [How to tune Fitting/embedding-net size?](doc/troubleshooting/howtoset_netsize.md)


# Contributing

See [DeePMD-kit Contributing Guide](CONTRIBUTING.md) to become a contributor! 🤓


[1]: https://arxiv.org/abs/1707.01478
[2]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[3]: https://arxiv.org/abs/1805.09003
[4]: https://aip.scitation.org/doi/full/10.1063/1.5027645
