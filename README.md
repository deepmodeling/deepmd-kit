<span style="font-size:larger;">DeePMD-kit Manual</span>
========
[![GitHub release](https://img.shields.io/github/release/deepmodeling/deepmd-kit.svg?maxAge=86400)](https://github.com/deepmodeling/deepmd-kit/releases)
[![Documentation Status](https://readthedocs.org/projects/deepmd/badge/?version=latest)](https://deepmd.readthedocs.io/en/latest/?badge=latest)

# Table of contents
- [About DeePMD-kit](#about-deepmd-kit)
 	- [Highlighted features](#highlighted-features)
 	- [Code structure](#code-structure)
 	- [License and credits](#license-and-credits)
 	- [Deep Potential in a nutshell](#deep-potential-in-a-nutshell)
- [Download and install](#download-and-install)
- [Use DeePMD-kit](#use-deepmd-kit)
- [Troubleshooting](#troubleshooting)

# About DeePMD-kit
DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning based model of interatomic potential energy and force field and to perform molecular dynamics (MD). This brings new hopes to addressing the accuracy-versus-efficiency dilemma in molecular simulations. Applications of DeePMD-kit span from finite molecules to extended systems and from metallic systems to chemically bonded systems. 

For more information, check the [documentation](https://deepmd.readthedocs.io/).

## Highlighted features
* **interfaced with TensorFlow**, one of the most popular deep learning frameworks, making the training process highly automatic and efficient, in addition Tensorboard can be used to visualize training procedure.
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

* `source/op`: tensorflow op implementation. working with library.

* `source/train`: Python modules and scripts for training and testing.


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

Please follow our [github](https://github.com/deepmodeling/deepmd-kit) webpage to download the [latest released version](https://github.com/deepmodeling/deepmd-kit/tree/master) and [development version](https://github.com/deepmodeling/deepmd-kit/tree/devel).

DeePMD-kit offers multiple installation methods. It is recommend using easily methods like [offline packages](doc/install.md#offline-packages), [conda](doc/install.md#with-conda) and [docker](doc/install.md#with-docker). 

One may manually install DeePMD-kit by following the instuctions on [installing the python interface](doc/install.md#install-the-python-interface) and [installing the C++ interface](doc/install.md#install-the-c-interface). The C++ interface is necessary when using DeePMD-kit with LAMMPS and i-PI.


# Use DeePMD-kit

The typical procedure of using DeePMD-kit includes 5 steps 

1. [Prepare data](doc/use-deepmd-kit.md#prepare-data)
2. [Train a model](doc/use-deepmd-kit.md#train-a-model)
3. [Analyze training with Tensorboard](doc/tensorboard.md)
4. [Freeze the model](doc/use-deepmd-kit.md#freeze-a-model)
5. [Test the model](doc/use-deepmd-kit.md#test-a-model)
6. [Compress the model](doc/use-deepmd-kit.md#compress-a-model)
7. [Inference the model in python](doc/use-deepmd-kit.md#model-inference) or using the model in other molecular simulation packages like [LAMMPS](doc/use-deepmd-kit.md#run-md-with-lammps), [i-PI](doc/use-deepmd-kit.md#run-path-integral-md-with-i-pi) or [ASE](doc/use-deepmd-kit.md#use-deep-potential-with-ase).

A quick-start on using DeePMD-kit can be found [here](doc/use-deepmd-kit.md).

A full [document](doc/train-input.rst) on options in the training input script is available.


# Troubleshooting
In consequence of various differences of computers or systems, problems may occur. Some common circumstances are listed as follows. 
If other unexpected problems occur, you're welcome to contact us for help.

## Model compatability

When the version of DeePMD-kit used to training model is different from the that of DeePMD-kit running MDs, one has the problem of model compatability.

DeePMD-kit guarantees that the codes with the same major and minor revisions are compatible. That is to say v0.12.5 is compatible to v0.12.0, but is not compatible to v0.11.0 nor v1.0.0. 

## Installation: inadequate versions of gcc/g++
Sometimes you may use a gcc/g++ of version <4.9. If you have a gcc/g++ of version > 4.9, say, 7.2.0, you may choose to use it by doing 
```bash
export CC=/path/to/gcc-7.2.0/bin/gcc
export CXX=/path/to/gcc-7.2.0/bin/g++
```

If, for any reason, for example, you only have a gcc/g++ of version 4.8.5, you can still compile all the parts of TensorFlow and most of the parts of DeePMD-kit. i-Pi will be disabled automatically.

## Installation: build files left in DeePMD-kit
When you try to build a second time when installing DeePMD-kit, files produced before may contribute to failure. Thus, you may clear them by
```bash
cd build
rm -r *
```
and redo the `cmake` process.

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
