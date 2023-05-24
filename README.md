[<picture><source media="(prefers-color-scheme: dark)" srcset="./doc/_static/logo-dark.svg"><source media="(prefers-color-scheme: light)" srcset="./doc/_static/logo.svg"><img alt="DeePMD-kit logo" src="./doc/_static/logo.svg"></picture>](./doc/logo.md)

--------------------------------------------------------------------------------

<span style="font-size:larger;">DeePMD-kit Manual</span>
========
[![GitHub release](https://img.shields.io/github/release/deepmodeling/deepmd-kit.svg?maxAge=86400)](https://github.com/deepmodeling/deepmd-kit/releases)
[![doi:10.1016/j.cpc.2018.03.016](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2018.03.016-blue)](https://doi.org/10.1016/j.cpc.2020.107206)
[![Citations](https://citations.njzjz.win/10.1016/j.cpc.2018.03.016)](https://badge.dimensions.ai/details/doi/10.1016/j.cpc.2018.03.016)
[![offline packages](https://img.shields.io/github/downloads/deepmodeling/deepmd-kit/total?label=offline%20packages)](https://github.com/deepmodeling/deepmd-kit/releases)
[![conda-forge](https://img.shields.io/conda/dn/conda-forge/deepmd-kit?color=red&label=conda-forge&logo=conda-forge)](https://anaconda.org/conda-forge/deepmd-kit)
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
DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning-based model of interatomic potential energy and force field and to perform molecular dynamics (MD). This brings new hopes to addressing the accuracy-versus-efficiency dilemma in molecular simulations. Applications of DeePMD-kit span from finite molecules to extended systems and from metallic systems to chemically bonded systems.

For more information, check the [documentation](https://deepmd.readthedocs.io/).

# Highlights in DeePMD-kit v2.0
* [Model compression](doc/freeze/compress.md). Accelerate the efficiency of model inference 4-15 times.
* [New descriptors](doc/model/overall.md). Including [`se_e2_r`](doc/model/train-se-e2-r.md) and [`se_e3`](doc/model/train-se-e3.md).
* [Hybridization of descriptors](doc/model/train-hybrid.md). Hybrid descriptor constructed from the concatenation of several descriptors.
* [Atom type embedding](doc/model/train-se-e2-a-tebd.md). Enable atom-type embedding to decline training complexity and refine performance.
* Training and inference of the dipole (vector) and polarizability (matrix).
* Split of training and validation dataset.
* Optimized training on GPUs.

## Highlighted features
* **interfaced with TensorFlow**, one of the most popular deep learning frameworks, making the training process highly automatic and efficient, in addition, Tensorboard can be used to visualize training procedures.
* **interfaced with high-performance classical MD and quantum (path-integral) MD packages**, i.e., LAMMPS and i-PI, respectively.
* **implements the Deep Potential series models**, which have been successfully applied to finite and extended systems including organic molecules, metals, semiconductors, insulators, etc.
* **implements MPI and GPU supports**, making it highly efficient for high-performance parallel and distributed computing.
* **highly modularized**, easy to adapt to different descriptors for deep learning-based potential energy models.

## License and credits
The project DeePMD-kit is licensed under [GNU LGPLv3.0](./LICENSE).
If you use this code in any future publications, please cite the following publications for general purpose:
- Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. "DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics." Computer Physics Communications 228 (2018): 178-184.
- Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang. "DeePMD-kit v2: A software package for Deep Potential models." [arXiv:2304.09409](https://doi.org/10.48550/arXiv.2304.09409).

In addition, please follow [the bib file](CITATIONS.bib) to cite the methods you used.

## Deep Potential in a nutshell
The goal of Deep Potential is to employ deep learning techniques and realize an inter-atomic potential energy model that is general, accurate, computationally efficient and scalable. The key component is to respect the extensive and symmetry-invariant properties of a potential energy model by assigning a local reference frame and a local environment to each atom. Each environment contains a finite number of atoms, whose local coordinates are arranged in a symmetry-preserving way. These local coordinates are then transformed, through a sub-network, to so-called *atomic energy*. Summing up all the atomic energies gives the potential energy of the system.

The initial proof of concept is in the [Deep Potential][1] paper, which employed an approach that was devised to train the neural network model with the potential energy only. With typical *ab initio* molecular dynamics (AIMD) datasets this is insufficient to reproduce the trajectories. The Deep Potential Molecular Dynamics ([DeePMD][2]) model overcomes this limitation. In addition, the learning process in DeePMD improves significantly over the Deep Potential method thanks to the introduction of a flexible family of loss functions. The NN potential constructed in this way reproduces accurately the AIMD trajectories, both classical and quantum (path integral), in extended and finite systems, at a cost that scales linearly with system size and is always several orders of magnitude lower than that of equivalent AIMD simulations.

Although highly efficient, the original Deep Potential model satisfies the extensive and symmetry-invariant properties of a potential energy model at the price of introducing discontinuities in the model. This has negligible influence on a trajectory from canonical sampling but might not be sufficient for calculations of dynamical and mechanical properties. These points motivated us to develop the Deep Potential-Smooth Edition ([DeepPot-SE][3]) model, which replaces the non-smooth local frame with a smooth and adaptive embedding network. DeepPot-SE shows great ability in modeling many kinds of systems that are of interest in the fields of physics, chemistry, biology, and materials science.

In addition to building up potential energy models, DeePMD-kit can also be used to build up coarse-grained models. In these models, the quantity that we want to parameterize is the free energy, or the coarse-grained potential, of the coarse-grained particles. See the [DeePCG paper][4] for more details.

See [our latest paper](https://doi.org/10.48550/arXiv.2304.09409) for details of all features.

# Download and install

Please follow our [GitHub](https://github.com/deepmodeling/deepmd-kit) webpage to download the [latest released version](https://github.com/deepmodeling/deepmd-kit/tree/master) and [development version](https://github.com/deepmodeling/deepmd-kit/tree/devel).

DeePMD-kit offers multiple installation methods. It is recommended to use easy methods like [offline packages](doc/install/easy-install.md#offline-packages), [conda](doc/install/easy-install.md#with-conda) and [docker](doc/install/easy-install.md#with-docker).

One may manually install DeePMD-kit by following the instructions on [installing the Python interface](doc/install/install-from-source.md#install-the-python-interface) and [installing the C++ interface](doc/install/install-from-source.md#install-the-c-interface). The C++ interface is necessary when using DeePMD-kit with LAMMPS, i-PI or GROMACS.


# Use DeePMD-kit

A quick start on using DeePMD-kit can be found [here](doc/getting-started/quick_start.ipynb).

A full [document](doc/train/train-input-auto.rst) on options in the training input script is available.

# Advanced

- [Installation](doc/install/index.md)
    - [Easy install](doc/install/easy-install.md)
    - [Install from source code](doc/install/install-from-source.md)
    - [Install from pre-compiled C library](doc/install/install-from-c-library.md)
    - [Install LAMMPS](doc/install/install-lammps.md)
    - [Install i-PI](doc/install/install-ipi.md)
    - [Install GROMACS](doc/install/install-gromacs.md)
    - [Building conda packages](doc/install/build-conda.md)
    - [Install Node.js interface](doc/install/install-nodejs.md)
- [Data](doc/data/index.md)
    - [System](doc/data/system.md)
    - [Formats of a system](doc/data/data-conv.md)
    - [Prepare data with dpdata](doc/data/dpdata.md)
- [Model](doc/model/index.md)
    - [Overall](doc/model/overall.md)
    - [Descriptor `"se_e2_a"`](doc/model/train-se-e2-a.md)
    - [Descriptor `"se_e2_r"`](doc/model/train-se-e2-r.md)
    - [Descriptor `"se_e3"`](doc/model/train-se-e3.md)
    - [Descriptor `"se_atten"`](doc/model/train-se-atten.md)
    - [Descriptor `"hybrid"`](doc/model/train-hybrid.md)
    - [Descriptor `sel`](doc/model/sel.md)
    - [Fit energy](doc/model/train-energy.md)
    - [Fit spin energy](doc/model/train-energy-spin.md)
    - [Fit `tensor` like `Dipole` and `Polarizability`](doc/model/train-fitting-tensor.md)
- [Fit electronic density of states (DOS)](doc/model/train-fitting-dos.md)
    - [Train a Deep Potential model using `type embedding` approach](doc/model/train-se-e2-a-tebd.md)
    - [Deep potential long-range](doc/model/dplr.md)
    - [Deep Potential - Range Correction (DPRc)](doc/model/dprc.md)
- [Training](doc/train/index.md)
    - [Training a model](doc/train/training.md)
    - [Advanced options](doc/train/training-advanced.md)
    - [Parallel training](doc/train/parallel-training.md)
    - [Multi-task training](doc/train/multi-task-training.md)
    - [TensorBoard Usage](doc/train/tensorboard.md)
    - [Known limitations of using GPUs](doc/train/gpu-limitations.md)
    - [Training Parameters](doc/train-input-auto.rst)
- [Freeze and Compress](doc/freeze/index.rst)
    - [Freeze a model](doc/freeze/freeze.md)
    - [Compress a model](doc/freeze/compress.md)
- [Test](doc/test/index.rst)
    - [Test a model](doc/test/test.md)
    - [Calculate Model Deviation](doc/test/model-deviation.md)
- [Inference](doc/inference/index.rst)
    - [Python interface](doc/inference/python.md)
    - [C++ interface](doc/inference/cxx.md)
    - [Node.js interface](doc/inference/nodejs.md)
- [Integrate with third-party packages](doc/third-party/index.rst)
    - [Use deep potential with ASE](doc/third-party/ase.md)
    - [Run MD with LAMMPS](doc/third-party/lammps.md)
    - [LAMMPS commands](doc/third-party/lammps-command.md)
    - [Run path-integral MD with i-PI](doc/third-party/ipi.md)
    - [Run MD with GROMACS](doc/third-party/gromacs.md)
    - [Interfaces out of DeePMD-kit](doc/third-party/out-of-deepmd-kit.md)
- [Use NVNMD](doc/nvnmd/index.md)

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
* `source/op`: TensorFlow op implementation. working with the library.


# Troubleshooting

- [Model compatibility](doc/troubleshooting/model_compatability.md)
- [Installation](doc/troubleshooting/installation.md)
- [The temperature undulates violently during the early stages of MD](doc/troubleshooting/md_energy_undulation.md)
- [MD: cannot run LAMMPS after installing a new version of DeePMD-kit](doc/troubleshooting/md_version_compatibility.md)
- [Do we need to set rcut < half boxsize?](doc/troubleshooting/howtoset_rcut.md)
- [How to set sel?](doc/troubleshooting/howtoset_sel.md)
- [How to control the parallelism of a job?](doc/troubleshooting/howtoset_num_nodes.md)
- [How to tune Fitting/embedding-net size?](doc/troubleshooting/howtoset_netsize.md)
- [Why does a model have low precision?](doc/troubleshooting/precision.md)

# Contributing

See [DeePMD-kit Contributing Guide](CONTRIBUTING.md) to become a contributor! 🤓


[1]: https://arxiv.org/abs/1707.01478
[2]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[3]: https://arxiv.org/abs/1805.09003
[4]: https://aip.scitation.org/doi/full/10.1063/1.5027645
