[<picture><source media="(prefers-color-scheme: dark)" srcset="./doc/_static/logo-dark.svg"><source media="(prefers-color-scheme: light)" srcset="./doc/_static/logo.svg"><img alt="DeePMD-kit logo" src="./doc/_static/logo.svg"></picture>](./doc/logo.md)

--------------------------------------------------------------------------------

# DeePMD-kit

[![GitHub release](https://img.shields.io/github/release/deepmodeling/deepmd-kit.svg?maxAge=86400)](https://github.com/deepmodeling/deepmd-kit/releases)
[![offline packages](https://img.shields.io/github/downloads/deepmodeling/deepmd-kit/total?label=offline%20packages)](https://github.com/deepmodeling/deepmd-kit/releases)
[![conda-forge](https://img.shields.io/conda/dn/conda-forge/deepmd-kit?color=red&label=conda-forge&logo=conda-forge)](https://anaconda.org/conda-forge/deepmd-kit)
[![pip install](https://img.shields.io/pypi/dm/deepmd-kit?label=pip%20install)](https://pypi.org/project/deepmd-kit)
[![docker pull](https://img.shields.io/docker/pulls/deepmodeling/deepmd-kit)](https://hub.docker.com/r/deepmodeling/deepmd-kit)
[![Documentation Status](https://readthedocs.org/projects/deepmd/badge/)](https://deepmd.readthedocs.io/)

## About DeePMD-kit
DeePMD-kit is a package written in Python/C++, designed to minimize the effort required to build deep learning-based model of interatomic potential energy and force field and to perform molecular dynamics (MD). This brings new hopes to addressing the accuracy-versus-efficiency dilemma in molecular simulations. Applications of DeePMD-kit span from finite molecules to extended systems and from metallic systems to chemically bonded systems.

For more information, check the [documentation](https://deepmd.readthedocs.io/).

### Highlighted features
* **interfaced with multiple backends**, including TensorFlow and PyTorch, the most popular deep learning frameworks, making the training process highly automatic and efficient.
* **interfaced with high-performance classical MD and quantum (path-integral) MD packages**, including LAMMPS, i-PI, AMBER, CP2K, GROMACS, OpenMM, and ABUCUS.
* **implements the Deep Potential series models**, which have been successfully applied to finite and extended systems, including organic molecules, metals, semiconductors, insulators, etc.
* **implements MPI and GPU supports**, making it highly efficient for high-performance parallel and distributed computing.
* **highly modularized**, easy to adapt to different descriptors for deep learning-based potential energy models.

### License and credits
The project DeePMD-kit is licensed under [GNU LGPLv3.0](./LICENSE).
If you use this code in any future publications, please cite the following publications for general purpose:
- Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. "DeePMD-kit: A deep learning package for many-body potential energy representation and molecular dynamics." Computer Physics Communications 228 (2018): 178-184.
[![doi:10.1016/j.cpc.2018.03.016](https://img.shields.io/badge/DOI-10.1016%2Fj.cpc.2018.03.016-blue)](https://doi.org/10.1016/j.cpc.2018.03.016)
[![Citations](https://citations.njzjz.win/10.1016/j.cpc.2018.03.016)](https://badge.dimensions.ai/details/doi/10.1016/j.cpc.2018.03.016)
- Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang. "DeePMD-kit v2: A software package for deep potential models." J. Chem. Phys. 159 (2023): 054801.
[![doi:10.1063/5.0155600](https://img.shields.io/badge/DOI-10.1063%2F5.0155600-blue)](https://doi.org/10.1063/5.0155600)
[![Citations](https://citations.njzjz.win/10.1063/5.0155600)](https://badge.dimensions.ai/details/doi/10.1063/5.0155600)

In addition, please follow [the bib file](CITATIONS.bib) to cite the methods you used.

### Highlights in major versions

#### Initial version
The goal of Deep Potential is to employ deep learning techniques and realize an inter-atomic potential energy model that is general, accurate, computationally efficient and scalable. The key component is to respect the extensive and symmetry-invariant properties of a potential energy model by assigning a local reference frame and a local environment to each atom. Each environment contains a finite number of atoms, whose local coordinates are arranged in a symmetry-preserving way. These local coordinates are then transformed, through a sub-network, to so-called *atomic energy*. Summing up all the atomic energies gives the potential energy of the system.

The initial proof of concept is in the [Deep Potential][1] paper, which employed an approach that was devised to train the neural network model with the potential energy only. With typical *ab initio* molecular dynamics (AIMD) datasets this is insufficient to reproduce the trajectories. The Deep Potential Molecular Dynamics ([DeePMD][2]) model overcomes this limitation. In addition, the learning process in DeePMD improves significantly over the Deep Potential method thanks to the introduction of a flexible family of loss functions. The NN potential constructed in this way reproduces accurately the AIMD trajectories, both classical and quantum (path integral), in extended and finite systems, at a cost that scales linearly with system size and is always several orders of magnitude lower than that of equivalent AIMD simulations.

Although highly efficient, the original Deep Potential model satisfies the extensive and symmetry-invariant properties of a potential energy model at the price of introducing discontinuities in the model. This has negligible influence on a trajectory from canonical sampling but might not be sufficient for calculations of dynamical and mechanical properties. These points motivated us to develop the Deep Potential-Smooth Edition ([DeepPot-SE][3]) model, which replaces the non-smooth local frame with a smooth and adaptive embedding network. DeepPot-SE shows great ability in modeling many kinds of systems that are of interest in the fields of physics, chemistry, biology, and materials science.

In addition to building up potential energy models, DeePMD-kit can also be used to build up coarse-grained models. In these models, the quantity that we want to parameterize is the free energy, or the coarse-grained potential, of the coarse-grained particles. See the [DeePCG paper][4] for more details.

#### v1

* Code refactor to make it highly modularized.
* GPU support for descriptors.

#### v2

* Model compression. Accelerate the efficiency of model inference 4-15 times.
* New descriptors. Including `se_e2_r`, `se_e3`, and `se_atten` (DPA-1).
* Hybridization of descriptors. Hybrid descriptor constructed from the concatenation of several descriptors.
* Atom type embedding. Enable atom-type embedding to decline training complexity and refine performance.
* Training and inference of the dipole (vector) and polarizability (matrix).
* Split of training and validation dataset.
* Optimized training on GPUs, including CUDA and ROCm.
* Non-von-Neumann.
* C API to interface with the third-party packages.

See [our latest paper](https://doi.org/10.1063/5.0155600) for details of all features until v2.2.3.

#### v3

* Multiple backends supported. Add a PyTorch backend.
* The DPA-2 model.

## Install and use DeePMD-kit

Please read the [online documentation](https://deepmd.readthedocs.io/) for how to install and use DeePMD-kit.

## Code structure

The code is organized as follows:

* `examples`: examples.
* `deepmd`: DeePMD-kit python modules.
* `source/lib`: source code of the core library.
* `source/op`: Operator (OP) implementation.
* `source/api_cc`: source code of DeePMD-kit C++ API.
* `source/api_c`: source code of the C API.
* `source/nodejs`: source code of the Node.js API.
* `source/ipi`: source code of i-PI client.
* `source/lmp`: source code of Lammps module.
* `source/gmx`: source code of Gromacs plugin.

# Contributing

See [DeePMD-kit Contributing Guide](CONTRIBUTING.md) to become a contributor! 🤓

[1]: https://arxiv.org/abs/1707.01478
[2]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.143001
[3]: https://arxiv.org/abs/1805.09003
[4]: https://aip.scitation.org/doi/full/10.1063/1.5027645
