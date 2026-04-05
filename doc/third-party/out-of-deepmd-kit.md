# Interfaces out of DeePMD-kit

The codes of the following interfaces are not a part of the DeePMD-kit package and maintained by other repositories. We list these interfaces here for user convenience.

## Plugins

### External GNN models (MACE/NequIP)

[DeePMD-GNN](https://gitlab.com/RutgersLBSR/deepmd-gnn) is DeePMD-kit plugin for various graph neural network (GNN) models.
It has interfaced with [MACE](https://github.com/ACEsuit/mace) (PyTorch version) and [NequIP](https://github.com/mir-group/nequip) (PyTorch version).
It is also the first example to the DeePMD-kit [plugin mechanism](../development/create-a-model-pt.md#package-new-codes).

## C/C++ interface used by other packages

### Third-party GROMACS interface to DeePMD-kit

A third-party GROMACS integration based on the GROMACS Neural Network Potentials (NNPot) infrastructure is reported in [Enabling AI Deep Potentials for Ab Initio-quality Molecular Dynamics Simulations in GROMACS](https://arxiv.org/abs/2602.02234) and maintained outside the DeePMD-kit repository at [HuXioAn/gromacs/tree/deepmd-oneModel](https://github.com/HuXioAn/gromacs/tree/deepmd-oneModel).

According to the paper, this implementation

- couples GROMACS NNPot to the DeePMD-kit C++/CUDA inference backend;
- enables inference for multiple DeePMD model families, including `se_e2_a`, `DPA`, `DPA2`, and `DPA3`;
- relies on DeePMD-kit's multi-backend support, with the paper demonstrating runs with the PyTorch backend and discussing TensorFlow and JAX support through DeePMD-kit;
- supports using DeePMD-kit for selected atom groups inside GROMACS NNPot workflows, including hybrid classical/DP simulations;
- is demonstrated on protein-in-water systems (1YRF, 1UBQ, 3LZM, and 2PTC) on NVIDIA A100 and GH200 GPUs.

The paper shows a workflow where DeePMD-kit provides the short-range interactions for a selected atom group, while the remaining interactions continue to be handled by standard GROMACS force-field terms. In the reported protein-in-water examples, DeePMD-kit is used for the protein internal interactions, while water and protein-water interactions remain classical.

The same study also notes current scope limitations:

- the benchmarks enable DeePMD inference only in the production MD stage, not in EM/NVT/NPT;
- the reported implementation uses single-rank inference inside the current GROMACS NNPot design;
- scalability and domain-decomposed inference remain optimization targets;
- some DPA3 benchmark cases run out of GPU memory on the tested hardware.

This interface is maintained outside DeePMD-kit. Please consult the corresponding third-party repository and paper for build instructions, supported GROMACS versions, and runtime details.

### OpenMM plugin for DeePMD-kit

An [OpenMM](https://github.com/openmm/openmm) plugin is provided from [JingHuangLab/openmm_deepmd_plugin](https://github.com/JingHuangLab/openmm_deepmd_plugin), written by the [Huang Lab](http://www.compbiophysics.org/) at Westlake University.

### Amber interface to DeePMD-kit

Starting from [AmberTools24](https://ambermd.org/), `sander` includes an interface to the DeePMD-kit, which implements the [Deep Potential Range Corrected (DPRc) correction](../model/dprc.md).
The DPRc model and the interface were developed by the [York Lab](https://theory.rutgers.edu/) from Rutgers University.
More details are available in

- [Amber Reference Manuals](https://ambermd.org/Manuals.php), providing documentation for how to enable the interface and the `&dprc` namelist;
- [GitLab RutgersLBSR/AmberDPRc](https://gitlab.com/RutgersLBSR/AmberDPRc/), providing examples mdin files;
- [DP-Amber](https://github.com/njzjz/dpamber/), a tiny tool to convert Amber trajectory to DPRc training data;
- [The original DPRc paper](https://doi.org/10.1021/acs.jctc.1c00201).

### CP2K interface to DeePMD-kit

[CP2K](https://github.com/cp2k/cp2k/) v2024.2 adds an interface to the DeePMD-kit for molecular dynamics. Read the [CP2K manual](https://manual.cp2k.org/trunk/methods/machine_learning/deepmd.html#deepmd-kit) for details.

### ABACUS

[ABACUS](https://github.com/deepmodeling/abacus-develop/) can run molecular dynamics with a DP model. User is required to [build ABACUS with DeePMD-kit](https://abacus.deepmodeling.com/en/latest/advanced/install.html#build-with-deepmd-kit).

## Command line interface used by other packages

### DP-GEN

[DP-GEN](https://github.com/deepmodeling/dpgen) provides a workflow to generate accurate DP models by calling DeePMD-kit's command line interface (CLI) in the local or remote server. Details can be found in [this paper](https://doi.org/10.1016/j.cpc.2020.107206).

### MLatom

[Mlatom](http://mlatom.com/) provides an interface to the DeePMD-kit within MLatom's workflow by calling DeePMD-kit's CLI. Details can be found in [this paper](https://doi.org/10.1007/s41061-021-00339-5).
