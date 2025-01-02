# Interfaces out of DeePMD-kit

The codes of the following interfaces are not a part of the DeePMD-kit package and maintained by other repositories. We list these interfaces here for user convenience.

## Plugins

### External GNN models (MACE/NequIP)

[DeePMD-GNN](https://gitlab.com/RutgersLBSR/deepmd-gnn) is DeePMD-kit plugin for various graph neural network (GNN) models.
It has interfaced with [MACE](https://github.com/ACEsuit/mace) (PyTorch version) and [NequIP](https://github.com/mir-group/nequip) (PyTorch version).
It is also the first example to the DeePMD-kit [plugin mechanism](../development/create-a-model-pt.md#package-new-codes).

## C/C++ interface used by other packages

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
