# Interfaces out of DeePMD-kit

The codes of the following interfaces are not a part of the DeePMD-kit package and maintained by other repositories. We list these interfaces here for user convenience.

## dpdata

[dpdata](https://github.com/deepmodeling/dpdata) provides the `predict` method for `System` class:

```py
import dpdata
dsys = dpdata.LabeledSystem('OUTCAR')
dp_sys = dsys.predict("frozen_model_compressed.pb")
```

By inferring with the DP model `frozen_model_compressed.pb`, dpdata will generate a new labeled system `dp_sys` with inferred energies, forces, and virials.

## OpenMM plugin for DeePMD-kit

An [OpenMM](https://github.com/openmm/openmm) plugin is provided from [JingHuangLab/openmm_deepmd_plugin](https://github.com/JingHuangLab/openmm_deepmd_plugin), written by the [Huang Lab](http://www.compbiophysics.org/) at Westlake University.

## AMBER interface to DeePMD-kit

An [AMBER](https://ambermd.org/) interface to DeePMD-kit is written by the [York [Lab](https://theory.rutgers.edu/) from Rutgers University. It is open-source at [GitLab RutgersLBSR/AmberDPRc](https://gitlab.com/RutgersLBSR/AmberDPRc/). Details can be found in [this paper](https://doi.org/10.1021/acs.jctc.1c00201).

## DP-GEN

[DP-GEN](https://github.com/deepmodeling/dpgen) provides a workflow to generate accurate DP models by calling DeePMD-kit's command line interface (CLI) in the local or remote server. Details can be found in [this paper](https://doi.org/10.1016/j.cpc.2020.107206).

## MLatom

[Mlatom](http://mlatom.com/) provides an interface to the DeePMD-kit within MLatom's workflow by calling DeePMD-kit's CLI. Details can be found in [this paper](https://doi.org/10.1007/s41061-021-00339-5).

## ABACUS

[ABACUS](https://github.com/deepmodeling/abacus-develop/) can run molecular dynamics with a DP model. User is required to [build ABACUS with DeePMD-kit](https://abacus.deepmodeling.com/en/latest/advanced/install.html#build-with-deepmd-kit).
