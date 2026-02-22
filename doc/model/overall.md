# Overall

## Theory

A Deep Potential (DP) model, denoted by $\mathcal{M}$, can be generally represented as

```math
\boldsymbol y_i = \mathcal M (\boldsymbol x_i, \{\boldsymbol x_j\}_{j\in n(i)}; \boldsymbol \theta)
= \mathcal{F} \big( \mathcal{D} (\boldsymbol x_i, \{\boldsymbol x_j\}_{j\in n(i)}; \boldsymbol \theta_d) ; \boldsymbol \theta_f \big),
```

where $\boldsymbol{y}_i$ is the fitting properties, $\mathcal{F}$ is the fitting network, $\mathcal{D}$ is the descriptor.
$\boldsymbol{x} = (\boldsymbol r_i, \alpha_i)$, with $\boldsymbol r_i$ being the Cartesian coordinates and $\alpha_i$ being the chemical species, denotes the degrees of freedom of the atom $i$.

The indices of the neighboring atoms (i.e. atoms within a certain cutoff radius) of atom $i$ are given by the notation $n(i)$.
Note that the Cartesian coordinates can be either under the periodic boundary condition (PBC) or in vacuum (under the open boundary condition).
The network parameters are denoted by $\boldsymbol \theta = \{\boldsymbol \theta_d, \boldsymbol \theta_f\}$, where $\boldsymbol \theta_d$ and $\boldsymbol\theta_f$ yield the network parameters of the descriptor (if any) and those of the fitting network, respectively.
From the above equation, one may compute the global property of the system by

```math
    \boldsymbol y = \sum_{i=1}^N \boldsymbol y_i,
```

where $N$ is the number of atoms in a frame.
For example, if $y_i$ represents the potential energy contribution of atom $i$, then $y$ gives the total potential energy of the frame.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions

A model has two parts, a descriptor that maps atomic configuration to a set of symmetry invariant features, and a fitting net that takes descriptor as input and predicts the atomic contribution to the target physical property. It's defined in the {ref}`model <model>` section of the `input.json`, for example,

```json
    "model": {
        "type_map":	["O", "H"],
        "descriptor" :{
            "...": "..."
        },
        "fitting_net" : {
            "...": "..."
        }
    }
```

The two subsections, {ref}`descriptor <model[standard]/descriptor>` and {ref}`fitting_net <model[standard]/fitting_net>`, define the descriptor and the fitting net, respectively.

The {ref}`type_map <model/type_map>` is optional, which provides the element names (but not necessarily same as the actual name of the element) of the corresponding atom types. A water model, as in this example, has two kinds of atoms. The atom types are internally recorded as integers, e.g., `0` for oxygen and `1` for hydrogen here. A mapping from the atom type to their names is provided by {ref}`type_map <model/type_map>`.

DeePMD-kit implements the following descriptors:

1. [`se_e2_a`](train-se-e2-a.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes the distance between atoms as input.
2. [`se_e2_r`](train-se-e2-r.md): DeepPot-SE constructed from radial information of atomic configurations. The embedding takes the distance between atoms as input.
3. [`se_e3`](train-se-e3.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The embedding takes angles between two neighboring atoms as input.
4. [`se_a_mask`](train-se-a-mask.md): DeepPot-SE constructed from all information (both angular and radial) of atomic configurations. The input frames in one system can have a varied number of atoms. Input particles are padded with virtual particles of the same length.
5. `loc_frame`: Defines a local frame at each atom and compute the descriptor as local coordinates under this frame.
6. [`hybrid`](train-hybrid.md): Concate a list of descriptors to form a new descriptor.

The fitting of the following physical properties is supported

1. [`ener`](train-energy.md): Fit the energy of the system. The force (derivative with atom positions), the virial (derivative with the box tensor) and the hessian (second-order derivative with atom positions) can also be trained.

:::{warning}
Due to the restrictions of torch jit script, the models trained with hessian are not jitable so that the frozen models cannot output hessians.
:::

2. [`dipole`](train-fitting-tensor.md): The dipole moment.
3. [`polar`](train-fitting-tensor.md): The polarizability.
