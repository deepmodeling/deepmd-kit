# Deep Potential - Range Correction (DPRc) {{ tensorflow_icon }} {{ pytorch_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, DP {{ dpmodel_icon }}
:::

Deep Potential - Range Correction (DPRc) is designed to combine with QM/MM method, and corrects energies from a low-level QM/MM method to a high-level QM/MM method:

```math
E=E_\text{QM}(\mathbf R; \mathbf P)  + E_\text{QM/MM}(\mathbf R; \mathbf P) + E_\text{MM}(\mathbf R) + E_\text{DPRc}(\mathbf R)
```

## Theory

Deep Potential - Range Correction (DPRc) was initially designed to correct the potential energy from a fast, linear-scaling low-level semiempirical QM/MM theory to a high-level ''ab initio'' QM/MM theory in a range-correction way to quantitatively correct short and mid-range non-bonded interactions leveraging the non-bonded lists routinely used in molecular dynamics simulations using molecular mechanical force fields such as AMBER.
In this way, long-ranged electrostatic interactions can be modeled efficiently using the particle mesh Ewald method or its extensions for multipolar and QM/MM potentials.
In a DPRc model, the switch function is modified to disable MM-MM interaction:

```math
  s_\text{DPRc}(r_{ij}) =
  \begin{cases}
  0, &\text{if $i \in \text{MM} \land j \in \text{MM}$}, \\
  s(r_{ij}), &\text{otherwise},
  \end{cases}
```

where $s_\text{DPRc}(r_{ij})$ is the new switch function and $s(r_{ij})$ is the old one.
This ensures the forces between MM atoms are zero, i.e.

```math
{\boldsymbol F}_{ij} = - \frac{\partial E}{\partial \boldsymbol r_{ij}} = 0, \quad i \in \text{MM} \land j \in \text{MM}.
```

The fitting network is revised to remove energy bias from MM atoms:

```math
  E_i=
  \begin{cases}
  \mathcal{F}_0(\mathcal{D}^i),  &\text{if $i \in \text{QM}$}, \\
  \mathcal{F}_0(\mathcal{D}^i) - \mathcal{F}_0(\mathbf{0}), &\text{if $i \in \text{MM}$},
  \end{cases}
```

where $\mathbf{0}$ is a zero matrix.
It is worth mentioning that usage of DPRc is not limited to its initial design for QM/MM correction and can be expanded to any similar interaction.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

See the [JCTC paper](https://doi.org/10.1021/acs.jctc.1c00201) for details.

## Training data

Instead the normal _ab initio_ data, one needs to provide the correction from a low-level QM/MM method to a high-level QM/MM method:

```math
E = E_\text{high-level QM/MM} - E_\text{low-level QM/MM}
```

Two levels of data use the same MM method, so $E_\text{MM}$ is eliminated.

## Training the DPRc model

In a DPRc model, QM atoms and MM atoms have different atom types. Assuming we have 4 QM atom types (C, H, O, P) and 2 MM atom types (HW, OW):

```json
"type_map": ["C", "H", "HW", "O", "OW", "P"]
```

As described in the paper, the DPRc model only corrects $E_\text{QM}$ and $E_\text{QM/MM}$ within the cutoff, so we use a hybrid descriptor to describe them separately:

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```json
"descriptor" :{
    "type":             "hybrid",
    "list" : [
        {
            "type":     "se_a_ebd_v2",
            "sel":              [6, 11, 0, 6, 0, 1],
            "rcut_smth":        1.00,
            "rcut":             9.00,
            "neuron":           [12, 25, 50],
            "exclude_types":    [[2, 2], [2, 4], [4, 4], [0, 2], [0, 4], [1, 2], [1, 4], [3, 2], [3, 4], [5, 2], [5, 4]],
            "axis_neuron":      12,
            "_comment": " QM/QM interaction"
        },
        {
            "type":     "se_a_ebd_v2",
            "sel":              [6, 11, 100, 6, 50, 1],
            "rcut_smth":        0.50,
            "rcut":             6.00,
            "neuron":           [12, 25, 50],
            "exclude_types":    [[0, 0], [0, 1], [0, 3], [0, 5], [1, 1], [1, 3], [1, 5], [3, 3], [3, 5], [5, 5], [2, 2], [2, 4], [4, 4]],
            "axis_neuron":      12,
            "set_davg_zero":    true,
            "_comment": " QM/MM interaction"
        }
    ]
}
```

:::

:::{tab-item} PyTorch {{ pytorch_icon }}

```json
"descriptor" :{
    "type":             "hybrid",
    "list" : [
        {
            "type":     "se_e2_a",
            "sel":              [6, 11, 0, 6, 0, 1],
            "rcut_smth":        1.00,
            "rcut":             9.00,
            "neuron":           [12, 25, 50],
            "exclude_types":    [[2, 2], [2, 4], [4, 4], [0, 2], [0, 4], [1, 2], [1, 4], [3, 2], [3, 4], [5, 2], [5, 4]],
            "axis_neuron":      12,
            "type_one_side":    true,
            "_comment": " QM/QM interaction"
        },
        {
            "type":     "se_e2_a",
            "sel":              [6, 11, 100, 6, 50, 1],
            "rcut_smth":        0.50,
            "rcut":             6.00,
            "neuron":           [12, 25, 50],
            "exclude_types":    [[0, 0], [0, 1], [0, 3], [0, 5], [1, 1], [1, 3], [1, 5], [3, 3], [3, 5], [5, 5], [2, 2], [2, 4], [4, 4]],
            "axis_neuron":      12,
            "set_davg_zero":    true,
            "type_one_side":    true,
            "_comment": " QM/MM interaction"
        }
    ]
}
```

:::

::::

{ref}`exclude_types <model[standard]/descriptor[se_a_ebd_v2]/exclude_types>` can be generated by the following Python script:

```py
from itertools import combinations_with_replacement, product

qm = (0, 1, 3, 5)
mm = (2, 4)
print(
    "QM/QM:",
    list(map(list, list(combinations_with_replacement(mm, 2)) + list(product(qm, mm)))),
)
print(
    "QM/MM:",
    list(
        map(
            list,
            list(combinations_with_replacement(qm, 2))
            + list(combinations_with_replacement(mm, 2)),
        )
    ),
)
```

Also, DPRc assumes MM atom energies ({ref}`atom_ener <model[standard]/fitting_net[ener]/atom_ener>`) are zero:

```json
"fitting_net": {
   "neuron": [240, 240, 240],
   "resnet_dt": true,
   "atom_ener": [null, null, 0.0, null, 0.0, null]
}
```

Note that {ref}`atom_ener <model[standard]/fitting_net[ener]/atom_ener>` only works when {ref}`descriptor/set_davg_zero <model[standard]/descriptor[se_a_ebd_v2]/set_davg_zero>` of the QM/MM part is `true`.

## Run MD simulations

The DPRc model has the best practices with the [AMBER](../third-party/out-of-deepmd-kit.md#amber-interface-to-deepmd-kit) QM/MM module. An example is given by [GitLab RutgersLBSR/AmberDPRc](https://gitlab.com/RutgersLBSR/AmberDPRc/). In theory, DPRc is able to be used with any QM/MM package, as long as the DeePMD-kit package accepts QM atoms and MM atoms within the cutoff range and returns energies and forces.

## Pairwise DPRc

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}
:::

If one wants to correct from a low-level method into a full DFT level, and the system is too large to do full DFT calculation, one may try the experimental pairwise DPRc model.
In a pairwise DPRc model, the total energy is divided into QM internal energy and the sum of QM/MM energy for each MM residue $l$:

$$ E = E*\text{QM} + \sum*{l} E\_{\text{QM/MM},l} $$

In this way, the interaction between the QM region and each MM fragmentation can be computed and trained separately.
Thus, the pairwise DPRc model is divided into two sub-[DPRc models](./dprc.md).
`qm_model` is for the QM internal interaction and `qmmm_model` is for the QM/MM interaction.
The configuration for these two models is similar to the non-pairwise DPRc model.
It is noted that the [`se_atten` descriptor](./train-se-atten.md) should be used, as it is the only descriptor to support the mixed type.

```json
{
  "model": {
    "type": "pairwise_dprc",
    "type_map": ["C", "P", "O", "H", "OW", "HW"],
    "type_embedding": {
      "neuron": [8],
      "precision": "float32"
    },
    "qm_model": {
      "descriptor": {
        "type": "se_atten_v2",
        "sel": 24,
        "rcut_smth": 0.5,
        "rcut": 9.0,
        "attn_layer": 0,
        "neuron": [25, 50, 100],
        "resnet_dt": false,
        "axis_neuron": 12,
        "precision": "float32",
        "seed": 1
      },
      "fitting_net": {
        "type": "ener",
        "neuron": [240, 240, 240],
        "resnet_dt": true,
        "precision": "float32",
        "atom_ener": [null, null, null, null, 0.0, 0.0],
        "seed": 1
      }
    },
    "qmmm_model": {
      "descriptor": {
        "type": "se_atten_v2",
        "sel": 27,
        "rcut_smth": 0.5,
        "rcut": 6.0,
        "attn_layer": 0,
        "neuron": [25, 50, 100],
        "resnet_dt": false,
        "axis_neuron": 12,
        "set_davg_zero": true,
        "exclude_types": [
          [0, 0],
          [0, 1],
          [0, 2],
          [0, 3],
          [1, 1],
          [1, 2],
          [1, 3],
          [2, 2],
          [2, 3],
          [3, 3],
          [4, 4],
          [4, 5],
          [5, 5]
        ],
        "precision": "float32",
        "seed": 1
      },
      "fitting_net": {
        "type": "ener",
        "neuron": [240, 240, 240],
        "resnet_dt": true,
        "seed": 1,
        "precision": "float32",
        "atom_ener": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      }
    }
  }
}
```

The pairwise model needs information for MM residues.
The model uses [`aparam`](../data/system.md) with the shape of `nframes x natoms` to get the residue index.
The QM residue should always use `0` as the index.
For example, `0 0 0 1 1 1 2 2 2` means these 9 atoms are grouped into one QM residue and two MM residues.
