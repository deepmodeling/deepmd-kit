# Interpolation or combination with a pairwise potential {{ tensorflow_icon }} {{ pytorch_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, DP {{ dpmodel_icon }}
:::

## Theory

In applications like the radiation damage simulation, the interatomic distance may become too close, so that the DFT calculations fail.
In such cases, the DP model that is an approximation of the DFT potential energy surface is usually replaced by an empirical potential, like the Ziegler-Biersack-Littmark (ZBL) screened nuclear repulsion potential in the radiation damage simulations.
The DeePMD-kit package supports the interpolation between DP and an empirical pairwise potential

```math
  E_i = (1-w_i) E_i^{\mathrm{DP}} + w_i (E_i^0 + E_i^{\mathrm{pair}}),
```

where the $w_i$ is the interpolation weight and the $E_i^{\mathrm{pair}}  $ is the atomic contribution due to the pairwise potential $u^{\mathrm{pair}}(r)$, i.e.

```math
  E_i^{\mathrm{pair}} = \sum_{j\in n(i)} u^{\mathrm{pair}}(r_{ij}).
```

The interpolation weight $w_i$ is defined by

```math
    w_i =
    \begin{cases}
    1, & \sigma_i \lt r_a, \\
    u_i^3 (-6 u_i^2 +15 u_i -10) +1, & r_a \leq \sigma_i \lt r_b, \\
    0, & \sigma_i \geq r_b,
    \end{cases}
```

where $u_i = (\sigma_i - r_a ) / (r_b - r_a)$.
$E_i^0$ is the atom energy bias.
In the range $[r_a, r_b]$, the DP model smoothly switched off and the pairwise potential smoothly switched on from $r_b$ to $r_a$. The $\sigma_i$ is the softmin of the distance between atom $i$ and its neighbors,

```math
  \sigma_i =
  \dfrac
  {\sum\limits_{j\in n(i)} r_{ij} e^{-r_{ij} / \alpha_s}}
  {\sum\limits_{j\in n(i)} e^{-r_{ij} / \alpha_s}},
```

where the scale $\alpha_s$ is a tunable scale of the interatomic distance $r_{ij}$.
The pairwise potential $u^{\textrm{pair}}(r)$ is defined by a user-defined table that provides the value of $u^{\textrm{pair}}$ on an evenly discretized grid from 0 to the cutoff distance.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

DeePMD-kit also supports combination with a pairwise potential {{ tensorflow_icon }}:

```math
  E_i = E_i^{\mathrm{DP}} + E_i^{\mathrm{pair}},
```

## Table file

The table file should be a text file that can be read by {py:meth}`numpy.loadtxt`.
The first column is the distance between two atoms, where upper range should be larger than the cutoff radius.
Other columns are two-body interaction energies for pairs of certain types,
in the order of Type_0-Type_0, Type_0-Type_1, ..., Type_0-Type_N, Type_1-Type_1, ..., Type_1-Type_N, ..., and Type_N-Type_N.

The interaction should be smooth at the cut-off distance.

:::{note}
In instances where the interaction at the cut-off distance is not delineated within the table file, extrapolation will be conducted utilizing the available interaction data. This extrapolative procedure guarantees a smooth transition from the table-provided value to `0` whenever feasible.
:::

## Interpolation with a short-range pairwise potential

```json
"model": {
  "use_srtab": "H2O_tab_potential.txt",
  "smin_alpha": 0.1,
  "sw_rmin": 0.8,
  "sw_rmax": 1.0,
  "_comment": "Below uses a normal DP model"
}
```

{ref}`sw_rmin <model/sw_rmin>` and {ref}`sw_rmax <model/sw_rmax>` must be smaller than the cutoff radius of the DP model.

## Combination with a pairwise potential {{ tensorflow_icon }}

To combine with a pairwise potential, use the [linear model](./linear.md):

```json
"model": {
  "type": "linear_ener",
  "weights": "sum",
  "models": [
    {
      "_comment": "Here uses a normal DP model"
    },
    {
      "type": "pairtab",
      "tab_file": "dftd3.txt",
      "rcut": 10.0,
      "sel": 534
    }
  ]
}
```

The {ref}`rcut <model[pairtab]/rcut>` can be larger than that of the DP model.
