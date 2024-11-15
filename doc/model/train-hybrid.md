# Descriptor `"hybrid"` {{ tensorflow_icon }} {{ pytorch_icon }} {{ jax_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}, PyTorch {{ pytorch_icon }}, JAX {{ jax_icon }}, DP {{ dpmodel_icon }}
:::

This descriptor hybridizes multiple descriptors to form a new descriptor. For example, we have a list of descriptors denoted by $\mathcal D_1$, $\mathcal D_2$, ..., $\mathcal D_N$, the hybrid descriptor this the concatenation of the list, i.e. $\mathcal D = (\mathcal D_1, \mathcal D_2, \cdots, \mathcal D_N)$.

## Theory

A hybrid descriptor $\mathcal{D}^i_\text{hyb}$ concatenates multiple kinds of descriptors into one descriptor:

```math
    \mathcal{D}^{i}_\text{hyb} = \{
    \begin{array}{cccc}
        \mathcal{D}^{i}_1 & \mathcal{D}^{i}_2 & \cdots & \mathcal{D}^{i}_n
    \end{array}
    \}.
```

The list of descriptors can be different types or the same descriptors with different parameters.
This way, one can set the different cutoff radii for different descriptors.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen, Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Instructions

To use the descriptor in DeePMD-kit, one firstly set the {ref}`type <model[standard]/descriptor/type>` to {ref}`hybrid <model[standard]/descriptor[hybrid]>`, then provide the definitions of the descriptors by the items in the `list`,

```json
        "descriptor" :{
            "type": "hybrid",
            "list" : [
                {
		    "type" : "se_e2_a",
		    ...
                },
                {
		    "type" : "se_e2_r",
		    ...
                }
            ]
        },
```

A complete training input script of this example can be found in the directory

```bash
$deepmd_source_dir/examples/water/hybrid/input.json
```

## Type embedding

Type embedding is different between the TensorFlow backend and other backends.
In the TensorFlow backend, all descriptors share the same descriptor that defined in the model level.
In other backends, each descriptor has its own type embedding and their parameters may be different.

## Model compression

Model compression is supported if all sub-descriptors support model compression.
