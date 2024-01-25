# Multi-task training {{ tensorflow_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}
:::

## Theory

The multi-task training process can simultaneously handle different datasets with properties that cannot be fitted in one network (e.g. properties from DFT calculations under different exchange-correlation functionals or different basis sets).
These datasets are denoted by $\boldsymbol x^{(1)}, \dots, \boldsymbol x^{(n_t)}$.
For each dataset, a training task is defined as
```math
    \min_{\boldsymbol \theta}   L^{(t)} (\boldsymbol x^{(t)}; \boldsymbol  \theta^{(t)}, \tau), \quad t=1, \dots, n_t.
```

During the multi-task training process, all tasks share one descriptor with trainable parameters $\boldsymbol{\theta}_ {d}$, while each of them has its own fitting network with trainable parameters $\boldsymbol{\theta}_ f^{(t)}$, thus
$\boldsymbol{\theta}^{(t)} = \{ \boldsymbol{\theta}_ {d} , \boldsymbol{\theta}_ {f}^{(t)} \}$.
At each training step, a task is randomly picked from ${1, \dots, n_t}$, and the Adam optimizer is executed to minimize $L^{(t)}$ for one step to update the parameter $\boldsymbol \theta^{(t)}$.
If different fitting networks have the same architecture, they can share the parameters of some layers
to improve training efficiency.[^1]

[^1]: This section is built upon Jinzhe Zeng, Duo Zhang, Denghui Lu, Pinghui Mo, Zeyu Li, Yixiao Chen,  Marián Rynik, Li'ang Huang, Ziyao Li, Shaochen Shi, Yingze Wang, Haotian Ye, Ping Tuo, Jiabin Yang, Ye Ding, Yifan Li, Davide Tisi, Qiyu Zeng, Han Bao, Yu Xia, Jiameng Huang, Koki Muraoka, Yibo Wang, Junhan Chang, Fengbo Yuan, Sigbjørn Løland Bore, Chun Cai, Yinnian Lin, Bo Wang, Jiayan Xu, Jia-Xin Zhu, Chenxing Luo, Yuzhi Zhang, Rhys E. A. Goodall, Wenshuo Liang, Anurag Kumar Singh, Sikai Yao, Jingchao Zhang, Renata Wentzcovitch, Jiequn Han, Jie Liu, Weile Jia, Darrin M. York, Weinan E, Roberto Car, Linfeng Zhang, Han Wang, [J. Chem. Phys. 159, 054801 (2023)](https://doi.org/10.1063/5.0155600) licensed under a [Creative Commons Attribution (CC BY) license](http://creativecommons.org/licenses/by/4.0/).

## Perform the multi-task training
Training on multiple data sets (each data set contains several data systems) can be performed in multi-task mode,
with one common descriptor and multiple specific fitting nets for each data set.
One can simply switch the following parameters in training input script to perform multi-task mode:
- {ref}`fitting_net <model/fitting_net>` --> {ref}`fitting_net_dict <model/fitting_net_dict>`,
each key of which can be one individual fitting net.
- {ref}`training_data <training/training_data>`,  {ref}`validation_data <training/validation_data>`
--> {ref}`data_dict <training/data_dict>`, each key of which can be one individual data set contains
several data systems for corresponding fitting net, the keys must be consistent with those in
{ref}`fitting_net_dict <model/fitting_net_dict>`.
- {ref}`loss <loss>` --> {ref}`loss_dict <loss_dict>`, each key of which can be one individual loss setting
for corresponding fitting net, the keys must be consistent with those in
{ref}`fitting_net_dict <model/fitting_net_dict>`, if not set, the corresponding fitting net will use the default loss.
- (Optional) {ref}`fitting_weight <training/fitting_weight>`, each key of which can be a non-negative integer or float,
deciding the chosen probability for corresponding fitting net in training, if not set or invalid,
the corresponding fitting net will not be used.

The training procedure will automatically choose single-task or multi-task mode, based on the above parameters.
Note that parameters of single-task mode and multi-task mode can not be mixed.

An example input for training energy and dipole in water system can be found here: [multi-task input on water](../../examples/water_multi_task/ener_dipole/input.json).

The supported descriptors for multi-task mode are listed:
- {ref}`se_a (se_e2_a) <model/descriptor[se_e2_a]>`
- {ref}`se_r (se_e2_r) <model/descriptor[se_e2_r]>`
- {ref}`se_at (se_e3) <model/descriptor[se_e3]>`
- {ref}`se_atten <model/descriptor[se_atten]>`
- {ref}`se_atten_v2 <model/descriptor[se_atten_v2]>`
- {ref}`hybrid <model/descriptor[hybrid]>`

The supported fitting nets for multi-task mode are listed:
- {ref}`ener <model/fitting_net[ener]>`
- {ref}`dipole <model/fitting_net[dipole]>`
- {ref}`polar <model/fitting_net[polar]>`

The output of `dp freeze` command in multi-task mode can be seen in [freeze command](../freeze/freeze.md).

## Initialization from pretrained multi-task model
For advance training in multi-task mode, one can first train the descriptor on several upstream datasets and then transfer it on new downstream ones with newly added fitting nets.
At the second step, you can also inherit some fitting nets trained on upstream datasets, by merely adding fitting net keys in {ref}`fitting_net_dict <model/fitting_net_dict>` and
optional fitting net weights in {ref}`fitting_weight <training/fitting_weight>`.

Take [multi-task input on water](../../examples/water_multi_task/ener_dipole/input.json) again for example.
You can first train a multi-task model using input script with the following {ref}`model <model>` part:
```json
    "model": {
        "type_map": ["O", "H"],
        "descriptor": {
            "type":     "se_e2_a",
            "sel":      [46, 92],
            "rcut_smth":    0.5,
            "rcut":     6.0,
            "neuron":       [25, 50, 100],
        },
        "fitting_net_dict": {
            "water_dipole": {
                "type":         "dipole",
                "neuron":       [100, 100, 100],
            },
            "water_ener": {
                "neuron":       [240, 240, 240],
                "resnet_dt":    true,
            }
        },
    }
```
After training, you can freeze this multi-task model into one unit graph:
```bash
$ dp freeze -o graph.pb --united-model
```
Then if you want to transfer the trained descriptor and some fitting nets (take `water_ener` for example) to newly added datasets with new fitting net `water_ener_2`,
you can modify the {ref}`model <model>` part of the new input script in a more simplified way:
```json
    "model": {
        "type_map": ["O", "H"],
        "descriptor": {},
        "fitting_net_dict": {
            "water_ener": {},
            "water_ener_2": {
                "neuron":       [240, 240, 240],
                "resnet_dt":    true,
            }
        },
    }
```
It will autocomplete the configurations according to the frozen graph.

Note that for newly added fitting net keys, other parts in the input script, including {ref}`data_dict <training/data_dict>` and {ref}`loss_dict <loss_dict>` (optionally {ref}`fitting_weight <training/fitting_weight>`),
should be set explicitly. While for old fitting net keys, it will inherit the old configurations if not set.

Finally, you can perform the modified multi-task training from the frozen model with command:
```bash
$ dp train input.json --init_frz_model graph.pb
```

## Share layers among energy fitting networks

The multi-task training can be used to train multiple levels of energies (e.g. DFT and CCSD(T)) at the same time.
In this situation, one can set {ref}`model/fitting_net[ener]/layer_name>` to share some of layers among fitting networks.
The architecture of the layers with the same name should be the same.

For example, if one want to share the first and the third layers for two three-hidden-layer fitting networks, the following parameters should be set.
```json
"fitting_net_dict": {
    "ccsd": {
        "neuron": [
            240,
            240,
            240
        ],
        "layer_name": ["l0", null, "l2", null]
    },
    "wb97m": {
        "neuron": [
            240,
            240,
            240
        ],
        "layer_name": ["l0", null, "l2", null]
    }
}
```
