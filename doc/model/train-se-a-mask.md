# Descriptor `"se_a_mask"` {{ tensorflow_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}
:::

Descriptor `se_a_mask` is a concise implementation of the descriptor `se_e2_a`,
but functions slightly differently.
`se_a_mask` is specially designed for DP/MM simulations where the number of atoms in DP regions is dynamically changed in simulations.

Therefore, the descriptor `se_a_mask` is not supported for training with PBC systems for simplicity.
Besides, to make the output shape of the descriptor matrix consistent,
the input coordinates are padded with virtual particle coordinates to the maximum number of atoms (specified with `sel` in the descriptor setting) in the system.
The real/virtual sign of the atoms is specified with the `aparam.npy` ( [ nframes * natoms ] ) file in the input systems set directory.
The `aparam.npy` can also be seen as the mask of the atoms in the system,
which is also the origin of the name `se_a_mask`.

In this example, we will train a DP Mask model for zinc protein interactions.
The input systems are the collection of zinc and its coordinates residues.
A sample input system that contains 2 frames is included in the directory.

```bash
$deepmd_source_dir/examples/zinc_protein/data_dp_mask
```

A complete training input script of this example can be found in the directory.

```bash
$deepmd_source_dir/examples/zinc_protein/zinc_se_a_mask.json
```

The construction of the descriptor is given by section {ref}`descriptor <model[standard]/descriptor>`. An example of the descriptor is provided as follows

```json
	"descriptor" :{
	    "type":	"se_a_mask",
	    "sel":		[36, 16, 24, 64, 6, 1],
	    "neuron":		[25, 50, 100],
		"axis_neuron": 16,
	    "type_one_side":	false,
	    "resnet_dt":	false,
	    "seed":		1
	}
```

- The {ref}`type <model[standard]/descriptor/type>` of the descriptor is set to `"se_a_mask"`.
- {ref}`sel <model[standard]/descriptor[se_a_mask]/sel>` gives the maximum number of atoms in input coordinates. It is a list, the length of which is the same as the number of atom types in the system, and `sel[i]` denotes the maximum number of atoms with type `i`.
- The {ref}`neuron <model[standard]/descriptor[se_a_mask]/neuron>` specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from the input end to the output end, respectively. If the outer layer is twice the size of the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
- The {ref}`axis_neuron <model[standard]/descriptor[se_a_mask]/axis_neuron>` specifies the size of the submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper](https://arxiv.org/abs/1805.09003)
- If the option {ref}`type_one_side <model[standard]/descriptor[se_a_mask]/type_one_side>` is set to `true`, the embedding network parameters vary by types of neighbor atoms only, so there will be $N_\text{types}$ sets of embedding network parameters. Otherwise, the embedding network parameters vary by types of centric atoms and types of neighbor atoms, so there will be $N_\text{types}^2$ sets of embedding network parameters.
- If the option {ref}`resnet_dt <model[standard]/descriptor[se_a_mask]/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
- {ref}`seed <model[standard]/descriptor[se_a_mask]/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.

To make the `aparam.npy` used for descriptor `se_a_mask`, two variables in `fitting_net` section are needed.

```json
	"fitting_net" :{
	    "neuron": [240, 240, 240],
      	"resnet_dt": true,
      	"seed": 1,
      	"numb_aparam": 1,
      	"use_aparam_as_mask": true
	}
```

- `neuron`, `resnet_dt` and `seed` are the same as the {ref}`fitting_net <model[standard]/fitting_net[ener]>` section for fitting energy.
- {ref}`numb_aparam <model[standard]/fitting_net[ener]/numb_aparam>` gives the dimension of the `aparam.npy` file. In this example, it is set to 1 and stores the real/virtual sign of the atoms. For real/virtual atoms, the corresponding sign in `aparam.npy` is set to 1/0.
- {ref}`use_aparam_as_mask <model[standard]/fitting_net[ener]/use_aparam_as_mask>` is set to `true` to use the `aparam.npy` as the mask of the atoms in the descriptor `se_a_mask`.

Finally, to make a reasonable fitting task with `se_a_mask` descriptor for DP/MM simulations, the loss function with `se_a_mask` is designed to include the atomic forces difference in specific atoms of the input particles only.
More details about the selection of the specific atoms can be found in paper [DP/MM](left to be filled).
Thus, `atom_pref.npy` ( [ nframes * natoms ] ) is required as the indicator of the specific atoms in the input particles.
And the `loss` section in the training input script should be set as follows.

```json
"loss": {
    "type": "ener",
    "start_pref_e": 0.0,
    "limit_pref_e": 0.0,
    "start_pref_f": 0.0,
    "limit_pref_f": 0.0,
    "start_pref_pf": 1.0,
    "limit_pref_pf": 1.0,
    "_comment": " that's all"
  }
```

## Type embedding

Same as [`se_e2_a`](./train-se-e2-a.md).

## Model compression

Same as [`se_e2_a`](./train-se-e2-a.md).
