# Type embedding approach

We generate specific a type embedding vector for each atom type so that we can share one descriptor embedding net and one fitting net in total, which decline training complexity largely.

The training input script is similar to that of [`se_e2_a`](train-se-e2-a.md), but different by adding the {ref}`type_embedding <model/type_embedding>` section.

## Type embedding net
The {ref}`model <model>` defines how the model is constructed, adding a section of type embedding net:
```json
    "model": {
	"type_map":	["O", "H"],
	"type_embedding":{
			...
	},
	"descriptor" :{
            ...
	},
	"fitting_net" : {
            ...
	}
    }
```
The model will automatically apply the type embedding approach and generate type embedding vectors. If the type embedding vector is detected, the descriptor and fitting net would take it as a part of the input.

The construction of type embedding net is given by {ref}`type_embedding <model/type_embedding>`. An example of {ref}`type_embedding <model/type_embedding>` is provided as follows
```json
	"type_embedding":{
	    "neuron":		[2, 4, 8],
	    "resnet_dt":	false,
	    "seed":		1
	}
```
* The {ref}`neuron <model/type_embedding/neuron>` specifies the size of the type embedding net. From left to right the members denote the sizes of each hidden layer from the input end to the output end, respectively. It takes a one-hot vector as input and output dimension equals to the last dimension of the {ref}`neuron <model/type_embedding/neuron>` list. If the outer layer is twice the size of the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
* If the option {ref}`resnet_dt <model/type_embedding/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
* {ref}`seed <model/type_embedding/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.


A complete training input script of this example can be found in the directory.
```bash
$deepmd_source_dir/examples/water/se_e2_a_tebd/input.json
```
See [here](../development/type-embedding.md) for further explanation of `type embedding`.

:::{note}
You can't apply the compression method while using the atom type embedding.
:::
