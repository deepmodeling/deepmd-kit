# Train a Deep Potential model using descriptor `"se_e2_a_tebd"`

The notation of `se_e2_a_tebd` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from all information (both angular and radial) of atomic configurations. The `e2` stands for the embedding with two-atoms information, and `tebd` stands for applying type embedding for each atom type. 

A complete training input script of this example can be find in the directory. 
```bash
$deepmd_source_dir/examples/water/se_e2_a_tebd/input.json
```
With atom type embedding, we can share one descriptor embedding net and one fitting net in total, which decline training complexity largely. 

The training input script is similar to that of [`se_e2_a`](train-se-e2-a.md#the-training-input-script). By adding the `type_embedding` section, model will automatically apply `se_e2_a_tebd` approach and generate type embedding vectors. If type embedding is detected, descriptor and fitting net would take it as a part of input.

### Type embedding net
The `model` defines how the model is constructed, adding a section of type embedding net:
```json=
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

The construction of type embedding net is given by `type_embedding`. An example of type_embedding is provided as follows
```json=
	"type_embedding":{
	    "neuron":		[2, 4, 8],
	    "resnet_dt":	false,
	    "seed":		1
	}
```
* The `neuron` specifies the size of the type embedding net. From left to right the members denote the sizes of each hidden layer from input end to the output end, respectively. It takes one-hot vector as input and output dimension equals to the last dimension of the `neuron` list. If the outer layer is of twice size as the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
* If the option `resnet_dt` is set `true`, then a timestep is used in the ResNet.
* `seed` gives the random seed that is used to generate random numbers when initializing the model parameters.
