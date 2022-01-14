# Descriptor `"se_e2_a"`

The notation of `se_e2_a` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from all information (both angular and radial) of atomic configurations. The `e2` stands for the embedding with two-atoms information. This descriptor was described in detail in [the DeepPot-SE paper](https://arxiv.org/abs/1805.09003).

In this example we will train a DeepPot-SE model for a water system.  A complete training input script of this example can be find in the directory. 
```bash
$deepmd_source_dir/examples/water/se_e2_a/input.json
```
With the training input script, data are also provided in the example directory. One may train the model with the DeePMD-kit from the directory.

The construction of the descriptor is given by section `descriptor`. An example of the descriptor is provided as follows
```json
	"descriptor" :{
	    "type":		"se_e2_a",
	    "rcut_smth":	0.50,
	    "rcut":		6.00,
	    "sel":		[46, 92],
	    "neuron":		[25, 50, 100],
	    "type_one_side":	true,
	    "axis_neuron":	16,
	    "resnet_dt":	false,
	    "seed":		1
	}
```
* The `type` of the descriptor is set to `"se_e2_a"`. 
* `rcut` is the cut-off radius for neighbor searching, and the `rcut_smth` gives where the smoothing starts. 
* `sel` gives the maximum possible number of neighbors in the cut-off radius. It is a list, the length of which is the same as the number of atom types in the system, and `sel[i]` denote the maximum possible number of neighbors with type `i`. 
* The `neuron` specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from input end to the output end, respectively. If the outer layer is of twice size as the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
* If the option `type_one_side` is set to `true`, then descriptor will consider the types of neighbor atoms. Otherwise, both the types of centric and  neighbor atoms are considered.
* The `axis_neuron` specifies the size of submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper](https://arxiv.org/abs/1805.09003) 
* If the option `resnet_dt` is set `true`, then a timestep is used in the ResNet.
* `seed` gives the random seed that is used to generate random numbers when initializing the model parameters.

