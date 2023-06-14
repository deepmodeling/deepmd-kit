# Descriptor `"se_e2_a"`

The notation of `se_e2_a` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from all information (both angular and radial) of atomic configurations. The `e2` stands for the embedding with two-atoms information. This descriptor was described in detail in [the DeepPot-SE paper](https://arxiv.org/abs/1805.09003).

Note that it is sometimes called a "two-atom embedding descriptor" which means the input of the embedding net is atomic distances. The descriptor **does** encode multi-body information (both angular and radial information of neighboring atoms).

In this example, we will train a DeepPot-SE model for a water system.  A complete training input script of this example can be found in the directory.
```bash
$deepmd_source_dir/examples/water/se_e2_a/input.json
```
With the training input script, data are also provided in the example directory. One may train the model with the DeePMD-kit from the directory.

The construction of the descriptor is given by section {ref}`descriptor <model/descriptor>`. An example of the descriptor is provided as follows
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
* The {ref}`type <model/descriptor/type>` of the descriptor is set to `"se_e2_a"`.
* {ref}`rcut <model/descriptor[se_e2_a]/rcut>` is the cut-off radius for neighbor searching, and the {ref}`rcut_smth <model/descriptor[se_e2_a]/rcut_smth>` gives where the smoothing starts.
* {ref}`sel <model/descriptor[se_e2_a]/sel>` gives the maximum possible number of neighbors in the cut-off radius. It is a list, the length of which is the same as the number of atom types in the system, and `sel[i]` denotes the maximum possible number of neighbors with type `i`.
* The {ref}`neuron <model/descriptor[se_e2_a]/neuron>` specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from the input end to the output end, respectively. If the outer layer is twice the size of the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
* If the option {ref}`type_one_side <model/descriptor[se_e2_a]/type_one_side>` is set to `true`, the embedding network parameters vary by types of neighbor atoms only, so there will be $N_\text{types}$ sets of embedding network parameters. Otherwise, the embedding network parameters vary by types of centric atoms and types of neighbor atoms, so there will be $N_\text{types}^2$ sets of embedding network parameters.
* The {ref}`axis_neuron <model/descriptor[se_e2_a]/axis_neuron>` specifies the size of the submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper](https://arxiv.org/abs/1805.09003)
* If the option {ref}`resnet_dt <model/descriptor[se_e2_a]/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
* {ref}`seed <model/descriptor[se_e2_a]/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.
