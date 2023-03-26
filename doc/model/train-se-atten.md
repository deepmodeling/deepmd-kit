# Descriptor `"se_atten"`

## DPA-1: Pretraining of Attention-based Deep Potential Model for Molecular Simulation

![ALT](../images/model_se_atten.png "model_se_atten")

Here we propose DPA-1, a Deep Potential model with a novel attention mechanism, which is highly effective for representing the conformation and chemical spaces of atomic systems and learning the PES.

See [this paper](https://arxiv.org/abs/2208.08236) for more information. DPA-1 is implemented as a new descriptor `"se_atten"` for model training, which can be used after simply editing the input.json.

## Installation
Follow the [standard installation](../install/install-from-source.md#install-the-python-interface) of Python interface in the DeePMD-kit.
After that, you can smoothly use the DPA-1 model with the following instructions.

## Introduction to new features of DPA-1
Next, we will list the detailed settings in input.json and the data format, especially for large systems with dozens of elements. An example of DPA-1 input can be found [here](../../examples/water/se_atten/input.json).

### Descriptor `"se_atten"`

The notation of `se_atten` is short for the smooth edition of Deep Potential with an attention mechanism.
This descriptor was described in detail in [the DPA-1 paper](https://arxiv.org/abs/2208.08236) and the images above.

In this example, we will train a DPA-1 model for a water system.  A complete training input script of this example can be found in the directory:
```bash
$deepmd_source_dir/examples/water/se_atten/input.json
```
With the training input script, data are also provided in the example directory. One may train the model with the DeePMD-kit from the directory.

An example of the DPA-1 descriptor is provided as follows
```json
	"descriptor" :{
          "type":		"se_atten",
          "rcut_smth":	0.50,
          "rcut":		6.00,
          "sel":		120,
          "neuron":		[25, 50, 100],
          "axis_neuron":	16,
          "resnet_dt":	false,
          "attn":	128,
          "attn_layer":	2,
          "attn_mask":	false,
          "attn_dotr":	true,
          "seed":	1
	}
```
* The {ref}`type <model/descriptor/type>` of the descriptor is set to `"se_atten"`, which will use DPA-1 structures.
* {ref}`rcut <model/descriptor[se_atten]/rcut>` is the cut-off radius for neighbor searching, and the {ref}`rcut_smth <model/descriptor[se_atten]/rcut_smth>` gives where the smoothing starts.
* **{ref}`sel <model/descriptor[se_atten]/sel>`** gives the maximum possible number of neighbors in the cut-off radius. It is an int. Note that this number highly affects the efficiency of training, which we usually use less than 200. (We use 120 for training 56 elements in [OC2M dataset](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md))
* The {ref}`neuron <model/descriptor[se_atten]/neuron>` specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from the input end to the output end, respectively. If the outer layer is twice the size of the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
* The {ref}`axis_neuron <model/descriptor[se_atten]/axis_neuron>` specifies the size of the submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper](https://arxiv.org/abs/1805.09003)
* If the option {ref}`resnet_dt <model/descriptor[se_atten]/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
* {ref}`seed <model/descriptor[se_atten]/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.
* {ref}`attn <model/descriptor[se_atten]/attn>` sets the length of a hidden vector during scale-dot attention computation.
* {ref}`attn_layer <model/descriptor[se_atten]/attn_layer>` sets the number of layers in attention mechanism.
* {ref}`attn_mask <model/descriptor[se_atten]/attn_mask>` determines whether to mask the diagonal in the attention weights and False is recommended.
* {ref}`attn_dotr <model/descriptor[se_atten]/attn_dotr>` determines whether to dot the relative coordinates on the attention weights as a gated scheme, True is recommended.

### Fitting `"ener"`
DPA-1 only supports `"ener"` fitting type, and you can refer [here](train-energy.md) for detailed information.

### Type embedding
DPA-1 only supports models with type embeddings. And the default setting is as follows:
```json
"type_embedding":{
            "neuron":           [8],
            "resnet_dt":        false,
            "seed":             1
        }
```
You can add these settings in input.json if you want to change the default ones, see [here](train-se-e2-a-tebd.md) for detailed information.


### Type map
For training large systems, especially those with dozens of elements, the {ref}`type <model/type_map>` determines the element index of training data:
```json
"type_map": [
   "Mg",
   "Al",
   "Cu"
  ]
```
which should include all the elements in the dataset you want to train on.
## Data format
DPA-1 supports the standard data format, which is detailed in [data-conv.md](../data/data-conv.md) and [system.md](../data/system.md).
Note that in this format, only those frames with the same fingerprint (i.e. the number of atoms of different elements) can be put together as a unified system.
This may lead to sparse frame numbers in those rare systems.

An ideal way is to put systems with the same total number of atoms together, which is the way we trained DPA-1 on [OC2M](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md).
This system format, which is called `mixed_type`, is proper to put frame-sparse systems together and is slightly different from the standard one.
Take an example, a `mixed_type` may contain the following files:
```
type.raw
type_map.raw
set.*/box.npy
set.*/coord.npy
set.*/energy.npy
set.*/force.npy
set.*/real_atom_types.npy
```
This system contains `Nframes` frames with the same atom number `Natoms`, the total number of element types contained in all frames is `Ntypes`. Most files are the same as those in [standard formats](../data/system.md), here we only list the distinct ones:

ID             | Property                         | File                | Required/Optional    | Shape                    | Description
----------     | -------------------------------- | ------------------- | -------------------- | -----------------------  | -----------
/              | Atom type indexes (place holder) | type.raw            | Required             | Natoms                   | All zeros to fake the type input
type_map       | Atom type names                  | type_map.raw        | Required             | Ntypes                   | Atom names that map to atom type contained in all the frames, which is unnecessart to be contained in the periodic table
type           | Atom type indexes of each frame  | real_atom_types.npy | Required             | Nframes \* Natoms        | Integers that describe atom types in each frame, corresponding to indexes in type_map. `-1` means virtual atoms.

With these edited files, one can put together frames with the same `Natoms`, instead of the same formula (like `H2O`). Note that this `mixed_type` format only supports `se_atten` descriptor.

To put frames with different `Natoms` into the same system, one can pad systems by adding virtual atoms whose type is `-1`. Virtual atoms do not contribute to any fitting property, so the atomic property of virtual atoms (e.g. forces) should be given zero.

The API to generate or transfer to `mixed_type` format is available on [dpdata](https://github.com/deepmodeling/dpdata) for a more convenient experience.

## Training example
Here we upload the AlMgCu example shown in the paper, you can download it here:
[Baidu disk](https://pan.baidu.com/s/1Mk9CihPHCmf8quwaMhT-nA?pwd=d586);
[Google disk](https://drive.google.com/file/d/11baEpRrvHoqxORFPSdJiGWusb3Y4AnRE/view?usp=sharing).
