# DPA-1: Pretraining of Attention-based Deep Potential Model for Molecular Simulation 

![ALT](../images/model_se_atten.png "model_se_atten")

Here we propose DPA-1, a Deep Potential model with a novel attention mechanism, which is highly effective for representing the conformation and chemical spaces of atomic systems and learning the PES.

See [this paper](https://arxiv.org/abs/2208.08236) for more information. DPA-1 is implemented as a new descriptor `"se_atten"` for model training, which can be used after simply editing the input.json.

# Installation 
DPA-1 will be merged into DeePMD-kit official repo: [github](https://github.com/deepmodeling/deepmd-kit), and before that, for early adopters, you can refer to the following steps for a quick start:

Get the DeePMD-kit source code by `git clone` from a temporary repo:
```bash
cd /some/workspace
git clone --recursive https://github.com/iProzd/deepmd-kit.git deepmd-kit
```
The `--recursive` option clones all [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules) needed by DeePMD-kit.

Note that, you **must change to the devel branch** for further installation. And for convenience, you may want to record the location of source to a variable, saying `deepmd_source_dir` by
```bash
cd deepmd-kit
git checkout devel
deepmd_source_dir=`pwd`
```

Then you can refer to [standard installation](../install/install-from-source.md#install-the-python-interface) of python interface in DeePMD-kit. After that, you can smoothly use the DPA-1 model with following instructions.

# Introduction to new features of DPA-1
Next we will list the detail settings in input.json and the data format, especially for large systems with dozens of elements. An example of DPA-1 input can be found in [here](../../examples/water/se_atten/input.json).

## Descriptor `"se_atten"`

The notation of `se_atten` is short for the Deep Potential Smooth Edition with an Attention Mechanism and Type Embedding. The `e2` stands for the embedding with two-atoms information. 
This descriptor was described in detail in [the DPA-1 paper](https://arxiv.org/abs/2208.08236) and the images above.

In this example we will train a DPA-1 model for a water system.  A complete training input script of this example can be find in the directory. 
```bash
$deepmd_source_dir/examples/water/se_atten/input.json
```
With the training input script, data are also provided in the example directory. One may train the model with the DeePMD-kit from the directory.

An example of the descriptor is provided as follows
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
* **{ref}`sel <model/descriptor[se_atten]/sel>`** gives the maximum possible number of neighbors in the cut-off radius. It is an int. Note that this number highly effects the efficiency of training, which we usually use less than 200. (We use 120 for training 56 elements in [OC2M dataset](https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md))
* The {ref}`neuron <model/descriptor[se_atten]/neuron>` specifies the size of the embedding net. From left to right the members denote the sizes of each hidden layer from input end to the output end, respectively. If the outer layer is of twice size as the inner layer, then the inner layer is copied and concatenated, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them.
* The {ref}`axis_neuron <model/descriptor[se_atten]/axis_neuron>` specifies the size of submatrix of the embedding matrix, the axis matrix as explained in the [DeepPot-SE paper](https://arxiv.org/abs/1805.09003) 
* If the option {ref}`resnet_dt <model/descriptor[se_atten]/resnet_dt>` is set to `true`, then a timestep is used in the ResNet.
* {ref}`seed <model/descriptor[se_atten]/seed>` gives the random seed that is used to generate random numbers when initializing the model parameters.
* {ref}`attn <model/descriptor[se_atten]/attn>` sets the length of hidden vector during scale-dot attention computation.
* {ref}`attn_layer <model/descriptor[se_atten]/attn_layer>` sets the number of layers in attention mechanism.
* {ref}`attn_mask <model/descriptor[se_atten]/attn_mask>` determines whether to mask the diagonal in the attention weights and False is recommended.
* {ref}`attn_dotr <model/descriptor[se_atten]/attn_dotr>` determines whether to dot the relative coordinates on the attention weights as a gated scheme, True is recommended.

## Fitting `"ener"`
DPA-1 only support `"ener"` fitting type, and you can refer [here](train-energy.md) for detail information.

## Type embedding
DPA-1 only support models with type embeddings on. And the default setting is as follows:
```json
"type_embedding":{
            "neuron":           [2, 4, 8],
            "resnet_dt":        false,
            "seed":             1
        },
```
You can add these settings in input.json if you want to change the defaul ones, see [here](train-se-e2-a-tebd.md) for detail information.


## Type map
For training a large systems, especially those with dozens of elements, the {ref}`type <model/type_map>` determines the element index of training data:
```json
"type_map": [
   "Mg",
   "Al",
   "Cu"
  ]
```
which should include all the elements in the dataset you want to train on. The detail of data format can be found in [here](data/data-conv.md).
Note that this data format requires that, only those frames with the same fingerprint(i.e. the number of atoms of different element) can be put together as a unit system.
This may lead to sparse frame number in those rare systems. 

An ideal way is to put systems with same total number of atoms together, which is the way we trained DPA-1 on OC2M. This API will be uploaded on dpdata soon for a more convenient experience.

# Training example
Here we upload the AlMgCu example showed in the paper, you can download here:
[Baidu disk](https://pan.baidu.com/s/1Mk9CihPHCmf8quwaMhT-nA?pwd=d586);
[Google disk](https://drive.google.com/file/d/11baEpRrvHoqxORFPSdJiGWusb3Y4AnRE/view?usp=sharing).





