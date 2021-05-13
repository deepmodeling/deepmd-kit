# Train a Deep Potential model using descriptor `"se_e2_a"`

The notation of `se_e2_a` is short for the Deep Potential Smooth Edition (DeepPot-SE) constructed from all information (both angular and radial) of atomic configurations. The `e2` stands for the embedding with two-atoms information. This descriptor was described in detail in [the DeepPot-SE paper](https://arxiv.org/abs/1805.09003).

In this example we will train a DeepPot-SE model for a water system.  A complete training input script of this example can be find in the directory. 
```bash
$deepmd_source_dir/examples/water/se_e2_a/input.json
```
With the training input script, data (please read the [warning](#warning)) are also provided in the example directory. One may train the model with the DeePMD-kit from the directory.

The contents of the example:
- [The training input](#the-training-input-script)
- [Train a Deep Potential model](#train-a-deep-potential-model)
- [Warning](#warning)

## The training input script

A working training script using descriptor `se_e2_a` is provided as `input.json` in the same directory as this README.

The `input.json` is divided in several sections, `model`, `learning_rate`, `loss` and `training`. 

For more information, one can find the [a full documentation](https://deepmd.readthedocs.io/en/master/train-input.html) on the training input script.

### Model
The `model` defines how the model is constructed, for example
```json=
    "model": {
	"type_map":	["O", "H"],
	"descriptor" :{
            ...
	},
	"fitting_net" : {
            ...
	}
    }
```
We are looking for a model for water, so we have two types of atoms. The atom types are recorded as integers. In this example, we denote `0` for oxygen and `1` for hydrogen. A mapping from the atom type to their names is provided by `type_map`. 

The model has two subsections `descritpor` and `fitting_net`, which defines the descriptor and the fitting net, respectively. The `type_map` is optional, which provides the element names (but not necessarily to be the element name) of the corresponding atom types.

#### Descriptor
The construction of the descriptor is given by section `descriptor`. An example of the descriptor is provided as follows
```json=
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


#### Fitting
The construction of the fitting net is give by section `fitting_net`
```json=
	"fitting_net" : {
	    "neuron":		[240, 240, 240],
	    "resnet_dt":	true,
	    "seed":		1
	},
```
* `neuron` specifies the size of the fitting net. If two neighboring layers are of the same size, then a [ResNet architecture](https://arxiv.org/abs/1512.03385) is built between them. 
* If the option `resnet_dt` is set `true`, then a timestep is used in the ResNet. 
* `seed` gives the random seed that is used to generate random numbers when initializing the model parameters.

### Learning rate

The `learning_rate` section in `input.json` is given as follows
```json=
    "learning_rate" :{
	"type":		"exp",
	"start_lr":	0.001,
	"stop_lr":	3.51e-8,
	"decay_steps":	5000,
	"_comment":	"that's all"
    }
```
* `start_lr` gives the learning rate at the beginning of the training.
* `stop_lr` gives the learning rate at the end of the training. It should be small enough to ensure that the network parameters satisfactorily converge. 
* During the training, the learning rate decays exponentially from `start_lr` to `stop_lr` following the formula.
    ```
    lr(t) = start_lr * decay_rate ^ ( t / decay_steps )
    ```
    where `t` is the training step.
       
### Loss

The loss function of DeePMD-kit is given by
```
loss = pref_e * loss_e + pref_f * loss_f + pref_v * loss_v
```
where `loss_e`, `loss_f` and `loss_v` denote the loss in energy, force and virial, respectively. `pref_e`, `pref_f` and `pref_v` give the prefactors of the energy, force and virial losses. The prefectors may not be a constant, rather it changes linearly with the learning rate. Taking the force prefactor for example, at training step `t`, it is given by
```math
pref_f(t) = start_pref_f * ( lr(t) / start_lr ) + limit_pref_f * ( 1 - lr(t) / start_lr )
```
where `lr(t)` denotes the learning rate at step `t`. `start_pref_f` and `limit_pref_f` specifies the `pref_f` at the start of the training and at the limit of `t -> inf`.

The `loss` section in the `input.json` is 
```json=
    "loss" : {
	"start_pref_e":	0.02,
	"limit_pref_e":	1,
	"start_pref_f":	1000,
	"limit_pref_f":	1,
	"start_pref_v":	0,
	"limit_pref_v":	0
    }
```
The options `start_pref_e`, `limit_pref_e`, `start_pref_f`, `limit_pref_f`, `start_pref_v` and `limit_pref_v` determine the start and limit prefactors of energy, force and virial, respectively.

If one does not want to train with virial, then he/she may set the virial prefactors `start_pref_v` and `limit_pref_v` to 0.
    
### Training parameters

Other training parameters are given in the `training` section.
```json=
    "training": {
 	"training_data": {
	    "systems":		["../data_water/data_0/", "../data_water/data_1/", "../data_water/data_2/"],
	    "batch_size":	"auto"
	},
	"validation_data":{
	    "systems":		["../data_water/data_3"],
	    "batch_size":	1,
	    "numb_btch":	3
	},

	"numb_step":	1000000,
	"seed":		1,
	"disp_file":	"lcurve.out",
	"disp_freq":	100,
	"save_freq":	1000
    }
```
The sections `"training_data"` and `"validation_data"` give the training dataset and validation dataset, respectively. Taking the training dataset for example, the keys are explained below:
* `systems` provide paths of the training data systems. DeePMD-kit allows you to provide multiple systems. This key can be a `list` or a `str`.
    * `list`: `systems` gives the training data systems.
    * `str`: `systems` should be a valid path. DeePMD-kit will recursively search all data systems in this path.
* At each training step, DeePMD-kit randomly pick `batch_size` frame(s) from one of the systems. The probability of using a system is by default in proportion to the number of batches in the system. More optional are available for automatically determining the probability of using systems. One can set the key `auto_prob` to
    * `"prob_uniform"` all systems are used with the same probability.
    * `"prob_sys_size"` the probability of using a system is in proportional to its size (number of frames).
    * `"prob_sys_size; sidx_0:eidx_0:w_0; sidx_1:eidx_1:w_1;..."` the `list` of systems are divided into blocks. The block `i` has systems ranging from `sidx_i` to `eidx_i`. The probability of using a system from block `i` is in proportional to `w_i`. Within one block, the probability of using a system is in proportional to its size.
* An example of using `"auto_prob"` is given as below. The probability of using `systems[2]` is 0.4, and the sum of the probabilities of using `systems[0]` and `systems[1]` is 0.6. If the number of frames in `systems[1]` is twice as `system[0]`, then the probability of using `system[1]` is 0.4 and that of `system[0]` is 0.2.
```json=
 	"training_data": {
	    "systems":		["../data_water/data_0/", "../data_water/data_1/", "../data_water/data_2/"],
	    "auto_prob":	"prob_sys_size; 0:2:0.6; 2:3:0.4",
	    "batch_size":	"auto"
	}
```
* The probability of using systems can also be specified explicitly with key `"sys_prob"` that is a list having the length of the number of systems. For example
```json=
 	"training_data": {
	    "systems":		["../data_water/data_0/", "../data_water/data_1/", "../data_water/data_2/"],
	    "sys_prob":	[0.5, 0.3, 0.2],
	    "batch_size":	"auto:32"
	}
```
* The key `batch_size` specifies the number of frames used to train or validate the model in a training step. It can be set to
    * `list`: the length of which is the same as the `systems`. The batch size of each system is given by the elements of the list.
    * `int`: all systems use the same batch size.
    * `"auto"`: the same as `"auto:32"`, see `"auto:N"`
    * `"auto:N"`: automatically determines the batch size so that the `batch_size` times the number of atoms in the system is no less than `N`.
* The key `numb_batch` in `validate_data` gives the number of batches of model validation. Note that the batches may not be from the same system

Other keys in the `training` section are explained below:
* `numb_step` The number of training steps.
* `seed` The random seed for getting frames from the training data set.
* `disp_file` The file for printing learning curve.
* `disp_freq` The frequency of printing learning curve. Set in the unit of training steps
* `save_freq` The frequency of saving check point.


## Train a Deep Potential model
When the input script is prepared, one may start training by 
```bash=
dp train input.json
```
By default, the verbosity level of the DeePMD-kit is `INFO`, one may see a lot of important information on the code and environment showing on the screen. Among them two pieces of information regarding data systems worth special notice. 
```bash=
DEEPMD INFO    ---Summary of DataSystem: training     -----------------------------------------------
DEEPMD INFO    found 3 system(s):
DEEPMD INFO                                        system  natoms  bch_sz   n_bch   prob  pbc
DEEPMD INFO                         ../data_water/data_0/     192       1      80  0.250    T
DEEPMD INFO                         ../data_water/data_1/     192       1     160  0.500    T
DEEPMD INFO                         ../data_water/data_2/     192       1      80  0.250    T
DEEPMD INFO    --------------------------------------------------------------------------------------
DEEPMD INFO    ---Summary of DataSystem: validation   -----------------------------------------------
DEEPMD INFO    found 1 system(s):
DEEPMD INFO                                        system  natoms  bch_sz   n_bch   prob  pbc
DEEPMD INFO                          ../data_water/data_3     192       1      80  1.000    T
DEEPMD INFO    --------------------------------------------------------------------------------------
```
The DeePMD-kit prints detailed informaiton on the training and validation data sets. The data sets are defined by `"training_data"` and `"validation_data"` defined in the `"training"` section of the input script. The training data set is composed by three data systems, while the validation data set is composed by one data system. The number of atoms, batch size, number of batches in the system and the probability of using the system are all shown on the screen. The last column presents if the periodic boundary condition is assumed for the system. 

During the training, the error of the model is tested every `disp_freq` training steps with the batch used to train the model and with `numb_btch` batches from the validating data. The training error and validation error are printed correspondingly in the file `disp_file`. The batch size can be set in the input script by the key `batch_size` in the corresponding sections for training and validation data set. An example of the output 
```bash=
#  step      rmse_val    rmse_trn    rmse_e_val  rmse_e_trn    rmse_f_val  rmse_f_trn         lr
      0      3.33e+01    3.41e+01      1.03e+01    1.03e+01      8.39e-01    8.72e-01    1.0e-03
    100      2.57e+01    2.56e+01      1.87e+00    1.88e+00      8.03e-01    8.02e-01    1.0e-03
    200      2.45e+01    2.56e+01      2.26e-01    2.21e-01      7.73e-01    8.10e-01    1.0e-03
    300      1.62e+01    1.66e+01      5.01e-02    4.46e-02      5.11e-01    5.26e-01    1.0e-03
    400      1.36e+01    1.32e+01      1.07e-02    2.07e-03      4.29e-01    4.19e-01    1.0e-03
    500      1.07e+01    1.05e+01      2.45e-03    4.11e-03      3.38e-01    3.31e-01    1.0e-03
```
The file contains 8 columns, form right to left, are the training step, the validation loss, training loss, root mean square (RMS) validation error of energy, RMS training error of energy, RMS validation error of force, RMS training error of force and the learning rate. The RMS error (RMSE) of the energy is normalized by number of atoms in the system.

## Warning
It is warned that the example water data (in folder `examples/water/data`) is of very limited amount, is provided only for testing purpose, and should not be used to train a productive model.



