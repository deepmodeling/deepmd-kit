# Advanced options

In this section, we will take `$deepmd_source_dir/examples/water/se_e2_a/input.json` as an example of the input file.

## Learning rate

The `learning_rate` section in `input.json` is given as follows
```json
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

## Training parameters

Other training parameters are given in the `training` section.
```json
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
* `systems` provide paths of the training data systems. DeePMD-kit allows you to provide multiple systems with different numbers of atoms. This key can be a `list` or a `str`.
    * `list`: `systems` gives the training data systems.
    * `str`: `systems` should be a valid path. DeePMD-kit will recursively search all data systems in this path.
* At each training step, DeePMD-kit randomly pick `batch_size` frame(s) from one of the systems. The probability of using a system is by default in proportion to the number of batches in the system. More optional are available for automatically determining the probability of using systems. One can set the key `auto_prob` to
    * `"prob_uniform"` all systems are used with the same probability.
    * `"prob_sys_size"` the probability of using a system is in proportional to its size (number of frames).
    * `"prob_sys_size; sidx_0:eidx_0:w_0; sidx_1:eidx_1:w_1;..."` the `list` of systems are divided into blocks. The block `i` has systems ranging from `sidx_i` to `eidx_i`. The probability of using a system from block `i` is in proportional to `w_i`. Within one block, the probability of using a system is in proportional to its size.
* An example of using `"auto_prob"` is given as below. The probability of using `systems[2]` is 0.4, and the sum of the probabilities of using `systems[0]` and `systems[1]` is 0.6. If the number of frames in `systems[1]` is twice as `system[0]`, then the probability of using `system[1]` is 0.4 and that of `system[0]` is 0.2.
```json
 	"training_data": {
	    "systems":		["../data_water/data_0/", "../data_water/data_1/", "../data_water/data_2/"],
	    "auto_prob":	"prob_sys_size; 0:2:0.6; 2:3:0.4",
	    "batch_size":	"auto"
	}
```
* The probability of using systems can also be specified explicitly with key `"sys_prob"` that is a list having the length of the number of systems. For example
```json
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

## Options and environment variables

Several command line options can be passed to `dp train`, which can be checked with
```bash
$ dp train --help
```
An explanation will be provided
```
positional arguments:
  INPUT                 the input json database

optional arguments:
  -h, --help            show this help message and exit
 
  --init-model INIT_MODEL
                        Initialize a model by the provided checkpoint

  --restart RESTART     Restart the training from the provided checkpoint
 
  --init-frz-model INIT_FRZ_MODEL
                        Initialize the training from the frozen model.
```

**`--init-model model.ckpt`**, initializes the model training with an existing model that is stored in the checkpoint `model.ckpt`, the network architectures should match.

**`--restart model.ckpt`**, continues the training from the checkpoint `model.ckpt`.

**`--init-frz-model frozen_model.pb`**, initializes the training with an existing model that is stored in `frozen_model.pb`.

On some resources limited machines, one may want to control the number of threads used by DeePMD-kit. This is achieved by three environmental variables: `OMP_NUM_THREADS`, `TF_INTRA_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS`. `OMP_NUM_THREADS` controls the multithreading of DeePMD-kit implemented operations. `TF_INTRA_OP_PARALLELISM_THREADS` and `TF_INTER_OP_PARALLELISM_THREADS` controls `intra_op_parallelism_threads` and `inter_op_parallelism_threads`, which are  Tensorflow configurations for multithreading. An explanation is found [here](https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads).

For example if you wish to use 3 cores of 2 CPUs on one node, you may set the environmental variables and run DeePMD-kit as follows:
```bash
export OMP_NUM_THREADS=6
export TF_INTRA_OP_PARALLELISM_THREADS=3
export TF_INTER_OP_PARALLELISM_THREADS=2
dp train input.json
```

One can set other environmental variables:

| Environment variables | Allowed value          | Default value | Usage                      |
| --------------------- | ---------------------- | ------------- | -------------------------- |
| DP_INTERFACE_PREC     | `high`, `low`          | `high`        | Control high (double) or low (float) precision of training. |
