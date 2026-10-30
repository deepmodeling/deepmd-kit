# Advanced options

In this section, we will take `$deepmd_source_dir/examples/water/se_e2_a/input.json` as an example of the input file.

## Learning rate

See {doc}`learning-rate` for detailed documentation on learning rate schedules.

## Optimizer

The {ref}`optimizer <optimizer>` section in `input.json` is given as follows

```json
    "optimizer" :{
        "type": "Adam",
        "_comment": "that's all"
    }
```

- TensorFlow/Paddle: only {ref}`Adam <optimizer[Adam]>` is supported.
- PyTorch: {ref}`Adam <optimizer[Adam]>`, {ref}`AdamW <optimizer[AdamW]>`, {ref}`LKF <optimizer[LKF]>`, {ref}`AdaMuon <optimizer[AdaMuon]>`, {ref}`HybridMuon <optimizer[HybridMuon]>`.
- {ref}`adam_beta1 <optimizer[Adam]/adam_beta1>` and {ref}`adam_beta2 <optimizer[Adam]/adam_beta2>` control the Adam/AdamW moment decay.
- {ref}`weight_decay <optimizer[Adam]/weight_decay>` applies L2 penalty in Adam, while {ref}`weight_decay <optimizer[AdamW]/weight_decay>` is decoupled in AdamW. TensorFlow does not support weight decay in Adam.

## Training parameters

Other training parameters are given in the {ref}`training <training>` section.

```json
    "training": {
        "training_data": {
            "systems":    ["../data_water/data_0/", "../data_water/data_1/", "../data_water/data_2/"],
            "batch_size": "auto"
        },
        "validation_data":{
            "systems":    ["../data_water/data_3"],
            "batch_size": 1,
            "numb_btch":  3
        },
        "mixed_precision": {
            "output_prec":  "float32",
            "compute_prec": "float16"
        },

        "numb_steps": 1000000,
        "seed":       1,
        "disp_file":  "lcurve.out",
        "disp_freq":  100,
        "save_freq":  1000
    }
```

The sections {ref}`training_data <training/training_data>` and {ref}`validation_data <training/validation_data>` give the training dataset and validation dataset, respectively. Taking the training dataset for example, the keys are explained below:

- {ref}`systems <training/training_data/systems>` provide paths of the training data systems. DeePMD-kit allows you to provide multiple systems with different numbers of atoms. This key can be a `list` or a `str`.
  - `str`: {ref}`systems <training/training_data/systems>` should be a valid path. It can be a system directory path (containing 'type.raw'), a DeePMD HDF5 container or group, or a parent directory path to recursively search for all system subdirectories.
  - `list`: {ref}`systems <training/training_data/systems>` gives a list of paths. Each string item in the list is processed the same way as individual string inputs, i.e., each path can be a system directory or a parent directory to recursively search for all system subdirectories.
  - The PyTorch and PyTorch Exportable backends also accept LMDB paths, lists of LMDB paths, and parent directories containing LMDB datasets. An LMDB dataset is terminal: discovery does not inspect its contents as additional datasets. NPY and LMDB sources may appear in the same group. Other backends discover DeePMD NPY/HDF5 systems only.
- At each training step, DeePMD-kit randomly picks {ref}`batch_size <training/training_data/batch_size>` frame(s) from one of the systems. The probability of using a system is by default in proportion to the number of batches in the system. More options are available for automatically determining the probability of using systems. One can set the key {ref}`auto_prob <training/training_data/auto_prob>` to
  - `"prob_uniform"` each retained system supplies the same long-run share of sampled frames. Batch-selection probabilities account for the number of frames returned per batch.
  - `"prob_sys_size"` the probability of using a system is proportional to its number of batches.
  - `"prob_sys_size; sidx_0:eidx_0:w_0; sidx_1:eidx_1:w_1;..."` the `list` of systems is divided into blocks. Block `i` has systems ranging from `sidx_i` to `eidx_i`. The probability of using a system from block `i` is proportional to `w_i`. Within one block, the probability of using a system is proportional to its size.
- An example of using `"auto_prob"` is given below. The probability of using `systems[2]` is 0.4, and the sum of the probabilities of using `systems[0]` and `systems[1]` is 0.6. If the number of frames in `systems[1]` is twice of `system[0]`, then the probability of using `system[1]` is 0.4 and that of `system[0]` is 0.2.

```json
    "training_data": {
        "systems":    ["../data_water/data_0/", "../data_water/data_1/", "../data_water/data_2/"],
        "auto_prob":  "prob_sys_size; 0:2:0.6; 2:3:0.4",
        "batch_size": "auto"
    }
```

- The probability of using systems can also be specified explicitly with key {ref}`sys_probs <training/training_data/sys_probs>` which is a list having the length of the number of systems. For example

```json
    "training_data": {
        "systems":    ["../data_water/data_0/", "../data_water/data_1/", "../data_water/data_2/"],
        "sys_probs":  [0.5, 0.3, 0.2],
        "batch_size": "auto:32"
    }
```

- The key {ref}`batch_size <training/training_data/batch_size>` specifies the number of frames used to train or validate the model in a training step. It can be set to
  - `list`: the length of which is the same as the {ref}`systems`. The batch size of each system is given by the elements of the list.

  - `int`: all systems use the same batch size.

  - `"auto"`: the same as `"auto:32"`, see `"auto:N"`

  - `"auto:N"`: automatically determines the batch size so that the {ref}`batch_size <training/training_data/batch_size>` times the number of atoms in the system is **no less than** `N`.

  - `"max:N"`: automatically determines the batch size so that the {ref}`batch_size <training/training_data/batch_size>` times the number of atoms in the system is **no more than** `N`. The minimum batch size is 1. **Supported backends**: PyTorch {{ pytorch_icon }}, Paddle {{ paddle_icon }}

  - `"filter:N"`: the same as `"max:N"` but removes the systems with the number of atoms larger than `N` from the data set. Throws an error if no system is left in a dataset. **Supported backends**: PyTorch {{ pytorch_icon }}, Paddle {{ paddle_icon }}

  - `"mix:N"`: mixed-nloc batching for LMDB data sets. Frames of different atom counts share a batch, which is closed only when the next frame would push its atom axis past `N`. How that axis is laid out follows from the model and needs no configuration of its own:

    - A descriptor reading a **flat** node axis — the graph route of the PyTorch Exportable backend — takes the frames of a batch concatenated. `N` counts real atoms, nothing is padded, and the frames are packed in the shuffled order they arrive in, so one batch stays mixed in system size.

    - Every other descriptor reads a **rectangular** `(nframes, nloc, ...)` axis, on which the frames of a batch must share one atom count. Frames are therefore sorted by atom count and padded up to the widest one of their batch, which makes `N` the padded-slot count `nframes * max_nloc`. The padded slots hold phantom atoms marked `atype = -1`, which the neighbor list, the model and the loss all skip. Sorting is what keeps their number small, and it also makes the packing optimal: no arrangement of the same frames under the same budget yields fewer batches.

    Either way, a lone frame with more than `N` atoms still forms a batch of its own. Every other rule keeps a batch uniform in atom count, which leaves batches under-filled wherever an atom-count group is small; `"mix:N"` keeps them all close to `N` atoms instead. **Supported backends**: PyTorch {{ pytorch_icon }} (rectangular layout only), PyTorch Exportable

    Filling batches also makes the *sampling weight* more faithful. Per-atom loss terms (force, atomic energy) pool over a batch's real labels, so a frame already counts in proportion to its atom count there. Frame-level terms (energy, virial) weigh the frames of a batch equally, so a frame counts for `1 / nframes`, and an atom budget makes `nframes` follow the atom count. `"max:N"` distorts that wherever an atom-count group is too small to fill a batch — its few frames form a short batch and each is over-weighted many times over — while `"mix:N"` packs them with their neighbours and lands much closer to the intended weighting.
- The key {ref}`numb_batch <training/validation_data/numb_btch>` in {ref}`validate_data <training/validation_data>` gives the number of batches of model validation. Note that the batches may not be from the same system

### Data groups

`training_data` accepts either the dictionary shown above or a nonempty list of dictionaries. Each dictionary defines a group with its own `systems`, `batch_size`, and within-group sampling options. The optional `prob` specifies the probability of drawing a training batch from the entire group, including every dataset found recursively under its paths:

```json
"training_data": [
    {
        "systems": "../data/collection_a",
        "batch_size": "auto:32",
        "prob": 0.6
    },
    {
        "systems": ["../data/collection_b", "../data/extra"],
        "batch_size": "auto:64",
        "prob": 0.4
    }
]
```

The first group's datasets together receive probability 0.6, regardless of how many datasets its path contains. The second group's datasets together receive probability 0.4. Batches do not combine frames from different groups.

`prob` is an absolute probability between 0 and 1. If every group specifies it, the probabilities must sum to 1. Groups without `prob` share the remaining probability in proportion to their batch counts. Omitting `prob` everywhere therefore preserves size-based sampling; it does not give equally sized shares to differently sized groups. Within each group, `auto_prob` determines the system sampling rule; explicit `sys_probs` specify batch-selection probabilities.

For a group containing exactly one LMDB, `auto_prob` addresses that LMDB's internal systems, independently of the group's `prob`:

```json
"training_data": [
    {
        "systems": "../data/first.lmdb",
        "batch_size": "filter:450",
        "auto_prob": "prob_sys_size;0:10:0.5;10:30:0.5",
        "prob": 0.6
    },
    {
        "systems": "../data/second.lmdb",
        "batch_size": "filter:450",
        "prob": 0.4
    }
]
```

If a group expands to multiple sources and includes LMDB, `auto_prob` can be `"prob_sys_size"` (the default) or `"prob_uniform"`, and `sys_probs` must be omitted. Custom block rules still require a single LMDB because their system indices are local to that dataset. To assign source weights or configure different block rules, put each LMDB in a separate group. Existing dictionary configurations, NPY `sys_probs`, and single-LMDB block rules remain supported.

For `"prob_uniform"`, each NPY/HDF5 system and each nonempty internal LMDB system counts once after `filter:N` is applied. An LMDB without internal system identifiers counts as one system. The rule is independent of how systems are split among LMDB files: systems containing 100 and 900 frames each supply half the sampled frames. LMDB sampling balances the internal frame counts before applying the native batch packing; `mix:N` continues to combine systems. Source-selection weights also correct for differing batch frame counts. Uniform frame shares do not imply equal contributions to each batch or to the loss, and subsequent geometric frame rejection can change the retained training shares.

For an LMDB containing `N` retained frames in `M` nonempty systems, a uniform sampling pass draws `ceil(N / M)` frames from each system. Before distributed batch padding, the pass therefore contains fewer than `N + M` frame indices, rather than scaling with the largest system. Larger systems contribute a random subset without replacement; smaller systems retain their frames and repeat them to fill their quota. A pass need not visit every frame of a larger system. Subsets change with the sampler epoch and remain reproducible from its seed. `numb_epoch` and omitted outer-group probabilities use the resulting native batch counts; an explicit outer `prob` is unchanged. Custom block rules retain their existing non-shrinking targets.

This frame-share definition intentionally changes older NPY/HDF5 `prob_uniform` behavior when batch sizes differ. For example, systems returning batches of 1 and 10 frames receive batch probabilities of 10/11 and 1/11, rather than 1/2 each. Short native batches are included in the correction. Explicit `sys_probs` and outer group `prob` remain batch probabilities.

`validation_data` remains a dictionary. Its `systems` accepts the same recursive paths and multiple-LMDB inputs on supported backends; `numb_btch` continues to apply to the whole validation dataset.

### Automatic format conversion

Set `format` on a training or validation data group to a dpdata format such as
`extxyz`, `abacus/scf`, or `ase/traj`. Each selected input path is converted and
cached separately; `out_format` defaults to `deepmd/lmdb`. LMDB requires an
explicit model `type_map`. Paddle, data modifiers, and explicit `sys_probs`
require another output format such as `deepmd/hdf5`.

Automatic conversion loads each input path eagerly through dpdata. Its supported
scale is limited to inputs whose parsed arrays **and intermediate copies fit in
available RAM**; it is not a streaming converter. For large trajectories, split
inputs into separate paths or convert them offline before training. Multiple
converted paths use the same data-group sampling rules described above, including
the current frame-share definition of `prob_uniform`.

### Mixed precision

The section {ref}`mixed_precision <training/mixed_precision>` specifies the mixed precision settings, which will enable the mixed precision training workflow for DeePMD-kit. The keys are explained below:

- {ref}`output_prec <training/mixed_precision/output_prec>` precision used in the output tensors, only `float32` is supported currently.
- {ref}`compute_prec <training/mixed_precision/compute_prec>` precision used in the computing tensors, only `float16` is supported currently.
  Note there are several limitations about mixed precision training:
- Only {ref}`se_e2_a <model[standard]/descriptor[se_e2_a]>` type descriptor is supported by the mixed precision training workflow.
- The precision of the embedding net and the fitting net are forced to be set to `float32`.

Other keys in the {ref}`training <training>` section are explained below:

- {ref}`numb_steps <training/numb_steps>` The number of training steps.
- {ref}`seed <training/seed>` The random seed for getting frames from the training data set.
- {ref}`disp_file <training/disp_file>` The file for printing learning curve.
- {ref}`disp_freq <training/disp_freq>` The frequency of printing learning curve. Set in the unit of training steps
- {ref}`save_freq <training/save_freq>` The frequency of saving checkpoint.
- {ref}`save_dir <training/save_dir>` The directory where periodic checkpoints are written (PyTorch-TorchScript and PyTorch Exportable backends). It is created recursively if missing, while the `model.ckpt.pt` symlinks and the `checkpoint` pointer file stay in the working directory. Defaults to the working directory.
- {ref}`max_ckpt_keep <training/max_ckpt_keep>` The number of recent periodic checkpoints retained for each checkpoint family. EMA checkpoints inherit this window by default; {ref}`ema_ckpt_keep <training/ema_ckpt_keep>` may override it when a different EMA window is required.
- {ref}`ckpt_keep_ratio <training/ckpt_keep_ratio>` An alternative to `max_ckpt_keep` (PyTorch-TorchScript and PyTorch Exportable backends) that keeps a sliding window of `ceil(ckpt_keep_ratio * ceil(numb_steps / save_freq))` most recent checkpoints, i.e. the final `ckpt_keep_ratio` fraction of the run by step. It overrides `max_ckpt_keep` (and `ema_ckpt_keep`) when set, and works the same whether the run length is given by `numb_steps` or `numb_epoch`.

## Options and environment variables

Several command line options can be passed to `dp train`, which can be checked with

```bash
$ dp train --help
```

An explanation will be provided

```{program-output} dp train -h

```

**`--init-model model.ckpt`**, initializes the model training with an existing model that is stored in the path prefix of checkpoint files `model.ckpt`, the network architectures should match.

**`--restart model.ckpt`**, continues the training from the checkpoint `model.ckpt`.
For the PyTorch and PyTorch Exportable backends, when checkpoint retention is
enabled, writing the next periodic or EMA checkpoint at step `N` removes
checkpoints from the same family numbered above `N`. This prevents remnants of
a longer earlier run from displacing the newly written checkpoint from the
retention window.

PyTorch Exportable optimizer checkpoints also retain each rank's per-task
training-data cursor and task/system selection RNG. With the same immutable
data at the same locations, batching settings, and number of ranks, a restart
continues with the next LMDB or directory/HDF5 training batches rather than
starting their iteration again. Data-source
progress is compact and restored lazily; checkpoints do not store decoded
frames or shuffle permutations. Statistics, compilation warmup, and learning
curve evaluation do not consume this training progression.
Distributed ranks exchange this compact state at checkpoint boundaries, not
on every batch; rank-local filtering may legitimately advance their cursors
by different amounts.

Restarting with a different number of ranks still restores the model, optimizer,
and training step, but warns and keeps newly initialized data cursors and
selection RNG instead of remapping the saved rank-local progress. Data may be
repeated, and exact data-order continuation is not guaranteed in this case.
Malformed progress envelopes remain errors; by default, so do incompatible data
settings at the same world size.

PyTorch Exportable initializes rank-local data sampling from `training.seed`
and the rank before constructing data sources, including directory and HDF5
sources. Task selection has an independent RNG shared across ranks, so local
data filtering cannot change the task sequence. Without a configured seed,
the chief initializes that task RNG and broadcasts its state once at startup.
Both RNG streams resume from checkpoints. The configured seed is also part of
the checkpoint's data-progress identity: changing it requires an explicit reset,
independently of the input format. Earlier experimental progress records without
seed/task-RNG identity likewise require a reset; checkpoints without data
progress retain their legacy restart behavior.

Grouped inputs also preserve each group's child-selection RNG and every native
source's cursor, including nested groups combining directory/HDF5 and LMDB data.
The group topology, assigned probabilities, and uniform-sampling policy must
remain compatible when restoring progress.

For PyTorch Exportable, set `training.reset_data_progress` to `true` when
intentionally restarting with relocated data or changed data settings. This
explicitly keeps newly initialized data cursors and task/system selection RNG
for **all** training tasks; model weights, optimizer, learning-rate schedule,
and training step still resume. A warning explains that batches may repeat and
exact data-order continuation is unavailable. The default is `false`, which
retains strict data-source compatibility checks. Malformed progress envelopes
remain errors even when resetting.

Source paths are identity guards, not content fingerprints: equal frame counts,
atom types, or LMDB transaction numbers do not prove that two datasets contain
the same frames. Consequently, copying an unchanged dataset to another location
requires an explicit data-progress reset; automatic exact continuation after
relocation is not supported. Directory/HDF5 paths must also retain the same path
spelling, including the HDF5 group. `reset_data_progress` has no effect on fresh
training, `--init-model`, finetuning, frozen models, or older checkpoints without
data progress.

Directory/HDF5 batches now shuffle indices into a canonical, unshuffled set.
For a single-set source, later passes therefore differ from older releases
that shuffled their already shuffled cache. This also prevents `numb_copy`
multiplicities from compounding over repeated passes.

Older checkpoints without data progress retain the previous restart behavior.
`--init-model`, finetuning, and EMA-only deployment snapshots do not restore data
progress. This is a data-order guarantee, not a guarantee of bitwise-identical
training or of restoring every stochastic data transform.

**`--init-frz-model frozen_model.pb`**, initializes the training with an existing model that is stored in `frozen_model.pb`.

**`--skip-neighbor-stat`** will skip calculating neighbor statistics if one is concerned about performance. Some features will be disabled.

To maximize the performance, one should follow [FAQ: How to control the parallelism of a job](../troubleshooting/howtoset_num_nodes.md) to control the number of threads.
See [Runtime environment variables](../env.md) for all runtime environment variables.

## Adjust `sel` of a frozen model {{ tensorflow_icon }}

One can use `--init-frz-model` features to adjust (increase or decrease) [`sel`](../model/sel.md) of an existing model. Firstly, one needs to adjust [`sel`](./train-input.rst) in `input.json`. For example, adjust from `[46, 92]` to `[23, 46]`.

```json
"model": {
    "descriptor": {
        "sel": [23, 46]
    }
}
```

To obtain the new model at once, [`numb_steps`](./train-input.rst) should be set to zero:

```json
"training": {
    "numb_steps": 0
}
```

Then, one can initialize the training from the frozen model and freeze the new model at once:

```sh
dp train input.json --init-frz-model frozen_model.pb
dp freeze -o frozen_model_adjusted_sel.pb
```

Two models should give the same result when the input satisfies both constraints.

Note: At this time, this feature is only supported by [`se_e2_a`](../model/train-se-e2-a.md) descriptor with [`set_davg_true`](./train-input.rst) enabled, or `hybrid` composed of the above descriptors.
