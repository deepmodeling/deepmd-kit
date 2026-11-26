# Descriptor DPA4 {{ pytorch_icon }} {{ jax_icon }}

> [!NOTE]
> Supported backends: PyTorch Exportable (`dp --pt-expt`, recommended), PyTorch (`dp --pt`) and JAX (`dp --jax`, training only).

DPA4 implements the SeZM (Smooth Equivariant Zone-bridging Model) architecture
for interatomic potentials. It learns atomic environments from the positions
and types of neighboring atoms, then predicts the energy. Forces and virials
are derivatives of that energy. Rotating the structure leaves the energy
unchanged and rotates the forces with it.

Choose DPA4 when accuracy matters most. It trains energy, property and spin
models and exports them to `.pt2` for inference and molecular dynamics. For a
faster model that also supports compression, see [DPA4C](dpa4c.md).

This guide follows a first project: train a model from a preset, export it and
run it. [Zone bridging](#zone-bridging), [spin](#spin) and a
[mixture of experts (MoE)](#mixture-of-experts-moe) are optional and have
sections of their own.

## Quick start

From the DeePMD-kit source directory, train the water example with the
PyTorch Exportable backend:

```bash
cd examples/water/dpa4
dp --pt-expt train input_preset.json
```

After training, export a checkpoint and test it on the validation data:

```bash
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model.pt2
dp test -m frozen_model.pt2 -s ../data/data_3
```

Replace `model.ckpt.pt` with the checkpoint you want to use. The export
compiles the model and takes a few minutes.

The [example input](../../examples/water/dpa4/input_preset.json) is a complete
training configuration. To use your own data, change `model.type_map` and the
system paths under `training.training_data` and `training.validation_data`.
See [training energy models](train-energy.md) for the data, loss and
learning-rate settings. For a short trial run, lower `training.numb_steps`.
On a GPU without native bfloat16 support, set `descriptor.use_amp` to `false`;
see [training precision](#training-precision).

The JSON examples below show sections to put in a complete input; they are not
standalone training files.

### Backends

| Backend            | Command        | Use                                                                                                   |
| ------------------ | -------------- | ----------------------------------------------------------------------------------------------------- |
| PyTorch Exportable | `dp --pt-expt` | Recommended. Needed for MoE.                                                                          |
| PyTorch            | `dp --pt`      | Property fitting, LoRA fine-tuning and direct-force denoising.                                        |
| JAX                | `dp --jax`     | Training only, including property fitting and LoRA fine-tuning. Use PyTorch Exportable for inference. |

## Choosing a model

### Presets

A preset is a named model size. It selects the descriptor and fitting-network
settings together, so the model section only names the preset, the elements
and the settings you want to change:

```json
{
  "model": {
    "preset": "dpa4-mini-v20260911",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "rcut": 6.0,
      "seed": 42
    },
    "fitting_net": {
      "seed": 42
    }
  }
}
```

A preset specifies an architecture; it does not load pretrained weights.
Settings written alongside the preset override it. `type_map` replaces the
preset's full list of 118 elements; individual settings inside `descriptor`
and `fitting_net` replace the corresponding preset values. The resolved
configuration is written to `out.json`.

Two families of presets are available. Sizes are listed from smaller to larger
models; compare validation errors and runtime on your data when choosing one.
Use the latest version of a family unless you reproduce an older model.

DPA4 presets are named `dpa4-<size>-<version>` and work with every backend:

| Version     | Available sizes                                             |
| ----------- | ----------------------------------------------------------- |
| `v20260911` | `nano`, `mini`, `neo`, `air`, `plus`, `pro`, `max`, `ultra` |
| `v20260820` | `nano`, `mini`, `neo`, `air`, `plus`, `pro`                 |

DPA4.5 presets are named `dpa4.5-<size>-<version>`. They define
O(3)-equivariant DPA4 models, run on the PyTorch Exportable backend, and can
be trained with the JAX backend:

| Version     | Available sizes                                    |
| ----------- | -------------------------------------------------- |
| `v20261111` | `nano`, `mini`, `neo`, `air`, `plus`, `pro`, `max` |

Every DPA4.5 preset conditions its descriptor on the charge and spin
multiplicity of the frame (`add_chg_spin_ebd`, default `[0, 1]`) and asks its
fitting network for the isolated-atom energy reference
([`vacuum_ref`](train-energy.md#isolated-atom-energy-reference)), which takes
effect on an input that assigns the isolated-atom energies
(`preset_out_bias`). An element is then referenced to the condition its table
entry names, or to the default condition when the entry names none, so the
table decides whether an atom without neighbors is pinned or keeps the
deviation learned for another electronic state.

Train DPA4.5 models with the HybridMuon optimizer and a weight decay of 0.03:

```json
{
  "optimizer": {
    "type": "HybridMuon",
    "weight_decay": 0.03
  }
}
```

The [DPA4.5 example](../../examples/water/dpa4/input_dpa45.json) is a complete
input with this optimizer and [zone bridging](#zone-bridging).

A DPA4.5 model also takes the total charge and the spin multiplicity of each
frame. Put a `charge_spin.npy` file with one `[charge, multiplicity]` pair per
frame in the sets of a system; frames without it are treated as neutral
singlets. Change that default with `descriptor.default_chg_spin`. At
inference, give the pair through the `charge_spin` keyword of the
[LAMMPS pair style](../third-party/lammps-command.md) or through
`atoms.info["charge_spin"]` in ASE.

For multi-task training, a preset beside `model_dict` applies to all branches
and their shared descriptor and fitting settings. A preset inside a branch
applies to that branch. See the
[multi-task preset example](../../examples/water/dpa4/input_multitask_preset.json).

### Model without a preset

Set `model.type` to `dpa4`, list the elements in `type_map`, and choose a cutoff
radius in Å:

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "rcut": 6.0
    },
    "fitting_net": {}
  }
}
```

This selects the DPA4 descriptor and energy fitting network with their default
settings. `DPA4`, `SeZM` and `sezm` are also accepted model names. The default
descriptor precision is `float32`. See
[the example without a preset](../../examples/water/dpa4/input.json) for a
complete input.

### Main options

Start with a preset and adjust the options that affect your task:

| Option                     | Use                                                                                        |
| -------------------------- | ------------------------------------------------------------------------------------------ |
| `rcut`                     | Set the neighbor cutoff in Å. A larger cutoff includes more atoms and increases cost.      |
| `channels`                 | Set the model width. Larger values use more computation and memory.                        |
| `n_blocks`                 | Set the number of layers that exchange information between neighboring atoms.              |
| `lmax`, `mmax`             | Control how much angular detail the model retains. Larger values increase cost.            |
| `l_schedule`, `m_schedule` | Set angular detail separately for each block. `l_schedule` also sets the number of blocks. |
| `n_radial`, `basis_type`   | Control how distances are represented.                                                     |
| `focus_dim`                | Set the width of the messages between neighbors; `0` uses `channels`.                      |
| `mixing_layers`            | Set the depth of each message.                                                             |

The {ref}`argument reference <model[dpa4]/descriptor[dpa4]>` lists all options
and their defaults. When `l_schedule` is set, as in the DPA4.5 presets, `lmax`
and `n_blocks` are not used, and a `mixing_layers` list has one entry per
block.

You do not need to set `sel`: the neighbor search uses every atom within
`rcut`, and `sel` only sets its initial capacity. The experimental `dens` mode
is the exception. There `sel` limits the neighbor list; use `auto` or choose a
value large enough for your data.

## Training

Use `loss.type: "ener"` for energy, force and virial training. The model accepts
both [standard](../data/system.md) and
[mixed-type](../data/system.md#mixed-type) DeePMD-kit data.

### Fine-tuning

To fine-tune a pretrained DPA4 checkpoint, prepare an input for the target data
and pass the checkpoint to training:

```bash
dp --pt-expt train input_finetune.json --finetune pretrained.pt
```

Keep the pretrained model's architecture when transferring its weights. See
[fine-tuning](../train/finetuning.md) for choosing fitting networks and
multi-task branches.

### LoRA fine-tuning

The PyTorch (`dp --pt`) and JAX (`dp --jax`) backends fine-tune a single-task
model with LoRA through `model.lora.rank` and `model.lora.alpha`. Use the
[LoRA input](../../examples/water/dpa4/lora_ft.json) with the `--finetune`
command. Best checkpoints include the trained adapters and can be deployed
without a separate adapter file.

### Training precision

`descriptor.use_amp`, which is `true` by default, uses bfloat16 mixed
precision during CUDA training and can reduce memory use. Set it to `false` on
a GPU without native bfloat16 support. It has no effect on a CPU.

Compilation and TF32 settings depend on the backend:

| Setting                    | PyTorch             | PyTorch Exportable        | JAX                    |
| -------------------------- | ------------------- | ------------------------- | ---------------------- |
| Compile training           | `model.use_compile` | `training.enable_compile` | Not needed             |
| Allow TF32 during training | `model.enable_tf32` | `training.enable_tf32`    | `training.enable_tf32` |

Training precision and inference precision are configured separately. Check
validation errors when changing precision settings.

### Other training tasks

For multi-task training, see the
[multi-task guide](../train/multi-task-training.md) and the
[DPA4 example](../../examples/water/dpa4/input_multitask.json).

The following tasks, like [LoRA fine-tuning](#lora-fine-tuning), use another
backend than PyTorch Exportable:

- **Property fitting**, with the PyTorch or the JAX backend. Set
  `fitting_net.type` and `loss.type` to `property`. The
  [property example](../../examples/property/train/input_dpa4.json) trains
  three properties from `band_prop.npy`; set `property_name`, `task_dim` and
  `intensive` to match your labels.
- **Direct-force denoising** (experimental), with the PyTorch backend and
  `loss.type: "dens"`. See
  [the denoising example](../../examples/water/dpa4/input_dens.json).

## Zone bridging

Zone bridging adds an analytical repulsive potential at short interatomic
distances. Use it when simulations may contain close contacts, such as
collisions. The analytical term is a screened Coulomb repulsion that falls
smoothly to zero at 1.1 times each element pair's covalent bond length and is
added to the learned energy. The pair term follows the descriptor precision:
FP64 for a double-precision descriptor, FP32 otherwise.

Enable it in the model configuration:

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "rcut": 6.0
    },
    "fitting_net": {},
    "bridging_method": "zbl"
  }
}
```

Choose `zbl` for the Ziegler–Biersack–Littmark universal potential or `nlh` for
the pair-specific Nordlund–Lehtola–Hobler potential. Both use four exponential
screening terms and subtract the cubic Taylor polynomial at each pair's end
radius. The NLH coefficients are fitted to reference repulsive energies for
individual element pairs, rather than to a universal screening curve.
`type_map` must contain chemical element names. See [the ZBL example](../../examples/water/dpa4/input-zbl.json) for a complete
input.

`bridging_truncation` sets where the analytical potential ends, as a multiple
of the covalent bond length of each element pair (default 1.1).
For each element pair, the learned term reads the physical separation unchanged
down to its data edge: the `bridging_support`-th smallest distinct per-frame
minimum separation (default 3), capped at the covalent bond length. Below the
edge the read distance changes smoothly and becomes constant at
`bridging_clamp` times the edge (default 0.75); `null` disables the clamp.
The bond direction is unchanged. Training keeps every frame unless
`training_data.min_pair_dist` is set.

To add bridging to a trained model, keep its architecture, add the bridging
settings and fine-tune; the data edges start at the covalent bond lengths and
follow the fine-tuning data.

A checkpoint fitted with the earlier switching window does not contain a
contact record and cannot reproduce its energies and forces with the truncated
potential. In the PyTorch backend, `--force-load` can initialize the missing
buffers, or a target with `bridging_clamp: null` can reuse the weights without
a clamp. Both start from a different energy function and require retraining;
neither restores the original potential.

Recalibrate an existing model's energy bias with
`dp change-bias --mode change`, which accounts for the model's complete energy,
including the analytical term. See [changing the energy bias](change-bias.md).

Bridging works with native spin and multi-rank LAMMPS.

The shipped NLH coefficients are this project's refit of the
[Nordlund, Hobler and Lehtola reference data](https://doi.org/10.5281/zenodo.14172633)
(v1.0, 2024, CC BY 4.0) for cubic-Taylor truncated zone bridging, rather than
the published coefficients. They use the
functional form in [Phys. Rev. A 111, 032818 (2025)](https://doi.org/10.1103/PhysRevA.111.032818).
Pairs containing elements heavier than uranium use normalized ZBL coefficients.

## Spin

DPA4 can take the magnetic moment of every atom as an input. Use it for
magnetic materials whose training data carry moments. Spin models train with
the PyTorch Exportable and PyTorch backends and the `ener_spin` loss; see
[training spin energy models](train-energy-spin.md) for the `spin` and
`force_mag` data formats and loss weights. Spin training cannot be combined
with `dens` mode.

Add a `spin` section to the model and select the scheme according to how the
training data were calculated:

| `spin.scheme`     | Use                                                                                                          |
| ----------------- | ------------------------------------------------------------------------------------------------------------ |
| `native:orbit`    | Data with spin–orbit coupling (SOC). The energy can depend on moment directions relative to the lattice.     |
| `native:exchange` | Data without SOC. Rotating all moments together while holding the lattice fixed leaves the energy unchanged. |
| `native`          | Shorthand for `native:orbit`.                                                                                |

Both schemes describe energies without an external magnetic field. Reversing
all moments leaves the energy unchanged. No virtual atoms are added.

```json
{
  "model": {
    "type": "dpa4",
    "type_map": [
      "Ni",
      "O"
    ],
    "spin": {
      "scheme": "native:orbit",
      "use_spin": [
        true,
        false
      ]
    },
    "descriptor": {
      "rcut": 6.0,
      "use_env_seed": true
    },
    "fitting_net": {}
  },
  "loss": {
    "type": "ener_spin"
  }
}
```

`use_spin` marks the magnetic types in `type_map` order. It also accepts element
names, for example `["Ni"]`, or type indices. Native spin requires
`descriptor.use_env_seed: true`, which is the default. The `spin` section can
also be added to a model built from a preset. A complete input is provided in
[the native-spin example](../../examples/spin/dpa4/input.json).

Supply moments when evaluating a model with magnetic types, including explicit
zero vectors for a zero-moment configuration. Moments on nonmagnetic types are
ignored. The magnetic force is the negative derivative of the energy with
respect to the input moment. Keep the dataset's moment units when preparing
inputs; `force_mag` has the corresponding energy-per-moment units. Native spin
does not use `virtual_scale`.

For training systems without a `spin` label, `spin.allow_missing_label: true`
fills the missing moments with zeros. Use this only when a zero moment is the
intended description of those systems.

A spin-free DPA4 checkpoint can initialize a spin model. Keep its architecture
and the same elements in the same `type_map` order, add the `spin` section,
change the loss to `ener_spin`, and fine-tune:

```bash
dp --pt-expt train input_spin.json --finetune pretrained.pt
```

The same procedure works with `--pt`. Initial predictions can differ from
the spin-free model, so train and validate the resulting spin model before
using it.

## Mixture of experts (MoE)

A mixture of experts (MoE) lets one DPA4.5 model use different weights for
different tasks, elements, and charge and spin states, where a model without
it uses one set of weights for all. A task is a branch of a multi-task model;
a single-task model is one task. Consider MoE when you train

- several data sets together as the branches of a multi-task model, for
  example data from different levels of theory;
- data that cover many elements, or molecules in several charge and spin
  states.

MoE is used in training. Before deployment you merge the model, which gives an
ordinary DPA4.5 model. Training takes longer with MoE, and it needs the
PyTorch Exportable backend.

To enable MoE, add an empty `moe` section to the descriptor, to the fitting
network, or to both, of a DPA4.5 preset:

```json
{
  "model": {
    "preset": "dpa4.5-mini-v20261111",
    "type_map": [
      "O",
      "H"
    ],
    "descriptor": {
      "moe": {}
    },
    "fitting_net": {
      "moe": {}
    }
  }
}
```

Every element of `type_map` gets element weights of its own, so the model
grows with the number of elements; the `element` option restricts them to a
list of elements. The other defaults suit most data; the
{ref}`argument reference <model[dpa4]/descriptor[dpa4]/moe>` lists the options.
A fitting network with `moe` takes no case embedding (`dim_case_embd`). See
the
[multi-task MoE example](../../examples/water/dpa4/input_dpa45_moe_multitask.json)
for a complete input.

### Merging an MoE model

After training, merge the model and freeze the merged checkpoint:

```bash
dp --pt-expt moe-merge model.ckpt.pt -o merged.pt
dp --pt-expt freeze -c merged.pt -o frozen_model.pt2
```

`moe-merge` takes the following options:

| Option               | Use                                                                                                                                                                                                                                                                                 |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--elements H C N O` | Keep the element weights of the listed elements; `--elements all` keeps those of every element. The other elements use the shared weights. Use it for an application whose systems contain only the listed elements: there the merged model gives the results of the trained model. |
| `--task NAME`        | Select the branch to merge (aliases: `--head`, `--model-branch`). Required for a multi-task checkpoint.                                                                                                                                                                             |
| `--charge-spin 0 1`  | Merge for this charge and spin multiplicity, which also become the default of the merged model. Without it, the model is merged for the default state of the trained model. It applies to a model with `moe` in the descriptor.                                                     |

A merged model matches the trained model in the charge and spin state it was
merged for; merge once for every state you need. The state becomes the
inference default of the merged model, but it does not move the reference of
the preset: an element whose table entry names no condition keeps the
condition it was referenced to when the model was built. Continue training
from the merged model as from any model; a fine-tuning task that defines its
own reference takes the one of its own input.

Without `--elements`, the merged model has no MoE and serves every element of
the type map. It runs at the speed of a model trained without MoE and is the
usual starting point for fine-tuning on your own data. A model that keeps
element weights is slightly slower.

A merged checkpoint holds the weights of the model. Freeze it, or start a
training from it with `--init-model` or `--finetune`. A model merged with
`--elements` keeps a `moe` section, which the input of that training repeats,
for example `"moe": {"n_experts": 0, "element": ["H", "C", "N", "O"]}`;
`--finetune` with `--use-pretrain-script` takes it from the checkpoint. See
[fine-tuning](../train/finetuning.md#new-tasks-of-an-moe-model) for adding
tasks to an MoE model.

### Merge-aware training (MAT)

The merge without `--elements` relies on merge-aware training (MAT): during
training, a share of the frames is computed the way that merged model computes
them, so that it is accurate when it is deployed.

MAT is on by default and needs no setting. The option `mat_prob` of the `moe`
section is the share of these frames, `0.37` by default. Set it to `0` only
for a model that will always keep its element weights: without MAT,
`moe-merge` refuses to merge without `--elements`.

## Export and inference

### Freeze to `.pt2`

Export with the backend used for training:

```bash
dp --pt-expt freeze -c model.ckpt.pt -o frozen_model.pt2
```

For a PyTorch checkpoint, use `dp --pt freeze` instead.

Either command produces a `.pt2` file for ASE or LAMMPS. For a multi-task model,
add `--task BRANCH_NAME` (aliases: `--head`, `--model-branch`) to select a branch. No extra export option is needed
for multi-rank LAMMPS. DPA4 does not support model compression.

Export on the device type that will run the model. For a CPU archive, hide
GPUs during export:

```bash
CUDA_VISIBLE_DEVICES="" dp --pt-expt freeze -c model.ckpt.pt -o frozen_cpu.pt2
```

### Inference precision and performance

GPU export defaults to `DP_TRITON_INFER=2`, with `DP_CUDA_INFER=2` for
`dp --pt-expt freeze` and `DP_CUDA_INFER=1` for `dp --pt freeze`. These settings
use optimized float32 operations. Keep `DP_TF32_INFER=0` and `DP_AMP_INFER=0`
when force smoothness matters, including molecular dynamics:

```bash
DP_TF32_INFER=0 DP_AMP_INFER=0 dp --pt-expt freeze -c model.ckpt.pt -o frozen_model.pt2
```

Set inference options before exporting. A `.pt2` file keeps the settings used
at export; changing the environment when ASE or LAMMPS loads it does not change
them. Export again to use different settings.

| Variable           | Use                                                                                                                                                                                                        |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DP_TRITON_INFER`  | Select GPU acceleration level `0`–`3`. Level `0` disables it; levels `1` and `2` retain float32 arithmetic. Level `3` uses lower-precision arithmetic in some operations; validate forces before using it. |
| `DP_CUDA_INFER`    | Select additional CUDA acceleration: `0` off, `1` on, `2` enables further operations whose performance depends on the model and GPU.                                                                       |
| `DP_TF32_INFER`    | Set matrix-multiplication precision: `0` highest, `1` high, `2` medium.                                                                                                                                    |
| `DP_AMP_INFER`     | Enable bfloat16 inference with `1`, independently of training AMP.                                                                                                                                         |
| `DP_COMPILE_INFER` | Enable compilation for direct PyTorch checkpoint inference with `1`. It is not needed to load a frozen `.pt2`.                                                                                             |

For direct checkpoint inference, GPU acceleration is off unless selected.
Alternative `DP_CUTILE_INFER` and `DP_CUTE_INFER` paths are available for Python
inference only; each is mutually exclusive with the other and with
`DP_TRITON_INFER`. They are not used in frozen `.pt2` files.

## Running in LAMMPS

LAMMPS needs the DeePMD-kit module; see
[installing LAMMPS](../install/install-lammps.md). For an energy model, enable
an atom map and use the `deepmd` pair style:

```lammps
atom_modify map yes
pair_style deepmd frozen_model.pt2
pair_coeff * * O H
```

The element names in `pair_coeff` map LAMMPS atom types to model elements.
See [the water example](../../examples/water/dpa4/lmp/) and
[the LAMMPS interface](../third-party/lammps-command.md) for complete inputs.

For a DPA4 spin model frozen with `dp --pt-expt freeze`, use `atom_style spin`
and the `dpa4spin` pair style:

```lammps
atom_style spin
atom_modify map yes
pair_style dpa4spin frozen_spin_model.pt2
pair_coeff * * Ni O
```

The pair style passes the magnetic moment of each atom to the model, which
uses only physical atoms. It runs on one MPI rank or several, in single or
double precision, on a GPU or a CPU; see
[its reference](../third-party/lammps-command.md#pair_style-dpa4spin). The
`deepspin` pair style runs the same models. Use it for a model frozen with
`dp --pt freeze`, or for a model with atomic parameters or with frame
parameters that have no default; see
[its reference](../third-party/lammps-command.md#pair_style-deepspin).

Several models on the `pair_style` line form an ensemble: the first one drives
the dynamics, and the deviation among all of them is written every `out_freq`
steps, with every pair style of this section; see
[the model deviation](../third-party/lammps-command.md#pair_style-deepmd).

### KOKKOS on a GPU

A single-precision model frozen with `dp --pt-expt freeze` on a GPU keeps the
neighbor list and every step on the GPU when LAMMPS runs with the KOKKOS
package on its GPU backend:

```bash
lmp -k on g 1 -sf kk -in in.lammps
```

`-sf kk` turns `pair_style deepmd` into `deepmd/kk` and, for a spin model,
`pair_style dpa4spin` into `dpa4spin/kk`. Zone bridging and multi-rank runs
work as with the host pair styles. A model frozen with `dp --pt freeze` runs
with the host pair styles.

### Multiple GPUs

Use one MPI rank per GPU and make all participating GPUs visible:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 lmp -in in.lammps
```

With KOKKOS, run `mpirun -np 4 lmp -k on g 4 -sf kk -in in.lammps` instead.
Both PyTorch backends export models that support multi-rank LAMMPS, including
native spin and zone bridging.
Use CUDA-aware MPI for GPU communication and a nonzero neighbor skin, for
example `neighbor 2.0 bin`. For PyTorch Exportable models, choose a domain
decomposition in which every rank has at least one local or ghost atom.
See [the MPI installation instructions](../install/install-lammps.md#mpi-for-runs-on-several-gpus).

## Embedding extraction

Use `dp embed` to extract the per-atom descriptor, the last fitting-layer
features, and their sum over each structure:

```bash
dp embed -m model.ckpt.pt -s /path/to/system -o embedding.hdf5
```

The command accepts a training checkpoint from either PyTorch backend or a `.pt2`
exported with `dp --pt-expt freeze`. For a model exported with `dp --pt freeze`,
use its training checkpoint instead. See
[model embeddings](../inference/embedding.md) for the output format and Python
interface.

## Citation

If you use DPA4/SeZM, cite the {ref}`DPA4 paper <cite-dpa4>`.
