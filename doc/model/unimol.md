# Descriptor Uni-Mol {{ pytorch_icon }} {{ dpmodel_icon }}

> [!NOTE]
> **Supported backends**: PyTorch-Exportable {{ pytorch_icon }}, DP {{ dpmodel_icon }}

Uni-Mol is a molecular representation model: a transformer over all atom pairs,
in which geometry enters only through pairwise distances. It was pretrained on
about 209 million RDKit conformers with three self-supervised objectives and no
energies or forces at all.

This is a port of Uni-Mol v1, faithful enough to load the released weights and
to reproduce the published objective. It exists for two reasons: to make
Uni-Mol's data and objectives available to multi-task training alongside
DFT-labelled data, and to give molecular property work a pretrained backbone.

> [!IMPORTANT]
> Uni-Mol is **not** a potential energy surface model. It attends over every
> atom pair with no cut-off and no smooth envelope, so it is not extensive, it
> does not support periodic boundaries, and its forces are neither smooth nor
> conserved. The descriptor refuses any frame whose atoms are not all local,
> which rules out periodic images and the ghost-atom layout that freezing and
> parallel evaluation assume, so it is not available for molecular dynamics or
> frozen deployment.

## Architecture

The backbone is a 15-layer pre-layer-norm transformer of width 512 with 64
attention heads. Distances are expanded in 128 Gaussians whose affine
parameters depend on the ordered element pair, projected to one bias per head,
and added to the attention logits. Each layer's pre-softmax logits become the
next layer's bias, so the running sum of logits is a pair representation that
the pretraining heads read. Two virtual tokens wrap each molecule.

Uni-Mol's own 31-token vocabulary is kept, because the released weights are
indexed by it: four special tokens, 26 elements, then `[MASK]`. A `type_map` is
mapped onto those ids, and an element outside the vocabulary becomes `[UNK]`.

## Pretraining objective

Fifteen percent of the atoms are selected. Of those, 90% become `[MASK]`, 5%
become a random element and 5% are left alone; all three are predicted. The
masked and randomly replaced atoms also have uniform noise of ±1 Å added to
each coordinate component. Five terms are minimized:

| Term | Weight | What it predicts |
| --- | --- | --- |
| element | 1 | the true element of every selected atom |
| coordinate | 5 | the clean coordinates, through the pair channel |
| distance | 10 | the clean pairwise distances |
| node norm | 0.01 | keeps node norms near $\sqrt{512}$ |
| pair-delta norm | 0.01 | keeps pair-delta norms near $\sqrt{64}$ |

Corruption happens in the data pipeline rather than inside the loss, which is
both what upstream does and what the PyTorch-Exportable backend requires, since
it runs the model before the loss sees a frame. The objective owns the settings
and hands the trainer the transform it needs, so a training run installs it
automatically; the masking rate, the 90/5/5 split, the noise and the seed are
all configurable under `loss`.

A masked atom is carried as a `[MASK]` pseudo-element, so the model's
`type_map` has to declare it alongside the elements.

## Training

```sh
dp --pt-expt train examples/unimol/pretrain/input.json
```

The dataset has to be an LMDB one, because the corruption happens as frames are
read; `deepmd.utils.unimol_data` below produces it. Give its path as a string
under `systems`, not as a list, which is how LMDB datasets are addressed.

## Using the released weights

The released checkpoints, `mol_pre_all_h_220816.pt` and
`mol_pre_no_h_220816.pt`, hold weights only and load with `weights_only=True`:

```py
from deepmd.utils.unimol_checkpoint import descriptor_from_unimol_checkpoint

descriptor = descriptor_from_unimol_checkpoint("mol_pre_all_h_220816.pt")
```

Parameter names line up one to one with the ported modules, but the arrays do
not: deepmd stores a linear weight as `(num_in, num_out)` and applies it as
`x @ w`, the transpose of `torch.nn.Linear.weight`. The converter renames and
transposes every weight, so there is no "just add a prefix" path.

## Converting the pretraining data

```sh
python -m deepmd.utils.unimol_data ligands.lmdb ./unimol_train --add-2d-conformer
```

One conformer becomes one frame, so ordinary frame sampling stands in for
upstream's per-epoch conformer draw. The two-dimensional RDKit conformer that
upstream appends while loading is added at conversion time, behind that flag,
so the training data path never needs RDKit.

## How closely this matches upstream

Every component is checked against tensors dumped from upstream running
unmodified on the same inputs. The data-side transforms agree bitwise, which
means the random stream itself is reproduced, down to which atoms are masked
and what noise each one receives. The encoder, fed upstream's own attention
bias, agrees to fp64 rounding.

What remains is upstream's own use of fp32 in three places, which sets the
floor at about 1e-7 relative on the whole objective:

- the Gaussian basis is evaluated in fp32, reproduced by default and
  switchable with `single_precision_basis`;
- the distance matrix is precomputed in fp32 by upstream's data pipeline,
  while the descriptor computes distances in the working precision, which is
  more accurate and is what gradients flow through; `single_precision_distance`
  reproduces upstream's numbers instead;
- `log_softmax` and both norm regularisers are evaluated in fp32, which is
  reproduced.

Training trajectories cannot be reproduced exactly in any case: upstream
pretrained a pure fp16 model with fused kernels and its own Adam variant, which
places epsilon differently from PyTorch's. The example carries upstream's
optimizer values, including `adam_eps`, so the recipe matches even though the
trajectory cannot.

Uni-Mol uses the exact error-function GELU, available here as `gelu_erf`.
deepmd's `gelu` and `gelu_tf` are the tanh approximation, which differs by up
to 4.7e-4 per element.

## Attribution

The ported code follows Uni-Mol (commit `90f52c4`) and the Uni-Core modules it
builds on (commit `ace6fae`), both MIT licensed, Copyright (c) DP Technology.
Parts of Uni-Core derive in turn from fairseq, Copyright (c) Facebook, Inc. and
its affiliates, also MIT licensed. Each ported file records its provenance.
