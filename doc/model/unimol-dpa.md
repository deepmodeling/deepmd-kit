# Uni-Mol pretraining on a DPA backbone {{ pytorch_icon }} {{ dpmodel_icon }}

> [!NOTE]
> **Supported backends**: PyTorch-Exportable {{ pytorch_icon }}, DP {{ dpmodel_icon }}

Uni-Mol's self-supervised objective does not need Uni-Mol's transformer. This
runs the same three tasks -- recover the element of a corrupted atom, denoise
the coordinates, predict the clean pairwise distances -- on a DPA4 backbone, so
that molecular pretraining data can train the same descriptor that DFT data
trains.

See [the Uni-Mol backbone](unimol.md) for the objective itself: the corruption,
the five terms and their weights, and how the data is prepared. Only what the
heads read differs here.

## What the heads read

Uni-Mol's transformer carries a representation per atom pair and wraps each
molecule in two virtual tokens. DPA4 carries neither: it attends per edge with a
scatter softmax and returns no pair axis at all. So:

| Head       | Uni-Mol reads               | Here it reads                         |
| ---------- | --------------------------- | ------------------------------------- |
| element    | the per-atom representation | the same                              |
| coordinate | the pair channel            | the l=1 part of the equivariant state |
| distance   | the pair channel            | the two endpoints' representations    |

The coordinate head projects the backbone's equivariant state with a degree-wise
linear whose weights are shared across the three $m$ components, so its output
rotates with the molecule rather than merely being three numbers. The distance
head combines the two endpoints symmetrically, $[h_i + h_j \,\|\, h_i \odot
h_j]$, which makes the predicted matrix symmetric by construction.

Uni-Mol's two norm regularisers constrain quantities belonging to its own
transformer and have no counterpart here, so they carry no weight; and because
there are no virtual tokens, the objective is configured with
`virtual_tokens: false`.

## Which pairs the distance term covers

This is the one place the objective departs from Uni-Mol by construction, and it
is a choice rather than a limitation of the port.

`dist_coverage: neighbour` (default)
: Only pairs inside the backbone's neighbour list. It reuses the locality the
rest of deepmd trains on and costs `O(nloc * nnei)`. It also sees less than
Uni-Mol does: on drug-like molecules a 6 Å cut-off holds about half of all
pairs, and under a third for the largest.

`dist_coverage: all_pairs`
: Every pair, which is Uni-Mol's own coverage and what to use to reproduce its
training. It costs `O(nloc^2)` and does not share the backbone's neighbour
structure.

The setting selects which pairs are scored and nothing else: the head predicts
the same numbers either way. Given a neighbour list that already holds every
pair, the two agree exactly.

## The `[MASK]` pseudo-element

A corrupted atom is carried as `[MASK]`, so the `type_map` declares it alongside
the elements and the backbone's type embedding gains a row for it. Two
consequences are worth knowing before pairing this with an existing model:

- A DPA4 model pretrained without `[MASK]` has a different `type_map`, and DPA4
  does not implement `change_type_map`, so it cannot be adapted to this one.
- Electronic-configuration type embedding (`use_econf_tebd`) rejects any
  `type_map` entry that is not a real element, so it cannot be combined with
  this objective. It is off by default and DPA4 does not use it.

## Multi-task training

Not yet. Training one shared backbone from both a DFT branch and this objective
is the point of putting the objective on a DPA backbone, and the multi-task
machinery itself is ready for it -- but DPA4 does not implement `share_params`
on these backends, so the branches cannot be linked. Until it does, this
objective trains a backbone of its own.

## Training

```sh
dp --pt-expt train examples/unimol/dpa_pretrain/input.json
```

The dataset is the same LMDB one the Uni-Mol backbone uses, built the same way;
see [the Uni-Mol page](unimol.md).
