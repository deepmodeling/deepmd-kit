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

Both are weighted `0.01` by default, which is Uni-Mol's own recipe, so a
configuration that selects this fitting and leaves the loss alone asks for two
terms this backbone cannot produce. That is refused when the model and the loss
are wired together, naming the weights to zero -- so the two lines below are
required, not decorative:

```json
"loss": {
  "type": "unimol",
  "x_norm_loss": 0.0,
  "delta_pair_repr_norm_loss": 0.0,
  "virtual_tokens": false
}
```

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
the same numbers either way. `all_pairs` includes the diagonal, because upstream
scores the zero self-distance too; a neighbour list never lists an atom as its
own neighbour, so given a list that already holds every other atom the two
differ by exactly those self-pairs and nothing more.

## The `[MASK]` pseudo-element

A corrupted atom is carried as `[MASK]`, so the `type_map` declares it alongside
the elements and the backbone's type embedding gains a row for it. Two
consequences are worth knowing before pairing this with an existing model:

- A DPA4 model pretrained without `[MASK]` has a different `type_map`, and DPA4
  does not implement `change_type_map`, so it cannot be adapted to this one.
- Electronic-configuration type embedding would reject `[MASK]`, since it is
  not a real element. DPA4 has no such option, so the two cannot meet today;
  worth knowing if this objective is ever put on a backbone that does.

## Periodic frames

Refused. The backbone handles a cell; this objective does not. Its distance
target is a plain coordinate difference with no minimum-image convention, and
its coverage keeps only local neighbours, so a periodic frame would train
against labels that are wrong by several Angstrom -- some of them longer than
the cut-off -- rather than fail. The refusal is at the atomic model, so it holds
on the evaluation path too, not only when a box is passed to the model.

## Reproducibility of the coordinate head

On the torch backend the equivariant state the coordinate head reads is not
bit-reproducible between identical calls when the backbone runs in single
precision with `use_env_seed` on, which is the shipped default. The difference
measures around 7e-09 — one part in a hundred million, far below anything the
objective resolves — and the element and distance heads are unaffected, since
the scalar read-out is exact. It disappears in double precision or with
`use_env_seed` off. This is a property of the backbone rather than of the heads;
it is mentioned because comparing `coord_update` across two runs will otherwise
look like a bug.

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
