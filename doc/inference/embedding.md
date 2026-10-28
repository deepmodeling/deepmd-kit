# Model embeddings

A trained model can export learned representations ("embeddings") for downstream
analysis, such as clustering, visualization, or training auxiliary models. A
single forward pass produces the embeddings without computing forces or virials.

> [!NOTE]
> **Supported backends**: PyTorch {{ pytorch_icon }} and PyTorch Exportable
> {{ pytorch_icon }}. The PyTorch backend serves every descriptor-fitting model,
> including DPA4/SeZM and DP+ZBL or other linear combinations. The PyTorch
> Exportable backend serves models whose descriptor has a graph form, such as
> DPA1, DPA2, DPA4 and DPA4C, including ZBL-bridged DPA4 and DPA4C. A combination
> reports the embeddings of its descriptor-fitting sub-model. Both backends serve
> energy, dipole, polarizability, dos and property models, though the
> `structural_feature` is only physically meaningful for energy models. Spin
> models are not supported.

Three embeddings are produced for each frame:

- `descriptor`: the per-atom local-environment representation, with shape
  (nframes, natoms, dim_descriptor).
- `atomic_feature`: the per-atom activation after the last fitting hidden layer
  (before the final output projection), with shape (nframes, natoms, dim_hidden).
- `structural_feature`: a per-structure summary obtained by summing
  `atomic_feature` over the atoms of each frame, with shape
  (nframes, dim_hidden).

## Command line

The embeddings of a model can be evaluated and saved using `dp embed`. A
typical usage is

```bash
dp embed -m model.ckpt.pt -s /path/to/system -o embedding.hdf5
```

where `-m` gives the model, `-s` the path to the system directory (or `-f` for a
datafile listing system directories, one per line), and `-o` the output HDF5
file. The model is a training checkpoint (`.pt`) or a frozen model: a `.pth`
written by `dp --pt freeze`, or a `.pt2` or `.pte` written by
`dp --pt-expt freeze`. The `.pt2` that `dp --pt freeze` writes for DPA4/SeZM is
not supported; read the training checkpoint of such a model instead. Use
`--dtype` to choose the output precision (`fp32`, `fp64`, or `native`; default
`fp32`). Reading from a multi-task model additionally accepts `--head` to select
the model branch.

A model that takes frame parameters, atomic parameters, or the total charge and
spin multiplicity of each frame reads them from the `fparam.npy`, `aparam.npy`,
and `charge_spin.npy` files in the sets of the system, as `dp test` does. The
first two follow the [system format](../data/system.md); `charge_spin.npy` holds
one `[charge, multiplicity]` pair per frame, with shape (nframes, 2). The atomic
parameters are always required. The frame parameters and the charge state may
be left out when the model defines a default (`default_fparam` or
`default_chg_spin`), which then applies to every frame. A compressed model
serves one charge state at a time, so all frames of a system must share it.

Several other command line options can be passed to `dp embed`, which can be
checked with

```bash
dp embed --help
```

## Output format

The output is a single HDF5 file. Each system is stored as a group named after
the system directory, with the source directory recorded in the group's
`system` attribute. Each group holds the datasets `descriptor`,
`atomic_feature`, `structural_feature`, and `atom_types` (with shape
(nframes, natoms); the frame axis follows the system's frame order), together
with an `nframes` attribute. In a system whose frames are padded to a common
atom count, a padding slot has type -1 in `atom_types`, and `structural_feature`
leaves it out; its rows of `descriptor` and `atomic_feature` carry no meaning,
so select the real atoms with `atom_types >= 0`.
The model `type_map` is stored as a file-level attribute. The three embedding
datasets are stored using the selected output dtype, and all datasets use gzip
and byte-shuffle compression.

The file can be read back with `h5py`:

```python
import h5py

with h5py.File("embedding.hdf5", "r") as f:
    type_map = f.attrs["type_map"]
    for system_name in f:
        group = f[system_name]
        source = group.attrs["system"]
        descriptor = group["descriptor"][:]
        atomic_feature = group["atomic_feature"][:]
        structural_feature = group["structural_feature"][:]
```

## Python interface

The same embeddings are available from the Python inference interface:

```python
from deepmd.infer import DeepPot
import numpy as np

dp = DeepPot("model.ckpt.pt")
coord = np.array([[1, 0, 0], [0, 0, 1.5], [1, 0, 3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1, 0, 1]
descriptor, atomic_feature, structural_feature = dp.eval_embedding(coord, cell, atype)
```

The embeddings are returned as float32 by default (both from the Python interface
and the `dp embed` command), which is ample for downstream analysis. Pass
`dtype="fp64"` or `dtype="native"` to {meth}`DeepPot.eval_embedding` (or
`--dtype fp64/native` to `dp embed`) when a different output precision is needed.

A model conditioned on the total charge and spin multiplicity of each frame takes
them through `charge_spin`, one `[charge, multiplicity]` pair per frame with shape
(nframes, 2); without the argument it uses its `default_chg_spin`. A compressed
model serves one condition at a time, so all frames of a call must share the same
pair.
