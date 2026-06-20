# Model embeddings

A trained model can export learned representations ("embeddings") for downstream
analysis, such as clustering, visualization, or training auxiliary models. A
single forward pass produces the embeddings without computing forces or virials.

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, for DPA4/SeZM energy models only.
:::

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

where `-m` gives the PyTorch checkpoint (`.pt`), `-s` the path to the system
directory (or `-f` for a datafile listing system directories, one per line), and
`-o` the output HDF5 file. Reading from a multi-task model additionally accepts
`--head` to select the model branch.

Several other command line options can be passed to `dp embed`, which can be
checked with

```bash
$ dp embed --help
```

## Output format

The output is a single HDF5 file. Each system is stored as a group named after
the system directory, with the source directory recorded in the group's
`system` attribute. Each group holds the datasets `descriptor`,
`atomic_feature`, `structural_feature`, and `atom_types` (with shape
(nframes, natoms); the frame axis follows the system's frame order), together
with an `nframes` attribute. The model `type_map` is stored as a file-level
attribute. The three embedding datasets are stored as float32, and all datasets
use gzip and byte-shuffle compression.

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
