# Python interface

> [!NOTE]
> See [Environment variables](../env.md) for the runtime environment variables.

One may use the python interface of DeePMD-kit for model inference, an example is given as follows

```python
from deepmd.infer import DeepPot
import numpy as np

dp = DeepPot("graph.pb")
coord = np.array([[1, 0, 0], [0, 0, 1.5], [1, 0, 3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1, 0, 1]
e, f, v = dp.eval(coord, cell, atype)
```

where `e`, `f` and `v` are predicted energy, force and virial of the system, respectively.

One can also evaluate the descriptors of the model:

```python
from deepmd.infer import DeepPot
import numpy as np

dp = DeepPot("graph.pb")
coord = np.array([[1, 0, 0], [0, 0, 1.5], [1, 0, 3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1, 0, 1]
descriptors = dp.eval_descriptor(coord, cell, atype)
```

where `descriptors` is the descriptor matrix of the system. This can also be done using the command line interface `dp eval-desc` as described in the [test documentation](../test/test.md).

> [!NOTE]
> `eval_descriptor` is the descriptor-only interface supported across backends.
> The PyTorch and PyTorch Exportable backends additionally provide
> [`eval_embedding`](embedding.md), which returns the descriptor, per-atom feature,
> and per-structure feature in a single forward pass. Both interfaces accept
> `dtype="fp32"`, `"fp64"`, or `"native"`; `eval_descriptor` defaults to `native`,
> while `eval_embedding` defaults to `fp32`.

Furthermore, one can use the python interface to calculate model deviation.

```python
from deepmd.infer import calc_model_devi
from deepmd.infer import DeepPot as DP
import numpy as np

coord = np.array([[1, 0, 0], [0, 0, 1.5], [1, 0, 3]]).reshape([1, -1])
cell = np.diag(10 * np.ones(3)).reshape([1, -1])
atype = [1, 0, 1]
graphs = [DP("graph.000.pb"), DP("graph.001.pb")]
model_devi = calc_model_devi(coord, cell, atype, graphs)
```

Note that if the model inference or model deviation is performed cyclically, one should avoid calling the same model multiple times.
Otherwise, TensorFlow or PyTorch will never release the memory, and this may lead to an out-of-memory (OOM) error.

## Choosing a neighbor search

### External neighbor list {{ tensorflow_icon }}

> [!NOTE]
> **Supported backends**: TensorFlow {{ tensorflow_icon }}

The native neighbor list algorithm of the DeePMD-kit is in $O(N^2)$ complexity ($N$ is the number of atoms).
While this is not a problem for small systems that quantum methods can afford, the large systems for molecular dynamics have slow performance.
In this case, one may pass an external neighbor list that has lower complexity to {class}`DeepPot <deepmd.infer.DeepPot>`, once it is compatible with {class}`ase.neighborlist.NewPrimitiveNeighborList`.

```py
import ase.neighborlist

neighbor_list = ase.neighborlist.NewPrimitiveNeighborList(
    cutoffs=6, bothways=True, self_interaction=False
)
dp = DeepPot("graph.pb", neighbor_list=neighbor_list)
```

The `update` and `build` methods will be called by {class}`DeepPot <deepmd.infer.DeepPot>`, and `first_neigh`, `pair_second`, and `offset_vec` properties will be used.

### PyTorch Exportable neighbor lists {{ pytorch_icon }}

For a frozen `.pt2` or `.pte` model that takes a neighbor list, both
`DeepPot` and the ASE calculator accept `nlist_backend`. The default,
`"auto"`, selects an available search for the input batch. For a single
frame it tries Vesin, then NV, then the native search. For multiple frames it
tries NV, then Vesin, then the native search. Virtual-atom spin models with
the neighbor-list lower accept `"auto"` and `"native"` and use the native
search for both. Native-spin edge-vector models can also use Vesin or NV.

Without an optional search library,
the native extended-coordinate search has quadratic cost in the number of
atoms (including periodic images).

| `nlist_backend` | Neighbor search                                                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `"auto"`        | Select by availability and frame count.                                                                                                                                                     |
| `"native"`      | Use the native extended-coordinate builder for neighbor-list inputs.                                                                                                                        |
| `"vesin"`       | Use Vesin's cell-list search, which scales as $O(N)$ at fixed density and cutoff. Install `vesin[torch]>=0.6`. Explicitly requesting Vesin without the Torch bindings raises `ImportError`. |
| `"nv"`          | Use the CPU or CUDA search provided by `nvalchemi-toolkit-ops`.                                                                                                                             |

```py
from deepmd.calculator import DP
from deepmd.infer import DeepPot

dp = DeepPot("model.pt2", nlist_backend="vesin")
atoms.calc = DP(model="model.pt2", nlist_backend="vesin")
```

Here `atoms` is an ASE `Atoms` object. For graph-form models, including
compressed canonical DPA4C, choose the builder with
`neighbor_graph_method` instead. Passing a non-default `nlist_backend`
to these models raises `ValueError`. Edge-vector models accept either
option to request Vesin or NV. Set only one of them.
