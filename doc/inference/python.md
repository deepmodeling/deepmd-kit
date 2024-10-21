# Python interface

:::{note}
See [Environment variables](../env.md) for the runtime environment variables.
:::

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

## External neighbor list algorithm {{ tensorflow_icon }}

:::{note}
**Supported backends**: TensorFlow {{ tensorflow_icon }}
:::

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
