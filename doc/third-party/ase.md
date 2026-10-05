# Use deep potential with ASE

> [!NOTE]
> See [Environment variables](../env.md) for the runtime environment variables.

Deep potential can be set up as a calculator with ASE to obtain potential energies and forces.

::::{tab-set}

:::{tab-item} TensorFlow {{ tensorflow_icon }}

```python
from ase import Atoms
from deepmd.calculator import DP

water = Atoms(
    "H2O",
    positions=[(0.7601, 1.9270, 1), (1.9575, 1, 1), (1.0, 1.0, 1.0)],
    cell=[100, 100, 100],
    calculator=DP(model="frozen_model.pb"),
)
print(water.get_potential_energy())
print(water.get_forces())
```
:::

:::{tab-item} PyTorch-TorchScript {{ pytorch_icon }}

```python
from ase import Atoms
from deepmd.calculator import DP

water = Atoms(
    "H2O",
    positions=[(0.7601, 1.9270, 1), (1.9575, 1, 1), (1.0, 1.0, 1.0)],
    cell=[100, 100, 100],
    calculator=DP(model="frozen_model.pth"),
)
print(water.get_potential_energy())
print(water.get_forces())
```
:::

:::{tab-item} Paddle {{ paddle_icon }}

```python
from ase import Atoms
from deepmd.calculator import DP

water = Atoms(
    "H2O",
    positions=[(0.7601, 1.9270, 1), (1.9575, 1, 1), (1.0, 1.0, 1.0)],
    cell=[100, 100, 100],
    calculator=DP(model="frozen_model.json"),
)
print(water.get_potential_energy())
print(water.get_forces())
```
:::

::::

For PT DPA4/SeZM checkpoints and graph-eligible PT-expt checkpoints, enable
automatic freezing for production inference:

```python
from deepmd.calculator import DP

calculator = DP(model="model.pt", freeze=True)
```

This exports `model-ase_single.pt2` beside the checkpoint and loads it through
the compiled inference backend. Each calculator construction with `freeze=True`
rebuilds the artifact using the selected checkpoint head and the backend's
normal freeze policy. The export omits the multi-rank communication artifact
because ASE inference runs in one process. For CUDA targets, DPA4 defaults to
`DP_CUDA_INFER=1` and `DP_TRITON_INFER=2`, with explicit environment settings
taking precedence; DPA4C keeps its required CUDA level of 2.

DPA4C checkpoints also enable compression through the same export path as
`dp --pt-expt freeze --compress`. Supported models use the dedicated CPU or
CUDA compression kernels. If the configuration cannot be compressed, such as
a `float64` descriptor or one with type exclusions, ASE logs the reason and
continues with an ordinary frozen model. Other model families keep their
uncompressed freeze behavior. Explicit `freeze --compress` reports an error
for an unsupported compression configuration.

The default `freeze=False` loads checkpoints directly and logs a warning for
these models: inference can be **substantially slower**. This mode is intended
for debugging and short runs that avoid compilation overhead. To reuse an
export without compiling again, pass the `.pt2` file directly. Already frozen
models, other PT models, and PT-expt models without graph eligibility ignore
`freeze` and keep their normal loading path.

Optimization is also available:

```python
from ase.optimize import BFGS

dyn = BFGS(water)
dyn.run(fmax=1e-6)
print(water.get_positions())
```
