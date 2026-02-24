---
name: add-descriptor
description: Guides through adding a new descriptor type to deepmd-kit. Covers implementing in dpmodel (array-API-compatible), wrapping for JAX/pt_expt backends, hard-coding for PT/PD, registering arguments, and writing all required tests.
license: LGPL-3.0-or-later
compatibility: Requires Python 3.10+, numpy, pytest. Optional backends for full testing (torch, jax, paddle).
metadata:
  author: deepmd-kit
  version: '2.0'
---

# Adding a New Descriptor to deepmd-kit

Follow these steps in order. Each step lists files to create/modify and patterns to follow.

## Step 1: Implement in dpmodel

**Create** `deepmd/dpmodel/descriptor/<name>.py`

Inherit from `NativeOP` and `BaseDescriptor`. Register with decorators:

```python
from deepmd.dpmodel import NativeOP
from .base_descriptor import BaseDescriptor


@BaseDescriptor.register("your_name")
@BaseDescriptor.register("alias_name")  # optional aliases
class DescrptYourName(NativeOP, BaseDescriptor): ...
```

Key requirements:

- `__init__`: initialize cutoff, sel, networks, davg/dstd statistics
- `call(coord_ext, atype_ext, nlist, mapping=None)`: forward pass returning `(descriptor, rot_mat, g2, h2, sw)`
- `serialize() -> dict`: save with `@class`, `type`, `@version`, `@variables` keys
- `deserialize(cls, data)`: reconstruct from dict
- Property/getter methods: `get_rcut`, `get_sel`, `get_dim_out`, `mixed_types`, etc.
- `__getitem__`/`__setitem__` for `davg`/`dstd` access via multiple key aliases

All dpmodel code **must** use `array_api_compat` for cross-backend compatibility (numpy/torch/jax/paddle). See [references/dpmodel-implementation.md](references/dpmodel-implementation.md) for full method table, array API pitfalls, and utilities.

**Reference implementations**:

- Simple: `deepmd/dpmodel/descriptor/se_e2_a.py`
- Three-body: `deepmd/dpmodel/descriptor/se_t.py`
- Attention-based: `deepmd/dpmodel/descriptor/dpa1.py`

## Step 2: Register

**Edit** `deepmd/dpmodel/descriptor/__init__.py` — add import and `__all__` entry.

**Edit** `deepmd/utils/argcheck.py` — register descriptor arguments:

```python
@descrpt_args_plugin.register("your_name", alias=["alias"], doc="Description")
def descrpt_your_name_args() -> list[Argument]:
    return [
        Argument("sel", [list[int], str], optional=True, default="auto", doc=doc_sel),
        Argument("rcut", float, optional=True, default=6.0, doc=doc_rcut),
        Argument("rcut_smth", float, optional=True, default=0.5, doc=doc_rcut_smth),
        Argument(
            "neuron", list[int], optional=True, default=[10, 20, 40], doc=doc_neuron
        ),
        # ... add all constructor parameters
    ]
```

## Step 3: Wrap for JAX backend

**Create** `deepmd/jax/descriptor/<name>.py`

Pattern: `@flax_module` decorator + custom `__setattr__` for attribute conversion.

```python
from deepmd.dpmodel.descriptor.your_name import DescrptYourName as DescrptYourNameDP
from deepmd.jax.common import ArrayAPIVariable, flax_module, to_jax_array
from deepmd.jax.descriptor.base_descriptor import BaseDescriptor
from deepmd.jax.utils.exclude_mask import PairExcludeMask
from deepmd.jax.utils.network import NetworkCollection


@BaseDescriptor.register("your_name")
@flax_module
class DescrptYourName(DescrptYourNameDP):
    def __setattr__(self, name, value):
        if name in {"davg", "dstd"}:
            value = to_jax_array(value)
            if value is not None:
                value = ArrayAPIVariable(value)
        elif name in {"embeddings"}:
            if value is not None:
                value = NetworkCollection.deserialize(value.serialize())
        elif name == "env_mat":
            pass  # stateless
        elif name == "emask":
            value = PairExcludeMask(value.ntypes, value.exclude_types)
        return super().__setattr__(name, value)
```

For nested sub-components, define wrapper classes bottom-up. See `deepmd/jax/descriptor/dpa1.py` for example.

**Edit** `deepmd/jax/descriptor/__init__.py` — add import and `__all__` entry.

## Step 4: Wrap for pt_expt backend

**Create** `deepmd/pt_expt/descriptor/<name>.py`

The `@torch_module` decorator handles everything automatically:

- Auto-generates `forward()` delegating to `call()` (and `forward_lower()` from `call_lower()`)
- Auto-generates `__setattr__` that converts numpy arrays to torch buffers and dpmodel objects to pt_expt modules via a converter registry
- Any unregistered `NativeOP` assigned as an attribute will raise `TypeError` — register it first

Simple descriptors (no custom sub-components) need only an empty body:

```python
from deepmd.dpmodel.descriptor.your_name import DescrptYourName as DescrptYourNameDP
from deepmd.pt_expt.common import torch_module
from deepmd.pt_expt.descriptor.base_descriptor import BaseDescriptor


@BaseDescriptor.register("your_name")
@torch_module
class DescrptYourName(DescrptYourNameDP):
    pass
```

Standard dpmodel sub-components (`NetworkCollection`, `EmbeddingNet`, `PairExcludeMask`, `EnvMat`, `TypeEmbedNet`) are pre-registered in `deepmd/pt_expt/utils/` and converted automatically. No `__setattr__` override needed.

For **custom sub-components** (e.g., a new block class inheriting `NativeOP`), create a separate wrapper file and register bottom-up with `register_dpmodel_mapping`:

```python
# deepmd/pt_expt/descriptor/your_block.py
from deepmd.dpmodel.descriptor.your_block import YourBlock as YourBlockDP
from deepmd.pt_expt.common import register_dpmodel_mapping, torch_module


@torch_module
class YourBlock(YourBlockDP):
    pass


register_dpmodel_mapping(
    YourBlockDP,
    lambda v: YourBlock.deserialize(v.serialize()),
)
```

Then import this module in `deepmd/pt_expt/descriptor/__init__.py` for its side effect (the registration must happen before the parent descriptor is instantiated).

Reference: `deepmd/pt_expt/descriptor/se_t_tebd.py` + `se_t_tebd_block.py`

**Edit** `deepmd/pt_expt/descriptor/__init__.py` — add import and `__all__` entry.

## Step 5: Hard-code for PT backend (if needed)

**Create** `deepmd/pt/model/descriptor/<name>.py`

PT descriptors are fully reimplemented in PyTorch (not wrapping dpmodel). They inherit from `BaseDescriptor` and `torch.nn.Module`. Must implement `forward()`, `serialize()`, `deserialize()`.

**Edit** `deepmd/pt/model/descriptor/__init__.py` — add import.

Reference: `deepmd/pt/model/descriptor/se_a.py`

## Step 6: Hard-code for TF backend (if needed)

**Create** `deepmd/tf/descriptor/<name>.py`

TF descriptors are fully reimplemented in TensorFlow. They inherit from `BaseDescriptor` and implement the TF computation graph.

**Edit** `deepmd/tf/descriptor/__init__.py` — add import.

Reference: `deepmd/tf/descriptor/se_a.py`

## Step 7: Hard-code for PD backend (if needed)

Same as PT but using Paddle. Inherit from `BaseDescriptor` and `paddle.nn.Layer`.

**Edit** `deepmd/pd/model/descriptor/__init__.py` — add import.

Reference: `deepmd/pd/model/descriptor/se_a.py`

## Step 8: Write tests

Eight test categories. See [references/test-patterns.md](references/test-patterns.md) for full code templates.

pt_expt tests use `pytest.mark.parametrize` (not `itertools.product`), do not inherit from `unittest.TestCase`, and use `setup_method` (not `setUp`).

| Test                  | File                                                           | Purpose                                           |
| --------------------- | -------------------------------------------------------------- | ------------------------------------------------- |
| 8a. dpmodel           | `source/tests/common/dpmodel/test_descriptor_<name>.py`        | Serialize/deserialize round-trip                  |
| 8b. pt_expt           | `source/tests/pt_expt/descriptor/test_<name>.py`               | Consistency + exportable + make_fx (float64 only) |
| 8c. PT                | `source/tests/pt/model/test_descriptor_<name>.py`              | PT hard-coded tests (if applicable)               |
| 8d. PD                | `source/tests/pd/model/test_descriptor_<name>.py`              | PD hard-coded tests (if applicable)               |
| 8e. array_api_strict  | `source/tests/array_api_strict/descriptor/<name>.py`           | Wrapper for consistency tests                     |
| 8f. Universal dpmodel | `source/tests/universal/dpmodel/descriptor/test_descriptor.py` | Add parametrized entry                            |
| 8g. Universal PT      | `source/tests/universal/pt/descriptor/test_descriptor.py`      | Add parametrized entry                            |
| 8h. Consistency       | `source/tests/consistent/descriptor/test_<name>.py`            | Cross-backend + API consistency                   |

## Step 9: Write documentation

**Create** `doc/model/<name>.md`

Each descriptor needs a documentation page in `doc/model/`. Use MyST Markdown format with Sphinx extensions. List supported backends using icon substitutions.

Template:

````markdown
# Descriptor `"your_name"` {{ pytorch_icon }} {{ dpmodel_icon }}

:::{note}
**Supported backends**: PyTorch {{ pytorch_icon }}, DP {{ dpmodel_icon }}
:::

Brief description of what the descriptor is and its theoretical motivation.

## Theory

Mathematical formulation using LaTeX:

```math
    \mathcal{D}^i = ...
```

## Instructions

Example JSON configuration:

```json
"descriptor": {
    "type": "your_name",
    "sel": [46, 92],
    "rcut_smth": 0.50,
    "rcut": 6.00,
    "neuron": [10, 20, 40],
    "resnet_dt": false,
    "seed": 1
}
```

Explain key parameters and link to the argument schema using `{ref}` directives,
e.g. `{ref}rcut <model[standard]/descriptor[your_name]/rcut>`.
````

Available backend icons: `{{ tensorflow_icon }}`, `{{ pytorch_icon }}`, `{{ jax_icon }}`, `{{ paddle_icon }}`, `{{ dpmodel_icon }}`. Only list backends that actually support this descriptor.

**Edit** `doc/model/index.rst` — add the new page to the `toctree`:

```rst
.. toctree::
   :maxdepth: 1

   ...
   <name>
```

**Reference docs**: `doc/model/train-se-e2-r.md` (simple), `doc/model/dpa2.md` (modern)

## Verification

```bash
# dpmodel self-consistency
python -m pytest source/tests/common/dpmodel/test_descriptor_<name>.py -v

# pt_expt unit tests
python -m pytest source/tests/pt_expt/descriptor/test_<name>.py -v

# Cross-backend consistency
python -m pytest source/tests/consistent/descriptor/test_<name>.py -v

# PT/PD unit tests (if hard-coded)
python -m pytest source/tests/pt/model/test_descriptor_<name>.py -v
python -m pytest source/tests/pd/model/test_descriptor_<name>.py -v

# Quick smoke test
python -c "
from deepmd.dpmodel.descriptor import DescrptYourName
d = DescrptYourName(rcut=6.0, rcut_smth=1.8, sel=[20, 20])
d2 = DescrptYourName.deserialize(d.serialize())
print('Round-trip OK:', d.get_dim_out() == d2.get_dim_out())
"
```

## Files summary

| Step | Action | File                                                           |
| ---- | ------ | -------------------------------------------------------------- |
| 1    | Create | `deepmd/dpmodel/descriptor/<name>.py`                          |
| 2    | Edit   | `deepmd/dpmodel/descriptor/__init__.py`                        |
| 2    | Edit   | `deepmd/utils/argcheck.py`                                     |
| 3    | Create | `deepmd/jax/descriptor/<name>.py`                              |
| 3    | Edit   | `deepmd/jax/descriptor/__init__.py`                            |
| 4    | Create | `deepmd/pt_expt/descriptor/<name>.py`                          |
| 4    | Edit   | `deepmd/pt_expt/descriptor/__init__.py`                        |
| 5    | Create | `deepmd/pt/model/descriptor/<name>.py` (if needed)             |
| 5    | Edit   | `deepmd/pt/model/descriptor/__init__.py` (if needed)           |
| 6    | Create | `deepmd/tf/descriptor/<name>.py` (if needed)                   |
| 6    | Edit   | `deepmd/tf/descriptor/__init__.py` (if needed)                 |
| 7    | Create | `deepmd/pd/model/descriptor/<name>.py` (if needed)             |
| 7    | Edit   | `deepmd/pd/model/descriptor/__init__.py` (if needed)           |
| 8a   | Create | `source/tests/common/dpmodel/test_descriptor_<name>.py`        |
| 8b   | Create | `source/tests/pt_expt/descriptor/test_<name>.py`               |
| 8c   | Create | `source/tests/pt/model/test_descriptor_<name>.py` (if PT)      |
| 8d   | Create | `source/tests/pd/model/test_descriptor_<name>.py` (if PD)      |
| 8e   | Create | `source/tests/array_api_strict/descriptor/<name>.py`           |
| 8e   | Edit   | `source/tests/array_api_strict/descriptor/__init__.py`         |
| 8f   | Edit   | `source/tests/universal/dpmodel/descriptor/test_descriptor.py` |
| 8g   | Edit   | `source/tests/universal/pt/descriptor/test_descriptor.py`      |
| 8h   | Create | `source/tests/consistent/descriptor/test_<name>.py`            |
| 9    | Create | `doc/model/<name>.md`                                          |
| 9    | Edit   | `doc/model/index.rst`                                          |
