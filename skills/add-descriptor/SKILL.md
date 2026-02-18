---
name: add-descriptor
description: Guides through adding a new descriptor type to deepmd-kit. Covers implementing in dpmodel (array-API-compatible), wrapping for JAX/pt_expt backends, hard-coding for PT/PD, registering arguments, and writing all required tests.
license: LGPL-3.0-or-later
compatibility: Requires Python 3.10+, numpy, pytest. Optional backends for full testing (torch, jax, paddle).
metadata:
  author: deepmd-kit
  version: "1.0"
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

Pattern: `@torch_module` decorator + `forward()` method delegating to `call()`.

```python
from deepmd.dpmodel.descriptor.your_name import DescrptYourName as DescrptYourNameDP
from deepmd.pt_expt.common import torch_module
from deepmd.pt_expt.descriptor.base_descriptor import BaseDescriptor


@BaseDescriptor.register("your_name")
@torch_module
class DescrptYourName(DescrptYourNameDP):
    def forward(self, *args, **kwargs):
        return self.call(*args, **kwargs)
```

For nested sub-components, wrap and register bottom-up with `register_dpmodel_mapping`. See `deepmd/pt_expt/descriptor/se_t_tebd.py` + `se_t_tebd_block.py`.

**Edit** `deepmd/pt_expt/descriptor/__init__.py` — add import and `__all__` entry.

## Step 5: Hard-code for PT backend (if needed)

**Create** `deepmd/pt/model/descriptor/<name>.py`

PT descriptors are fully reimplemented in PyTorch (not wrapping dpmodel). They inherit from `BaseDescriptor` and `torch.nn.Module`. Must implement `forward()`, `serialize()`, `deserialize()`.

**Edit** `deepmd/pt/model/descriptor/__init__.py` — add import.

Reference: `deepmd/pt/model/descriptor/se_a.py`

## Step 6: Hard-code for PD backend (if needed)

Same as PT but using Paddle. Inherit from `BaseDescriptor` and `paddle.nn.Layer`.

**Edit** `deepmd/pd/model/descriptor/__init__.py` — add import.

Reference: `deepmd/pd/model/descriptor/se_a.py`

## Step 7: Write tests

Seven test categories. See [references/test-patterns.md](references/test-patterns.md) for full code templates.

| Test                  | File                                                           | Purpose                             |
| --------------------- | -------------------------------------------------------------- | ----------------------------------- |
| 7a. dpmodel           | `source/tests/common/dpmodel/test_descriptor_<name>.py`        | Serialize/deserialize round-trip    |
| 7b. pt_expt           | `source/tests/pt_expt/descriptor/test_<name>.py`               | Consistency + exportable + make_fx  |
| 7c. PT                | `source/tests/pt/model/test_descriptor_<name>.py`              | PT hard-coded tests (if applicable) |
| 7d. PD                | `source/tests/pd/model/test_descriptor_<name>.py`              | PD hard-coded tests (if applicable) |
| 7e. array_api_strict  | `source/tests/array_api_strict/descriptor/<name>.py`           | Wrapper for consistency tests       |
| 7f. Universal dpmodel | `source/tests/universal/dpmodel/descriptor/test_descriptor.py` | Add parameterized entry             |
| 7g. Universal PT      | `source/tests/universal/pt/descriptor/test_descriptor.py`      | Add parameterized entry             |
| 7h. Consistency       | `source/tests/consistent/descriptor/test_<name>.py`            | Cross-backend comparison            |

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
| 6    | Create | `deepmd/pd/model/descriptor/<name>.py` (if needed)             |
| 6    | Edit   | `deepmd/pd/model/descriptor/__init__.py` (if needed)           |
| 7a   | Create | `source/tests/common/dpmodel/test_descriptor_<name>.py`        |
| 7b   | Create | `source/tests/pt_expt/descriptor/test_<name>.py`               |
| 7c   | Create | `source/tests/pt/model/test_descriptor_<name>.py` (if PT)      |
| 7d   | Create | `source/tests/pd/model/test_descriptor_<name>.py` (if PD)      |
| 7e   | Create | `source/tests/array_api_strict/descriptor/<name>.py`           |
| 7e   | Edit   | `source/tests/array_api_strict/descriptor/__init__.py`         |
| 7f   | Edit   | `source/tests/universal/dpmodel/descriptor/test_descriptor.py` |
| 7g   | Edit   | `source/tests/universal/pt/descriptor/test_descriptor.py`      |
| 7h   | Create | `source/tests/consistent/descriptor/test_<name>.py`            |
