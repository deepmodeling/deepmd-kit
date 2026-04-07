---
name: debug-gradient-flow
description: Diagnose gradient flow issues in training, especially for compiled models (torch.compile/make_fx). Systematically isolates which loss components (energy, force, virial) contribute gradients to which parameters, and identifies where the gradient chain breaks.
license: LGPL-3.0-or-later
metadata:
  author: deepmd-kit
  version: '1.0'
---

# Debugging Gradient Flow in Training

Use this method when a loss component (force, virial, energy) does not decrease during training, or when compiled model training diverges from uncompiled training.

## When to use

- A loss term (e.g. `rmse_f`, `rmse_v`) stays flat or NaN during training
- Compiled training (`enable_compile=True`) behaves differently from uncompiled
- After adding a new loss component or model output
- After changes to `make_fx` tracing, `torch.compile`, or `autograd.grad` code paths

## Method: Per-component gradient isolation

The core technique: **zero out all loss terms except one**, run `loss.backward()`, and count which model parameters receive non-zero gradients. Compare across uncompiled and compiled paths to pinpoint where gradients are lost.

### Step 1: Write a gradient probe script

Create a script that constructs a trainer, injects labels if needed, and reports per-parameter gradient status:

```python
def check_grad(trainer, label_overrides=None):
    trainer.wrapper.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    inp, lab = trainer.get_data(is_train=True)
    lr = trainer.scheduler.get_last_lr()[0]

    # Override labels to isolate a single loss component
    if label_overrides:
        lab.update(label_overrides)

    _, loss, more_loss = trainer.wrapper(**inp, cur_lr=lr, label=lab)
    loss.backward()

    status = {}
    for name, p in trainer.wrapper.named_parameters():
        if p.requires_grad:
            has_grad = p.grad is not None and p.grad.abs().sum() > 0
            status[name] = has_grad
    return status
```

### Step 2: Run for each loss component in isolation

Test each loss component separately by zeroing out the others:

```python
scenarios = {
    "energy only": {"find_force": 0.0, "find_virial": 0.0},
    "force only": {"find_energy": 0.0, "find_virial": 0.0},
    "virial only": {
        "find_energy": 0.0,
        "find_force": 0.0,
        "virial": torch.randn(nframes, 9, ...),  # inject if data lacks virial
        "find_virial": 1.0,
    },
    "all losses": {
        "virial": torch.randn(nframes, 9, ...),
        "find_virial": 1.0,
    },
}
```

If training data lacks virial labels, inject synthetic ones — the numerical values don't matter, only gradient flow matters.

### Step 3: Compare compiled vs uncompiled

Run each scenario for both compiled and uncompiled trainers. Present results as a table:

```
                       Uncompiled  Compiled
energy only:           22/22       22/22
force only:            20/22       16/22    <-- problem
virial only:           20/22       16/22    <-- problem
all losses:            22/22       22/22    <-- OK in practice
```

Key interpretations:

- **Same count, both paths**: gradient flow is correct
- **Compiled < Uncompiled**: `make_fx` or `torch.compile` breaks some gradient paths
- **0 grads in compiled**: catastrophic failure (e.g. wrong `create_graph`, wrong backend)
- **"all losses" is OK but isolated isn't**: the missing grads are covered by other loss terms; may be acceptable

### Step 4: Identify affected parameters

When compiled has fewer grads, print the per-parameter diff:

```python
print(f"{'Parameter':<60} {'Uncompiled':>10} {'Compiled':>10}")
for name in sorted(status_uncompiled):
    uc = "GRAD" if status_uncompiled[name] else "-"
    cc = "GRAD" if status_compiled[name] else "-"
    marker = " <-- DIFF" if uc != cc else ""
    print(f"{name:<60} {uc:>10} {cc:>10}{marker}")
```

This tells you exactly which layers lose gradients and helps locate the broken link in the computation graph.

### Step 5: Bisect the cause

If compiled has fewer grads, test these layers in order:

| Layer                                            | What to try                                             | What it tests                                          |
| ------------------------------------------------ | ------------------------------------------------------- | ------------------------------------------------------ |
| `make_fx` only (no `torch.compile`)              | Replace `torch.compile(traced, ...)` with just `traced` | Is `make_fx` the problem or `torch.compile`?           |
| Different `torch.compile` backends               | Try `eager`, `aot_eager`, `inductor`                    | Which backend breaks gradients?                        |
| `model.train()` vs `model.eval()` during tracing | Toggle training mode before `make_fx`                   | Does `create_graph=self.training` get the wrong value? |
| `coord.requires_grad_(True)` placement           | Check if coord has grad before entering compiled graph  | Is the autograd entry point correct?                   |

```python
# Test make_fx only (no torch.compile)
traced = make_fx(fn)(ext_coord, ext_atype, nlist, mapping, fparam, aparam)
# Use traced directly instead of torch.compile(traced)

# Test different backends
for backend in ["eager", "aot_eager", "inductor"]:
    compiled = torch.compile(traced, backend=backend, dynamic=False)
    # ... run gradient check
```

## Common root causes

### 1. `create_graph=False` during tracing

**Symptom**: force/virial loss doesn't decrease; 0 params get grad from force/virial loss.

**Cause**: `model.eval()` before `make_fx` tracing makes `create_graph=self.training` evaluate to `False`. The `autograd.grad` that computes force is traced without graph creation, so the force tensor is detached from model parameters.

**Fix**: `model.train()` before `make_fx` tracing.

**Location**: `_trace_and_compile` in `deepmd/pt_expt/train/training.py`

### 2. `torch.compile` inductor backend kills second-order gradients

**Symptom**: force/virial loss doesn't decrease; 0 params get grad with inductor, but `eager`/`aot_eager` work fine.

**Cause**: The inductor backend's graph lowering doesn't support backward through `make_fx`-decomposed `autograd.grad` ops.

**Fix**: Default to `aot_eager` backend.

### 3. Ghost force contributions discarded

**Symptom**: force values differ between compiled and uncompiled models.

**Cause**: Using `extended_force[:, :nloc, :]` (slice) instead of scatter-summing ghost atom contributions back to local atoms via `mapping`.

**Fix**: `torch.zeros(...).scatter_add_(1, mapping_idx, extended_force[:, :actual_nall, :])`

### 4. Virial RMSE normalization mismatch

**Symptom**: `rmse_v` values differ between backends by a factor of `natoms`.

**Cause**: dpmodel `rmse_v = sqrt(l2_virial_loss)` missing `* atom_norm` normalization that other backends apply.

**Fix**: `rmse_v = sqrt(l2_virial_loss) * atom_norm`

## Verification

After fixing, always verify:

1. **Gradient count matches**: uncompiled and compiled should have the same number of params with grad for each isolated loss component
1. **Numerical consistency**: compiled model energy/force/virial should match uncompiled to float precision (`atol=1e-10, rtol=1e-10`)
1. **Loss decreases**: run a few training steps and verify `rmse_f` / `rmse_v` actually decrease
1. **Regression test**: add a test that catches the bug by reverting the fix and confirming the test fails

```bash
# Run compiled consistency test
python -m pytest source/tests/pt_expt/test_training.py::TestCompiledConsistency -v
# Run loss consistency test
python -m pytest source/tests/consistent/loss/test_ener.py -v
# Run full training smoke test
python -m pytest source/tests/pt_expt/test_training.py -v
```
