# Gradient Probe Script

Complete, copy-pasteable script for diagnosing gradient flow issues. Adapt `make_config` for the model/loss you are testing.

```python
"""Gradient flow diagnostic for deepmd-kit training.

Tests each loss component in isolation across compiled/uncompiled paths.
Prints per-parameter gradient status to identify exactly where gradients
are lost.

Usage:
    cd ~/research/deepmodeling/deepmd-kit/source
    python /tmp/gradient_probe.py
"""

import os
import tempfile
from collections import defaultdict

import torch

from deepmd.pt_expt.entrypoints.main import get_trainer
from deepmd.utils.argcheck import normalize
from deepmd.utils.compat import update_deepmd_input

# Adapt this path to your training data
EXAMPLE_DIR = os.path.join(
    os.path.expanduser("~"),
    "research/deepmodeling/deepmd-kit/source/examples/water",
)


def make_config(data_dir, enable_compile=False):
    """Build a minimal config for gradient probing.

    Adapt this function for the model architecture and loss you are testing.
    Key: enable all loss terms (e, f, v) so we can selectively zero them.
    """
    config = {
        "model": {
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "se_e2_a",
                "sel": [6, 12],
                "rcut_smth": 0.50,
                "rcut": 3.00,
                "neuron": [8, 16],
                "resnet_dt": False,
                "axis_neuron": 4,
                "type_one_side": True,
                "seed": 1,
            },
            "fitting_net": {
                "neuron": [16, 16],
                "resnet_dt": True,
                "seed": 1,
            },
            "data_stat_nbatch": 1,
        },
        "learning_rate": {
            "type": "exp",
            "decay_steps": 500,
            "start_lr": 0.001,
            "stop_lr": 3.51e-8,
        },
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1,
            "start_pref_f": 1000,
            "limit_pref_f": 1,
            "start_pref_v": 1.0,
            "limit_pref_v": 1.0,
        },
        "training": {
            "training_data": {
                "systems": [os.path.join(data_dir, "data_0")],
                "batch_size": 1,
            },
            "validation_data": {
                "systems": [os.path.join(data_dir, "data_3")],
                "batch_size": 1,
                "numb_btch": 1,
            },
            "numb_steps": 1,
            "seed": 10,
            "disp_file": "lcurve.out",
            "disp_freq": 5,
            "save_freq": 1,
        },
    }
    if enable_compile:
        config["training"]["enable_compile"] = True
    return config


def run_and_get_grads(trainer, label_overrides=None):
    """Forward + backward, return per-parameter gradient status."""
    trainer.wrapper.train()
    trainer.optimizer.zero_grad(set_to_none=True)
    inp, lab = trainer.get_data(is_train=True)
    lr = trainer.scheduler.get_last_lr()[0]

    if label_overrides:
        for k, v in label_overrides.items():
            if callable(v):
                lab[k] = v(inp, lab)
            else:
                lab[k] = v

    _, loss, more_loss = trainer.wrapper(**inp, cur_lr=lr, label=lab)
    loss.backward()

    status = {}
    for name, p in trainer.wrapper.named_parameters():
        if p.requires_grad:
            has_grad = p.grad is not None and p.grad.abs().sum().item() > 0
            status[name] = has_grad
    return status, loss.item()


def make_virial_injector(dtype, device):
    """Return a callable that creates synthetic virial labels."""

    def inject(inp, lab):
        nframes = inp["atype"].shape[0]
        return torch.randn(nframes, 9, dtype=dtype, device=device)

    return inject


def main():
    data_dir = os.path.join(EXAMPLE_DIR, "data")
    if not os.path.isdir(data_dir):
        print(f"Data not found: {data_dir}")
        return

    tmpdir = tempfile.mkdtemp(prefix="grad_probe_")
    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        # --- Phase 1: Summary table ---
        print("=" * 80)
        print("PHASE 1: Gradient count per loss component")
        print("=" * 80)

        # Get dtype/device from a quick trainer
        config_tmp = make_config(data_dir)
        config_tmp = update_deepmd_input(config_tmp, warning=False)
        config_tmp = normalize(config_tmp)
        trainer_tmp = get_trainer(config_tmp)
        inp_tmp, _ = trainer_tmp.get_data(is_train=True)
        dtype = inp_tmp["coord"].dtype
        device = inp_tmp["coord"].device
        del trainer_tmp

        scenarios = {
            "energy only": {"find_force": 0.0, "find_virial": 0.0},
            "force only": {"find_energy": 0.0, "find_virial": 0.0},
            "virial only": {
                "find_energy": 0.0,
                "find_force": 0.0,
                "virial": make_virial_injector(dtype, device),
                "find_virial": 1.0,
            },
            "all losses": {
                "virial": make_virial_injector(dtype, device),
                "find_virial": 1.0,
            },
        }

        all_results = {}  # (compile_mode, scenario) -> (status_dict, count, total)

        for compile_mode in ["uncompiled", "compiled"]:
            enable = compile_mode == "compiled"
            for scenario, overrides in scenarios.items():
                config = make_config(data_dir, enable_compile=enable)
                config = update_deepmd_input(config, warning=False)
                config = normalize(config)
                trainer = get_trainer(config)
                status, loss_val = run_and_get_grads(trainer, overrides)
                count = sum(1 for v in status.values() if v)
                total = len(status)
                key = (compile_mode, scenario)
                all_results[key] = (status, count, total)
                del trainer

        # Print summary table
        print(f"\n{'Scenario':<20} {'Uncompiled':>12} {'Compiled':>12} {'Match':>8}")
        print("-" * 56)
        for scenario in scenarios:
            _, uc_count, uc_total = all_results[("uncompiled", scenario)]
            _, cc_count, cc_total = all_results[("compiled", scenario)]
            match = "OK" if uc_count == cc_count else "DIFF"
            print(
                f"{scenario:<20} "
                f"{uc_count:>5}/{uc_total:<5} "
                f"{cc_count:>5}/{cc_total:<5} "
                f"{match:>8}"
            )

        # --- Phase 2: Per-parameter diff for mismatches ---
        print("\n" + "=" * 80)
        print("PHASE 2: Per-parameter diff (only for mismatching scenarios)")
        print("=" * 80)

        for scenario in scenarios:
            uc_status, uc_count, _ = all_results[("uncompiled", scenario)]
            cc_status, cc_count, _ = all_results[("compiled", scenario)]
            if uc_count == cc_count:
                continue
            print(f"\n--- {scenario} ---")
            print(f"{'Parameter':<60} {'Uncompiled':>10} {'Compiled':>10}")
            print("-" * 84)
            for name in sorted(uc_status.keys()):
                uc = "GRAD" if uc_status.get(name, False) else "-"
                cc = "GRAD" if cc_status.get(name, False) else "-"
                marker = " <-- DIFF" if uc != cc else ""
                print(f"{name:<60} {uc:>10} {cc:>10}{marker}")

        # --- Phase 3: torch.compile backend comparison ---
        print("\n" + "=" * 80)
        print("PHASE 3: torch.compile backend comparison (force-only loss)")
        print("=" * 80)

        from deepmd.pt_expt.train import training as training_mod

        orig_trace = training_mod._trace_and_compile

        for backend in ["eager", "aot_eager", "inductor"]:

            def patched(model, ec, ea, nl, mp, fp, ap, opts, _b=backend):
                opts["backend"] = _b
                return orig_trace(model, ec, ea, nl, mp, fp, ap, opts)

            training_mod._trace_and_compile = patched
            try:
                config = make_config(data_dir, enable_compile=True)
                config = update_deepmd_input(config, warning=False)
                config = normalize(config)
                trainer = get_trainer(config)
                status, _ = run_and_get_grads(
                    trainer, {"find_energy": 0.0, "find_virial": 0.0}
                )
                count = sum(1 for v in status.values() if v)
                total = len(status)
                print(f"  {backend:<12}: {count}/{total} params have force grad")
                del trainer
            except Exception as e:
                print(f"  {backend:<12}: FAILED ({e})")

        training_mod._trace_and_compile = orig_trace

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
```

## Adapting the script

### Different model architecture

Change `make_config` to use a different descriptor/fitting. The rest of the script works unchanged.

### Different loss type

Change `scenarios` to match the loss component keys. For example, for a dipole model:

```python
scenarios = {
    "dipole only": {"find_energy": 0.0},
    "energy only": {"find_dipole": 0.0},
    "all losses": {},
}
```

### Testing without `get_trainer`

If you need to test a standalone model without the full training infrastructure:

```python
model = get_model(config["model"])
model.train()

# Build input
coord = torch.randn(1, natoms, 3, requires_grad=True)
atype = torch.tensor([[0, 0, 1, 1, 1, 1]])
box = torch.eye(3).reshape(1, 9) * 10.0

# Forward
pred = model(coord, atype, box)

# Backward from a specific output
pred["force"].sum().backward()

for name, p in model.named_parameters():
    has_grad = p.grad is not None and p.grad.abs().sum() > 0
    print(f"{name}: {'GRAD' if has_grad else '-'}")
```
