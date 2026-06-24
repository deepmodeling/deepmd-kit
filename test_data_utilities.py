#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Test dpa_adapt data utilities with QM9 demo dataset (8 entries)."""

import os
import sys
import tempfile
from pathlib import (
    Path,
)

# Ensure the *installed* deepmd-kit (with C extensions) is used instead of
# the source checkout when running from the project root.
_site_pkg = [p for p in sys.path if "site-packages" in p]
_other = [p for p in sys.path if "site-packages" not in p]
sys.path = _site_pkg + _other

import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent
DEMO_DIR = REPO_DIR / "examples" / "dpa_adapt" / "data"
TRAIN_DIR = DEMO_DIR / "train"
TEST_DIR = DEMO_DIR / "test"
TRAIN_GLOB = str(TRAIN_DIR / "sys_*")
TEST_GLOB = str(TEST_DIR / "sys_*")
PRETRAINED = os.environ.get("DPA_ADAPT_PRETRAINED", "DPA-3.1-3M")
N_TRAIN = 5
N_TEST = 3
N_TOTAL = N_TRAIN + N_TEST

# check that demo data exists
assert TRAIN_DIR.is_dir(), f"missing {TRAIN_DIR}"
assert TEST_DIR.is_dir(), f"missing {TEST_DIR}"

passed = 0
failed = 0


def check(description, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {description}")
    else:
        failed += 1
        print(f"  ✗ FAIL: {description}")


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def run_cli(args):
    """Run a dpa-adapt CLI command via sys.executable."""
    import subprocess as _sp

    code = (
        "import sys; "
        "_sp = [p for p in sys.path if 'site-packages' in p]; "
        "_ot = [p for p in sys.path if 'site-packages' not in p]; "
        "sys.path = _sp + _ot; "
        "from dpa_adapt.cli import main; "
        "sys.argv[:] = ['dpaad'] + " + repr(args) + "; "
        "main()"
    )
    return _sp.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. check_data() / dpaad data validate
# ═══════════════════════════════════════════════════════════════════════════
section("1. check_data() / dpaad data validate")

from dpa_adapt.data.loader import (
    load_data,
)
from dpa_adapt.data.validate import (
    check_data,
)

# 1a ── Python API: check_data() on training data ─────────────────────────
print("\n--- 1a. Python API: check_data() on training data ---")
train_systems = load_data(TRAIN_GLOB)
print(f"  Loaded {len(train_systems)} training systems")
check("load_data() returns 5 training systems", len(train_systems) == N_TRAIN)

issues = check_data(train_systems)
n_err = sum(1 for i in issues if i.severity == "error")
n_warn = sum(1 for i in issues if i.severity == "warn")
print(f"  Issues: {len(issues)} ({n_err} errors, {n_warn} warnings)")
check("check_data() on training data returns no errors", n_err == 0)

# 1b ── Python API: check_data() on test data ─────────────────────────────
print("\n--- 1b. Python API: check_data() on test data ---")
test_systems = load_data(TEST_GLOB)
print(f"  Loaded {len(test_systems)} test systems")
check("load_data() returns 3 test systems", len(test_systems) == N_TEST)

issues = check_data(test_systems)
n_err = sum(1 for i in issues if i.severity == "error")
print(f"  Issues: {len(issues)} ({n_err} errors)")
check("check_data() on test data returns no errors", n_err == 0)

# 1c ── Python API: check_data() on all 8 systems ──────────────────────────
print("\n--- 1c. Python API: check_data() on all 8 systems ---")
all_systems = load_data([TRAIN_GLOB, TEST_GLOB])
print(f"  Loaded {len(all_systems)} total systems")
check("load_data() returns 8 total systems", len(all_systems) == N_TOTAL)

issues = check_data(all_systems)
n_err = sum(1 for i in issues if i.severity == "error")
check("check_data() on all 8 systems returns no errors", n_err == 0)

# 1d ── CLI: dpaad data validate ──────────────────────────────────────────
print("\n--- 1d. CLI: dpaad data validate ---")
result = run_cli(["data", "validate", "--data", TRAIN_GLOB])
print(f"  stdout: {result.stdout.strip()}")
check("CLI data validate exit code 0", result.returncode == 0)
check("CLI output contains 'clean'", "clean" in result.stdout.lower())

# ═══════════════════════════════════════════════════════════════════════════
# 2. attach_labels() / CLI attach labels
# ═══════════════════════════════════════════════════════════════════════════
section("2. attach_labels() / CLI attach labels")

from dpa_adapt.data.convert import (
    attach_labels,
)

# 2a ── Python API: attach_labels(string head) on single system ──────────
print("\n--- 2a. Python API: attach_labels(string head) ---")
sys0_path = str(TRAIN_DIR / "sys_0000")
print(f"  Target: {sys0_path}")

# Attach a scalar label with a string head (writes set.000/bandgap.npy)
attach_labels(sys0_path, head="bandgap", values=np.array([13.74]))
written = np.load(TRAIN_DIR / "sys_0000" / "set.000" / "bandgap.npy")
check("'bandgap.npy' written to set.000/", written.shape == (1,))
check("bandgap value matches", np.isclose(written[0], 13.74))

# 2b ── Python API: attach_labels with dict head ─────────────────────────
print("\n--- 2b. Python API: attach_labels(dict head) ---")
sys1_path = str(TRAIN_DIR / "sys_0001")
attach_labels(
    sys1_path,
    head={"type": "property", "property_name": "my_prop", "task_dim": 1},
    values=np.array([[5.0]]),
)
written = np.load(TRAIN_DIR / "sys_0001" / "set.000" / "my_prop.npy")
check("dict-head 'my_prop.npy' written", written.shape == (1, 1))
check("my_prop value matches", np.isclose(written[0, 0], 5.0))

# 2c ── Python API: idempotent overwrite ─────────────────────────────────
print("\n--- 2c. Python API: idempotent overwrite ---")
attach_labels(sys0_path, head="bandgap", values=np.array([99.99]))
written = np.load(TRAIN_DIR / "sys_0000" / "set.000" / "bandgap.npy")
check("overwrite: bandgap updated", np.isclose(written[0], 99.99))

# 2d ── Python API: frame count mismatch raises ──────────────────────────
print("\n--- 2d. Python API: frame count mismatch ---")
try:
    attach_labels(sys0_path, head="bad_label", values=np.array([1.0, 2.0, 3.0]))
    check("ValueError raised on frame count mismatch", False)
except ValueError as e:
    check("ValueError raised on frame count mismatch", "frames" in str(e))
    print(f"       Error: {e}")

# 2e ── CLI: dpaad data attach-labels ────────────────────────────────────
print("\n--- 2e. CLI: dpaad data attach-labels ---")
with tempfile.TemporaryDirectory() as tmp:
    import shutil

    # Create a fresh copy of one system
    src = str(TRAIN_DIR / "sys_0000")
    dst = os.path.join(tmp, "sys_test")
    shutil.copytree(src, dst)

    # Create a labels npy file
    label_path = os.path.join(tmp, "labels.npy")
    np.save(label_path, np.array([3.14]))

    result = run_cli(
        [
            "data",
            "attach-labels",
            "--data",
            dst,
            "--head",
            "my_label",
            "--values",
            label_path,
        ]
    )
    print(f"  stdout: {result.stdout.strip()}")
    if result.stderr.strip():
        print(f"  stderr: {result.stderr.strip()}")
    check("CLI attach-labels exit code 0", result.returncode == 0)
    check(
        "CLI attach-labels log confirms attachment",
        "Labels attached" in result.stdout or "Labels attached" in result.stderr,
    )

    # Verify the .npy was written to disk
    cli_written = np.load(os.path.join(dst, "set.000", "my_label.npy"))
    check("CLI: my_label.npy written to disk", np.isclose(cli_written[0], 3.14))

# 2f ── Multi-system: attach_labels on parent directory ──────────────────
print("\n--- 2f. Python API: multi-system attach_labels ---")
with tempfile.TemporaryDirectory() as tmp:
    import shutil

    parent = os.path.join(tmp, "npy")
    os.makedirs(parent, exist_ok=True)
    # Copy 3 systems into the parent dir
    for i in range(3):
        src = str(TRAIN_DIR / f"sys_{i:04d}")
        dst = os.path.join(parent, f"sys_{i:04d}")
        shutil.copytree(src, dst)

    # Attach labels — values[i] → sorted(sys_*/) [i]
    labels = np.array([[1.0], [2.0], [3.0]])
    attach_labels(parent, head="multi_label", values=labels)

    for i in range(3):
        written = np.load(
            os.path.join(parent, f"sys_{i:04d}", "set.000", "multi_label.npy")
        )
        check(f"multi sys_{i:04d}: value matches", np.isclose(written[0], float(i + 1)))

# 2g ── Multi-system mismatch raises ValueError ──────────────────────────
print("\n--- 2g. Multi-system count mismatch ---")
with tempfile.TemporaryDirectory() as tmp:
    parent = os.path.join(tmp, "npy")
    os.makedirs(parent, exist_ok=True)
    for i in range(3):
        src = str(TRAIN_DIR / f"sys_{i:04d}")
        dst = os.path.join(parent, f"sys_{i:04d}")
        shutil.copytree(src, dst)
    try:
        attach_labels(
            parent, head="bad", values=np.array([[1.0], [2.0]])
        )  # 2 values, 3 systems
        check("ValueError raised for count mismatch", False)
    except ValueError as e:
        check(
            "ValueError raised for count mismatch",
            "entries along the first axis" in str(e) or "3 system" in str(e),
        )
        print(f"       Error: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 3. load_dataset(label_key="gap")
# ═══════════════════════════════════════════════════════════════════════════
section('3. load_dataset(label_key="gap")')

from dpa_adapt.data.dataset import (
    load_dataset,
)
from dpa_adapt.data.errors import (
    DPADataError,
)

# Note: dpdata's deepmd/npy loader only auto-loads standard keys
# (coord, box, energy, force, virial).  Custom labels like gap.npy
# must be attached first via attach_labels(), or you can pass already-
# labelled dpdata objects directly to load_dataset().

# 3a ── load_dataset with pre-attached labels ──────────────────────────────
print("\n--- 3a. load_dataset with pre-attached labels ---")
# Write gap labels to disk via path-based API
for sys_dir in sorted(TRAIN_DIR.glob("sys_*")):
    gap_val = np.load(sys_dir / "set.000" / "gap.npy")
    attach_labels(str(sys_dir), head="gap", values=gap_val)

# Load systems; dpdata ignores custom .npy labels, so we inject them manually.
# (DPAFineTuner._load_labels has the same fallback — reads set.*/gap.npy from
# disk when "gap" is not in system.data.)
all_train = load_data(TRAIN_GLOB)
for sys_dir, system in zip(sorted(TRAIN_DIR.glob("sys_*")), all_train):
    if "gap" not in system.data:
        system.data["gap"] = np.load(sys_dir / "set.000" / "gap.npy")
print(f"  Loaded {len(all_train)} systems")

gap_systems = load_dataset(all_train, label_key="gap")
print(f"  After filter: {len(gap_systems)} systems with 'gap' label")
check(
    "All 5 training systems have gap label after attach",
    len(gap_systems) == N_TRAIN,
)

all_have_gap = all("gap" in s.data for s in gap_systems)
check("Every returned system has 'gap' in data", all_have_gap)

# 3b ── load_dataset with label_key="energy" (none have it) ────────────────
print('\n--- 3b. load_dataset(label_key="energy") ---')
try:
    load_dataset(all_train, label_key="energy")
    check("DPADataError raised for missing energy label", False)
except DPADataError as e:
    check("DPADataError raised for missing energy label", "no valid systems" in str(e))
    print(f"       Error: {e}")

# 3c ── load_dataset on test data (with pre-attached gap) ─────────────────
print("\n--- 3c. load_dataset on test data ---")
for sys_dir in sorted(TEST_DIR.glob("sys_*")):
    gap_val = np.load(sys_dir / "set.000" / "gap.npy")
    attach_labels(str(sys_dir), head="gap", values=gap_val)
all_test = load_data(TEST_GLOB)
for sys_dir, system in zip(sorted(TEST_DIR.glob("sys_*")), all_test):
    if "gap" not in system.data:
        system.data["gap"] = np.load(sys_dir / "set.000" / "gap.npy")
gap_test = load_dataset(all_test, label_key="gap")
print(f"  Found {len(gap_test)} test systems with 'gap' label")
check("All 3 test systems have gap label", len(gap_test) == N_TEST)

# 3d ── load_dataset returns systems with the label key ───────────────────
print("\n--- 3d. load_dataset: returned systems carry the label ---")
# Note: systems loaded from deepmd/npy with non-standard labels (like gap.npy)
# are dpdata.System, not LabeledSystem.  dpdata only auto-promotes to
# LabeledSystem when standard keys (energy, force, virial) are present.
import dpdata

all_have_key = all("gap" in s.data for s in gap_systems)
check("All returned systems have 'gap' key in data", all_have_key)
# Also verify they are valid dpdata objects
all_dpdata = all(
    isinstance(s, (dpdata.System, dpdata.LabeledSystem)) for s in gap_systems
)
check("All returned systems are dpdata objects", all_dpdata)

# 3e ── load_dataset skips systems without the label ──────────────────────
print("\n--- 3e. load_dataset skips unlabelled systems ---")
# Mix labelled and unlabelled: inject gap labels into memory for first 5 only
mixed_dirs = sorted(TRAIN_DIR.glob("sys_*")) + sorted(TEST_DIR.glob("sys_*"))
for i, sys_dir in enumerate(mixed_dirs):
    if i < N_TRAIN:
        gap_val = np.load(sys_dir / "set.000" / "gap.npy")
        attach_labels(str(sys_dir), head="gap", values=gap_val)
mixed = load_data([str(d) for d in mixed_dirs])
for i, (sys_dir, system) in enumerate(zip(mixed_dirs, mixed)):
    if i < N_TRAIN and "gap" not in system.data:
        system.data["gap"] = np.load(sys_dir / "set.000" / "gap.npy")
result = load_dataset(mixed, label_key="gap")
print(f"  Mixed: {N_TOTAL} total, {len(result)} with gap label")
check("Only 5 of 8 mixed systems returned", len(result) == N_TRAIN)

# ═══════════════════════════════════════════════════════════════════════════
# 4. extract_descriptors() / CLI extract-descriptors
# ═══════════════════════════════════════════════════════════════════════════
section("4. extract_descriptors() / CLI extract-descriptors")

# Check whether deepmd C++ extensions are available (required for model
# construction).  If not available, verify the Python API surface and
# CLI wiring instead.
try:
    import deepmd.lib  # noqa: F401

    _HAVE_DEEPMD_LIB = True
except ImportError:
    _HAVE_DEEPMD_LIB = False

from dpa_adapt.finetuner import (
    extract_descriptors,
)

subset_paths = [str(TRAIN_DIR / f"sys_{i:04d}") for i in range(5)]

if _HAVE_DEEPMD_LIB:
    # ── full integration tests ───────────────────────────────────────────
    print("\n--- 4a. Python API: extract_descriptors on 5 systems ---")
    print(f"  Input: {len(subset_paths)} systems")

    descriptors = extract_descriptors(
        subset_paths,
        pretrained=PRETRAINED,
        model_branch="Domains_Drug",
        pooling="mean",
        cache=False,
    )
    print(f"  Output shape: {descriptors.shape}")
    check("descriptors is np.ndarray", isinstance(descriptors, np.ndarray))
    check("descriptors shape[0] == 5 (1 frame per system)", descriptors.shape[0] == 5)
    check("descriptors is 2D (n_frames, feat_dim)", descriptors.ndim == 2)
    print(f"  Feature dimension: {descriptors.shape[1]}")

    # 4b ── pooling strategies ───────────────────────────────────────────
    print("\n--- 4b. Python API: pooling='sum' ---")
    desc_sum = extract_descriptors(
        subset_paths,
        pretrained=PRETRAINED,
        model_branch="Domains_Drug",
        pooling="sum",
        cache=False,
    )
    print(f"  Output shape (sum): {desc_sum.shape}")
    check("sum pooling: 2D output", desc_sum.ndim == 2)
    check("sum pooling: n_frames matches", desc_sum.shape[0] == 5)

    print("\n--- 4c. Python API: pooling='mean+std' ---")
    desc_ms = extract_descriptors(
        subset_paths,
        pretrained=PRETRAINED,
        model_branch="Domains_Drug",
        pooling="mean+std",
        cache=False,
    )
    print(f"  Output shape (mean+std): {desc_ms.shape}")
    check("mean+std pooling: 2D output", desc_ms.ndim == 2)
    check("mean+std pooling: n_frames matches", desc_ms.shape[0] == 5)
    check(
        "mean+std feat_dim == 2 * mean feat_dim",
        desc_ms.shape[1] == 2 * descriptors.shape[1],
    )

    # 4d ── all 8 systems ────────────────────────────────────────────────
    print("\n--- 4d. Python API: extract_descriptors on all 8 systems ---")
    all_paths = sorted(TRAIN_DIR.glob("sys_*")) + sorted(TEST_DIR.glob("sys_*"))
    all_paths = [str(p) for p in all_paths]
    print(f"  Input: {len(all_paths)} systems")

    desc_all = extract_descriptors(
        all_paths,
        pretrained=PRETRAINED,
        model_branch="Domains_Drug",
        pooling="mean",
        cache=False,
    )
    print(f"  Output shape: {desc_all.shape}")
    check("all 8: shape[0] == 8", desc_all.shape[0] == N_TOTAL)
    check("all 8: 2D output", desc_all.ndim == 2)

    # 4e ── CLI ──────────────────────────────────────────────────────────
    print("\n--- 4e. CLI: dpaad extract-descriptors ---")
    with tempfile.TemporaryDirectory() as tmp:
        output_npy = os.path.join(tmp, "descriptors.npy")
        cli_paths = [str(TRAIN_DIR / f"sys_{i:04d}") for i in range(3)]
        result = run_cli(
            ["extract-descriptors", "--data"]
            + cli_paths
            + [
                "--pretrained",
                PRETRAINED,
                "--model-branch",
                "Domains_Drug",
                "--output",
                output_npy,
                "--no-cache",
            ]
        )
        print(f"  stdout: {result.stdout.strip()[:200]}")
        if result.stderr.strip():
            print(f"  stderr: {result.stderr.strip()[:200]}")
        check("CLI extract-descriptors exit code 0", result.returncode == 0)

        cli_desc = np.load(output_npy)
        print(f"  CLI output shape: {cli_desc.shape}")
        check("CLI output .npy shape[0] == 3", cli_desc.shape[0] == 3)
        check("CLI output .npy is 2D", cli_desc.ndim == 2)
        check(
            "CLI output feat_dim matches Python API",
            cli_desc.shape[1] == descriptors.shape[1],
        )

else:
    # ── smoke tests only (no deepmd C++ extensions) ─────────────────────
    print("\n  (deepmd C++ extensions not available — API smoke tests only)")
    print("\n--- 4a. extract_descriptors import + signature ---")
    import inspect

    sig = inspect.signature(extract_descriptors)
    params = list(sig.parameters.keys())
    print(f"  Signature: extract_descriptors({', '.join(params)})")
    check("extract_descriptors is callable", callable(extract_descriptors))
    check("extract_descriptors has 'data' param", "data" in params)
    check("extract_descriptors has 'pretrained' param", "pretrained" in params)
    check("extract_descriptors has 'pooling' param", "pooling" in params)

    # 4b ── Verify the function raises a clear error on missing deps ──────
    print("\n--- 4b. extract_descriptors raises clear error without deps ---")
    try:
        extract_descriptors(
            subset_paths,
            pretrained=PRETRAINED,
            model_branch="Domains_Drug",
            pooling="mean",
            cache=False,
        )
        check("ImportError raised for missing deepmd.lib", False)
    except ModuleNotFoundError as e:
        check("ModuleNotFoundError mentions deepmd", "deepmd" in str(e))
        print(f"       Error: {e}")
    except Exception as e:
        # Any exception is acceptable — the function shouldn't silently fail
        check(f"Exception raised (not silent): {type(e).__name__}", True)
        print(f"       Error: {e}")

    # 4c ── CLI shows help text ──────────────────────────────────────────
    print("\n--- 4c. CLI: dpaad extract-descriptors --help ---")
    result = run_cli(["extract-descriptors", "--help"])
    check("CLI help exit code 0", result.returncode == 0)
    check("CLI help mentions --data", "--data" in result.stdout)
    check("CLI help mentions --pretrained", "--pretrained" in result.stdout)
    check("CLI help mentions --output", "--output" in result.stdout)

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
section("Summary")
total = passed + failed
print(f"  {passed}/{total} passed", end="")
if failed:
    print(f", {failed} FAILED")
    sys.exit(1)
else:
    print(" — all good!")
