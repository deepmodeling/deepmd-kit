# SPDX-License-Identifier: LGPL-3.0-or-later
"""Common utilities shared by model generation scripts (gen_*.py)."""

import glob
import os
import sys


def ensure_inductor_compiler():
    """Ensure torch._inductor can find a C++ compiler.

    Honours the ``CXX`` environment variable (standard CMake / CI convention).
    Falls back to torch._inductor's built-in default if ``CXX`` is not set.
    Also clears ``LD_PRELOAD`` so compiler subprocesses don't inherit the LSAN
    runtime, which causes false leak reports and non-zero exit codes in g++.
    """
    cxx = os.environ.get("CXX")
    if cxx:
        import torch._inductor.config as inductor_config

        inductor_config.cpp.cxx = (cxx,)
    os.environ.pop("LD_PRELOAD", None)


def load_custom_ops():
    """Load custom op library if not already registered.

    Normally ``import deepmd.pt`` loads the library from SHARED_LIB_DIR
    (via ``cxx_op.py``).  This function is a fallback for running gen
    scripts in environments where the .so hasn't been installed there
    (e.g. standalone development).  It searches build directories for
    ``libdeepmd_op_pt.so`` and loads the first one found.

    Must be called AFTER importing deepmd.pt to avoid double-registration.
    """
    import torch

    if hasattr(torch.ops, "deepmd") and hasattr(torch.ops.deepmd, "border_op"):
        return
    # Search common build directory locations relative to this file
    search_base = os.path.realpath(os.path.dirname(__file__))
    candidates = glob.glob(
        os.path.join(
            search_base, "..", "..", "..", "build*", "op", "pt", "libdeepmd_op_pt.so"
        )
    ) + glob.glob(
        os.path.join(
            search_base, "..", "..", "build*", "op", "pt", "libdeepmd_op_pt.so"
        )
    )
    for lib in candidates:
        try:
            torch.ops.load_library(lib)
        except Exception as e:
            print(f"NOTE: custom op library not loaded ({e})", file=sys.stderr)  # noqa: T201
        break


def print_cpp_values(label, ae, f, av):
    """Print C++ reference arrays for energy, force, and virial."""
    print(f"\n// ---- {label} ----")  # noqa: T201
    atom_energy = ae[0, :, 0]
    print("  std::vector<VALUETYPE> expected_e = {")  # noqa: T201
    for ii, e in enumerate(atom_energy):
        comma = "," if ii < len(atom_energy) - 1 else ""
        print(f"      {e:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_f = {")  # noqa: T201
    force_flat = f[0].flatten()
    for ii, fv in enumerate(force_flat):
        comma = "," if ii < len(force_flat) - 1 else ""
        print(f"      {fv:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_v = {")  # noqa: T201
    virial_flat = av[0].flatten()
    for ii, v in enumerate(virial_flat):
        comma = "," if ii < len(virial_flat) - 1 else ""
        print(f"      {v:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201
