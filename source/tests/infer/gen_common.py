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
    """Print C++ reference arrays for energy, force, and virial.

    Debug helper. Production gen scripts should write sidecar reference files
    via ``write_expected_ref`` instead; this is kept for ad-hoc inspection
    when porting models or comparing pt/pt_expt outputs by hand.
    """
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


def print_cpp_spin_values(label, ae, f, fm, tot_v, av):
    """Print C++ reference arrays for spin models.

    Debug helper, see ``print_cpp_values``.
    """
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

    print("  std::vector<VALUETYPE> expected_fm = {")  # noqa: T201
    fm_flat = fm[0].flatten()
    for ii, fv in enumerate(fm_flat):
        comma = "," if ii < len(fm_flat) - 1 else ""
        print(f"      {fv:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_tot_v = {")  # noqa: T201
    tot_v_flat = tot_v[0].flatten()
    for ii, v in enumerate(tot_v_flat):
        comma = "," if ii < len(tot_v_flat) - 1 else ""
        print(f"      {v:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201

    print("  std::vector<VALUETYPE> expected_atom_v = {")  # noqa: T201
    av_flat = av[0].flatten()
    for ii, v in enumerate(av_flat):
        comma = "," if ii < len(av_flat) - 1 else ""
        print(f"      {v:.18e}{comma}")  # noqa: T201
    print("  };")  # noqa: T201


def write_expected_ref(path, sections, source_script=None):
    """Write a plain-text sidecar reference file for C++ tests.

    Parameters
    ----------
    path : str
        Output file path.
    sections : dict[str, dict[str, np.ndarray]]
        Mapping ``case_name -> {array_name: 1-D or flattenable array}``.
    source_script : str, optional
        Name of the gen script for the header comment.

    Notes
    -----
    File layout::

        # auto-generated by <source_script> -- do not edit
        [pbc]
        expected_e 6
        5.386638169248214592e-02
        ...
        expected_f 18
        ...

        [nopbc]
        ...
    """
    import numpy as np

    lines = []
    if source_script:
        lines.append(f"# auto-generated by {source_script} -- do not edit")
    else:
        lines.append("# auto-generated -- do not edit")
    for case_name, arrays in sections.items():
        lines.append(f"[{case_name}]")
        for key, arr in arrays.items():
            flat = np.asarray(arr).reshape(-1)
            lines.append(f"{key} {flat.size}")
            for v in flat:
                lines.append(f"{float(v):.18e}")
        lines.append("")  # blank line between sections
    with open(path, "w") as fp:
        fp.write("\n".join(lines).rstrip() + "\n")
