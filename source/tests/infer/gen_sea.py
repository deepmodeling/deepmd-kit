#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_sea.pt2 test model.

Converts the existing deeppot_sea.pth (checked into git) to .pt2 format
via serialize -> deserialize. Reference values are already in the C++ test files.
"""

import glob
import os
import shutil
import sys

# Ensure the source tree is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _ensure_inductor_compiler():
    """Ensure torch._inductor can find a C++ compiler.

    torch._inductor searches for 'g++' by default.  On some CI images only
    versioned binaries (e.g. g++-11) or 'c++' exist.  Fall back to those.
    """
    import torch._inductor.config as inductor_config

    search = inductor_config.cpp.cxx
    if isinstance(search, (list, tuple)):
        search = list(search)
    else:
        search = [search]
    # Append common fallbacks that are not in the default search list
    for fallback in ["c++", "g++-14", "g++-13", "g++-12", "g++-11"]:
        if fallback not in search and shutil.which(fallback):
            search.append(fallback)
    inductor_config.cpp.cxx = tuple(search)
    # Clear LD_PRELOAD so compiler subprocesses don't inherit the LSAN runtime,
    # which causes false leak reports and non-zero exit codes in g++.
    os.environ.pop("LD_PRELOAD", None)


def _load_custom_ops():
    """Load custom op library if not already registered.

    Must be called AFTER importing deepmd (which may register ops from the
    pip-installed library) to avoid double-registration crashes.
    """
    import torch

    if hasattr(torch.ops, "deepmd") and hasattr(torch.ops.deepmd, "border_op"):
        return
    _search_base = os.path.realpath(os.path.dirname(__file__))
    for pattern in [
        os.path.join(
            _search_base, "..", "..", "..", "build*", "op", "pt", "libdeepmd_op_pt.so"
        ),
        os.path.join(
            _search_base, "..", "..", "build*", "op", "pt", "libdeepmd_op_pt.so"
        ),
    ]:
        libs = glob.glob(pattern)
        if libs:
            try:
                torch.ops.load_library(libs[0])
            except Exception as e:
                print(f"NOTE: custom op library not loaded ({e})", file=sys.stderr)  # noqa: T201
            break


def main():
    import json

    import torch

    from deepmd.pt.model.model import (
        get_model,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    # Load custom ops after deepmd.pt import to avoid double registration
    _load_custom_ops()
    _ensure_inductor_compiler()

    base_dir = os.path.dirname(__file__)
    pth_path = os.path.join(base_dir, "deeppot_sea.pth")
    pt2_path = os.path.join(base_dir, "deeppot_sea.pt2")

    # Load the pre-committed .pth and serialize manually.
    # Use strict=False because the current model may have new keys
    # (e.g. min_nbor_dist, compress_info) not present in the old .pth.
    print(f"Loading {pth_path} ...")  # noqa: T201
    saved_model = torch.jit.load(pth_path, map_location="cpu")
    model_def_script = json.loads(saved_model.model_def_script)
    model = get_model(model_def_script)
    model.load_state_dict(saved_model.state_dict(), strict=False)
    model_dict = model.serialize()

    data = {
        "model": model_dict,
        "model_def_script": model_def_script,
        "backend": "PyTorch",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }

    print(f"Exporting to {pt2_path} ...")  # noqa: T201
    deserialize_to_file(pt2_path, data)

    print("Done!")  # noqa: T201


if __name__ == "__main__":
    main()
