#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate deeppot_sea.pt2 test model.

Converts the existing deeppot_sea.pth (checked into git) to .pt2 format
via serialize -> deserialize. Reference values are already in the C++ test files.
"""

import os
import sys

# Ensure the source tree is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
)


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
    load_custom_ops()
    ensure_inductor_compiler()

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
