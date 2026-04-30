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
    deserialize_to_file(pt2_path, data, do_atomic_virial=True)

    # Produce a variant for regression-testing the C++ "atomic &&
    # !do_atomic_virial" throw path by copying the .pt2 archive and
    # flipping the do_atomic_virial flag in its metadata.json — much
    # cheaper than running a second AOTInductor compile.  The compiled
    # graph itself supports atomic virial; only the C++ guard differs.
    import shutil

    pt2_no_aviral = os.path.join(base_dir, "deeppot_sea_no_atomic_virial.pt2")
    print(f"Patching to {pt2_no_aviral} ...")  # noqa: T201
    shutil.copyfile(pt2_path, pt2_no_aviral)
    _patch_no_atomic_virial(pt2_no_aviral)

    print("Done!")  # noqa: T201


def _patch_no_atomic_virial(pt2_path: str) -> None:
    """Flip do_atomic_virial=False in the metadata.json of a .pt2 archive.

    The .pt2 is a ZIP archive; the metadata blob lives at
    ``model/extra/metadata.json``.  We rewrite the archive with that one entry
    replaced and all other entries preserved verbatim.
    """
    import json
    import zipfile

    from deepmd.pt_expt.utils.serialization import (
        PT2_EXTRA_PREFIX,
    )

    metadata_name = PT2_EXTRA_PREFIX + "metadata.json"
    tmp_path = pt2_path + ".tmp"
    # PyTorch .pt2 archives use ZIP_STORED (uncompressed) so that the C++
    # reader (read_zip_entry in commonPTExpt.h) and torch's mmap-based
    # tensor loader can read entries without decompression.  Preserve
    # that on rewrite — using ZIP_DEFLATED would yield bytes the C++
    # reader treats as raw, resulting in JSON parse errors.
    with zipfile.ZipFile(pt2_path, "r") as src:
        names = src.namelist()
        meta = json.loads(src.read(metadata_name).decode("utf-8"))
        meta["do_atomic_virial"] = False
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED) as dst:
            for name in names:
                if name == metadata_name:
                    dst.writestr(name, json.dumps(meta))
                else:
                    dst.writestr(name, src.read(name))
    os.replace(tmp_path, pt2_path)


if __name__ == "__main__":
    main()
