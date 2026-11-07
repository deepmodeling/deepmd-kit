#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the compressed CUDA fixture for LAMMPS ``dpa4spin/kk``.

The Kokkos adapter requires the compact canonical graph ABI, which is only
available for a compressed float32 DPA4C model with a fused fitting network.
Keep this archive separate from the float64 CPU graph fixture. Both carry
native spin and no frame, atomic, or charge/spin conditioning parameters.

Run with a CUDA-enabled DeePMD build and a visible GPU. A CPU-only invocation
fails rather than silently exporting an archive the Kokkos adapter cannot use.
"""

from __future__ import (
    annotations,
)

import copy
import json
import os
import zipfile

import gen_dpa4c_spin_lmp as host_fixture
import numpy as np
from gen_common import (
    ensure_inductor_compiler,
    load_custom_ops,
)

CFG = copy.deepcopy(host_fixture.CFG)
CFG["descriptor"]["precision"] = "float32"
CFG["fitting_net"]["precision"] = "float32"
# The fused fitting kernel does not support per-layer residual timesteps.
CFG["fitting_net"]["resnet_dt"] = False
OUT = os.path.join(os.path.dirname(__file__), "deeppot_dpa4c_spin_lmp_kokkos.pt2")


def _build_model_data() -> dict:
    """Build reproducible weights and compression tables on the CPU."""
    from deepmd.dpmodel.model.model import (
        get_model,
    )
    from deepmd.pt_expt.kernels.dpa4c.graph_compress import (
        mega_eligible,
    )
    from deepmd.pt_expt.kernels.graph_fitting import (
        fitting_eligible,
    )
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )

    model_dict = host_fixture.jitter_zero_arrays(
        get_model(copy.deepcopy(CFG)).serialize(),
        np.random.default_rng(host_fixture._JITTER_SEED),
    )
    model = BaseModel.deserialize(model_dict).to("cpu").eval()
    descriptor = model.get_descriptor()
    assert mega_eligible(descriptor), "the fixture needs a compiled DPA4C profile"
    assert fitting_eligible(model.get_fitting_net()), "the fixture needs fused fitting"
    # Jitter before compression so the tabulated snapshot retains the live
    # native-spin branch, including its derivative at exactly zero spin.
    descriptor.enable_compression(min_nbor_dist=0.5)
    assert descriptor.compress
    return {
        "model": model.serialize(),
        "model_def_script": copy.deepcopy(CFG),
        "backend": "pt_expt",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }


def _check_metadata(path: str) -> None:
    """Reject an archive that would exercise a different deployment route."""
    with zipfile.ZipFile(path) as zf:
        metadata = json.loads(zf.read("model/extra/metadata.json"))
    expected = {
        "type_map": ["Ni", "O"],
        "spin_scheme": "native",
        "use_spin": [True, False],
        "has_comm_artifact": False,
        "dim_fparam": 0,
        "dim_aparam": 0,
        "dim_chg_spin": 0,
        "lower_input_kind": "dpa4c_canonical",
        "canonical_index_dtype": "auto",
        "canonical_index_storage": "byte_words",
        "graph_edge_dtype": "float32",
    }
    for key, value in expected.items():
        assert metadata.get(key) == value, (key, metadata.get(key), value)
    for key in ("energy", "force", "force_mag", "virial", "atom_virial"):
        assert key in metadata["output_keys"], key
    print(json.dumps({key: metadata[key] for key in expected}, indent=2))  # noqa: T201


def main() -> None:
    from deepmd.pt_expt.kernels.dpa4c.canonical import (
        canonical_model_eligible,
        op_available,
    )
    from deepmd.pt_expt.model.model import (
        BaseModel,
    )
    from deepmd.pt_expt.utils.env import (
        DEVICE,
    )
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    if DEVICE.type != "cuda":
        raise RuntimeError(
            "The dpa4spin/kk fixture requires a CUDA-enabled DeePMD build "
            "and a visible GPU (without DEVICE=cpu)."
        )
    ensure_inductor_compiler()
    load_custom_ops()
    if not op_available():
        raise RuntimeError(
            "The compressed canonical DPA4C CUDA operators are required."
        )
    data = _build_model_data()
    assert canonical_model_eligible(BaseModel.deserialize(data["model"]))
    print(f"Exporting {OUT}")  # noqa: T201
    deserialize_to_file(OUT, data, do_atomic_virial=True, lower_kind="dpa4c_canonical")
    _check_metadata(OUT)


if __name__ == "__main__":
    main()
