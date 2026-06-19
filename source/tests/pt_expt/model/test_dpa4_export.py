# SPDX-License-Identifier: LGPL-3.0-or-later
"""Model-level freeze test for the DPA4/SeZM energy model.

Mirrors the DPA3 ``test_export_with_comm`` round-trip: a DPA4 model is a
GNN (``has_message_passing_across_ranks() == True``), so
``deserialize_to_file`` produces a .pt2 archive containing TWO compiled
artifacts:
  * the regular ``forward_lower`` (no comm), packed at the top of the ZIP;
  * a ``forward_lower_with_comm`` variant nested at
    ``model/extra/forward_lower_with_comm.pt2``.

This test verifies:
  1. The .pt2 archive is produced and both artifacts are present.
  2. ``metadata.json`` carries the correct ``type_map``/``rcut`` and
     ``has_message_passing: true`` (DPA4 is a message-passing descriptor).
  3. The regular artifact loads via ``aoti_load_package``.
  4. The loaded artifact's ``forward_common_lower`` output matches the
     eager model (fp64 AOTI parity, rtol 1e-10).
"""

from __future__ import (
    annotations,
)

import json
import os
import zipfile

import numpy as np
import pytest

# Note: registration of the deepmd_export::border_op opaque wrapper (needed by
# the with-comm artifact) happens inside ``deserialize_to_file`` via
# ``ensure_comm_registered()``; no explicit comm import is required here.
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils.serialization import (
    _make_sample_inputs,
    deserialize_to_file,
)

# Small fp64 DPA4 config (channels 16, n_radial 8, lmax 2, mmax 1,
# n_blocks 2) — large enough to exercise the SO(2)/SO(3) + attention +
# embedding paths that previously specialized ``nloc`` during export, but
# small enough to keep the AOTInductor compile time bounded.
_DPA4_CONFIG = {
    "type": "dpa4",
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4",
        "sel": 20,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 1,
    },
    "fitting_net": {
        "type": "dpa4_ener",
        "neuron": [16],
        "precision": "float64",
        "seed": 1,
    },
}


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
def test_dpa4_freeze_to_pt2(tmp_path) -> None:
    """End-to-end: DPA4 model freezes to a dual-artifact .pt2 and the
    regular artifact reproduces the eager ``forward_common_lower``.
    """
    model = get_model(_DPA4_CONFIG)
    model.to("cpu")
    model.eval()

    # 1. Serialize → deserialize_to_file (compiles and packs both artifacts).
    pt2_path = str(tmp_path / "test_dpa4.pt2")
    deserialize_to_file(pt2_path, {"model": model.serialize()})
    assert os.path.exists(pt2_path)

    # 2. ZIP layout + metadata sanity. PyTorch's strict layout puts our
    #    sidecars under ``model/extra/`` (PT2_EXTRA_PREFIX).
    with zipfile.ZipFile(pt2_path, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
        assert "model/extra/forward_lower_with_comm.pt2" in names, (
            f"with-comm artifact missing; names={sorted(names)}"
        )
    assert meta["type_map"] == _DPA4_CONFIG["type_map"]
    assert meta["rcut"] == model.get_rcut()
    # DPA4 is a message-passing GNN descriptor.
    assert meta["has_message_passing"] is True
    assert meta["has_comm_artifact"] is True

    # 3. The regular artifact loads.
    from torch._inductor import (
        aoti_load_package,
    )

    regular = aoti_load_package(pt2_path)

    # 4. Eager reference vs. AOTI artifact parity on forward_common_lower.
    sample = _make_sample_inputs(model, nframes=1, has_spin=False)
    ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam, charge_spin = sample

    eager_out = model.forward_common_lower(
        ext_coord.detach().requires_grad_(True),
        ext_atype,
        nlist_t,
        mapping_t,
        fparam=fparam,
        aparam=aparam,
        do_atomic_virial=False,
        charge_spin=charge_spin,
    )

    artifact_out = regular(
        ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam, charge_spin
    )

    # The AOTI artifact returns the internal forward_common_lower keys; compare
    # every key it produces against the eager reference (fp64 AOTI tolerance).
    compared = 0
    for key, val in artifact_out.items():
        if key not in eager_out or eager_out[key] is None or val is None:
            continue
        np.testing.assert_allclose(
            val.detach().cpu().numpy(),
            eager_out[key].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"artifact vs eager forward_common_lower differs: {key}",
        )
        compared += 1
    # Guard against a vacuous pass (no overlapping keys compared).
    assert compared > 0, (
        f"no overlapping output keys compared; artifact keys="
        f"{sorted(artifact_out)}, eager keys={sorted(eager_out)}"
    )
    # The energy output must be among the compared keys.
    assert "energy_redu" in artifact_out or "energy" in artifact_out
