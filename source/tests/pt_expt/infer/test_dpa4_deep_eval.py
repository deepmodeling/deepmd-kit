# SPDX-License-Identifier: LGPL-3.0-or-later
"""DPA4/SeZM DeepEval parity: pt (.pt) vs pt_expt (.pt2).

This test doubles as the pt-checkpoint -> pt_expt interop proof.  A single
DPA4/SeZM model is built with the *pt* backend and random-initialised, then:

1. pt reference: the pt model + its ``model_params`` are written to a ``.pt``
   checkpoint and evaluated through ``DeepPot(.pt)`` (routes to the pt backend;
   SeZM disables torch.jit so this is eager pt inference).
2. pt_expt path: the SAME pt model's ``serialize()`` dict is fed to
   ``deserialize_to_file`` which calls ``pt_expt.BaseModel.deserialize`` (the
   checkpoint-interop step), compiles via AOTInductor, and packs a ``.pt2``
   archive evaluated through ``DeepPot(.pt2)``.

Because the weights are transferred by serialize/deserialize (not retrained),
the two backends must produce identical conservative quantities.  Energy,
force and the *global* virial are compared at fp64 cross-backend tolerance
(rtol/atol 1e-10).

Per-atom virial is NOT compared element-wise: pt's SeZM force/virial uses an
edge-force scatter that distributes the per-atom virial differently from
pt_expt's generic ``fit_output_to_model_output`` assembly (#5518).  Both are
correct -- their sum (the global virial) matches at 1e-10 -- but the per-atom
distribution legitimately differs, so we assert the global virial only and
additionally check that pt_expt's per-atom virial *sums* to its global virial.
"""

from __future__ import (
    annotations,
)

import copy
import os

import numpy as np
import pytest
import torch

from deepmd.infer import (
    DeepPot,
)
from deepmd.pt.model.model import (
    get_model as pt_get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper as PtModelWrapper,
)
from deepmd.pt_expt.utils.serialization import (
    deserialize_to_file,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)

# Small fp64 DPA4 config: channels 16, n_radial 8, lmax 2, mmax 1, n_blocks 2,
# fitting neuron [16] -- mirrors test_dpa4_export so the AOTI compile time is
# bounded but still exercises the SO(2)/SO(3) + attention + embedding paths.
_DPA4_RAW_CONFIG = {
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


def _normalize_model(model: dict) -> dict:
    config = {
        "model": copy.deepcopy(model),
        "training": {"training_data": {"systems": ["dummy"]}, "numb_steps": 1},
        "loss": {"type": "ener"},
        "learning_rate": {"type": "exp", "start_lr": 1e-3},
    }
    config = update_deepmd_input(config, warning=False)
    config = normalize(config)
    return config["model"]


# A small, fixed water-like box: 2 oxygens + 4 hydrogens.  Coordinates are
# explicit (no RNG) so the test is fully deterministic.
_NATOMS = 6
_ATYPES = np.array([0, 0, 1, 1, 1, 1], dtype=np.int32)  # O, O, H, H, H, H
_COORDS = np.array(
    [
        [1.0, 1.0, 1.0],
        [3.2, 1.4, 1.1],
        [1.3, 1.8, 1.0],
        [0.4, 1.2, 1.6],
        [3.6, 2.0, 1.3],
        [3.4, 0.7, 1.7],
    ],
    dtype=np.float64,
).reshape(1, _NATOMS, 3)
_CELL = (np.eye(3, dtype=np.float64) * 6.0).reshape(1, 9)


@pytest.fixture(scope="module")
def dpa4_pt_and_pt2(tmp_path_factory):
    """Build one pt DPA4 model; emit a pt ``.pt`` and a pt_expt ``.pt2``."""
    tmp_path = tmp_path_factory.mktemp("dpa4_deep_eval")
    model_params = _normalize_model(_DPA4_RAW_CONFIG)

    # Build the pt model and random-init it (fp64, eval mode).
    pt_model = pt_get_model(copy.deepcopy(model_params))
    pt_model = pt_model.to(torch.float64)
    pt_model.eval()

    # 1. pt `.pt` checkpoint: state_dict + model_params in _extra_state.
    pt_path = str(tmp_path / "dpa4.pt")
    wrapper = PtModelWrapper(pt_model, model_params=copy.deepcopy(model_params))
    torch.save({"model": wrapper.state_dict()}, pt_path)

    # 2. pt_expt `.pt2`: transfer weights via serialize() -> BaseModel.deserialize
    #    (the interop step) inside deserialize_to_file, then AOTI-compile/pack.
    pt2_path = str(tmp_path / "dpa4.pt2")
    data = {"model": pt_model.serialize()}
    deserialize_to_file(pt2_path, data, do_atomic_virial=True)

    return {"pt": pt_path, "pt2": pt2_path}


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
@pytest.mark.parametrize("pbc", [True, False])  # periodic vs open boundary
def test_dpa4_deep_eval_parity(dpa4_pt_and_pt2, pbc) -> None:
    """Backends agree: pt (.pt) vs pt_expt (.pt2) at fp64 tolerance."""
    dp_pt = DeepPot(dpa4_pt_and_pt2["pt"])
    dp_pt2 = DeepPot(dpa4_pt_and_pt2["pt2"])

    cell = _CELL if pbc else None

    e_pt, f_pt, v_pt, ae_pt, av_pt = dp_pt.eval(_COORDS, cell, _ATYPES, atomic=True)
    e_x, f_x, v_x, ae_x, av_x = dp_pt2.eval(_COORDS, cell, _ATYPES, atomic=True)

    tag = "pbc" if pbc else "nopbc"
    np.testing.assert_allclose(
        e_x, e_pt, rtol=1e-10, atol=1e-10, err_msg=f"{tag}: energy"
    )
    np.testing.assert_allclose(
        f_x, f_pt, rtol=1e-10, atol=1e-10, err_msg=f"{tag}: force"
    )
    np.testing.assert_allclose(
        v_x, v_pt, rtol=1e-10, atol=1e-10, err_msg=f"{tag}: global virial"
    )
    np.testing.assert_allclose(
        ae_x, ae_pt, rtol=1e-10, atol=1e-10, err_msg=f"{tag}: atom energy"
    )

    # Per-atom virial: pt's edge-force scatter (#5518) distributes the
    # per-atom virial differently from pt_expt's generic assembly.  The two
    # are NOT expected to match element-wise; only the global virial (their
    # sum) is a physical observable.  Verify pt_expt's per-atom virial reduces
    # to its own global virial so the assembly stays self-consistent.
    av_x_sum = av_x.reshape(1, _NATOMS, 9).sum(axis=1)
    np.testing.assert_allclose(
        av_x_sum,
        v_x.reshape(1, 9),
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"{tag}: pt_expt atom_virial does not sum to global virial",
    )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (minutes); run locally only by default.",
)
def test_dpa4_deep_eval_metadata(dpa4_pt_and_pt2) -> None:
    """Both backends expose the same model metadata (rcut/type_map/...)."""
    dp_pt = DeepPot(dpa4_pt_and_pt2["pt"])
    dp_pt2 = DeepPot(dpa4_pt_and_pt2["pt2"])

    assert dp_pt2.deep_eval.get_rcut() == dp_pt.deep_eval.get_rcut()
    assert dp_pt2.deep_eval.get_type_map() == dp_pt.deep_eval.get_type_map()
    assert dp_pt2.deep_eval.get_ntypes() == dp_pt.deep_eval.get_ntypes()
    assert dp_pt2.deep_eval.get_dim_fparam() == dp_pt.deep_eval.get_dim_fparam()
    assert dp_pt2.deep_eval.get_dim_aparam() == dp_pt.deep_eval.get_dim_aparam()
    assert not dp_pt2.has_spin
