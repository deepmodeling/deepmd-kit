# SPDX-License-Identifier: LGPL-3.0-or-later
"""Regression test: SpinModel backbone must stay on the legacy dense-nlist path.

A spin system uses virtual/placeholder types whose pair exclusions double the
effective atom density.  Before the fix (commit 6c2b007c9), the dpmodel
``call_common`` auto-flip (decision #17) moved graph-eligible mixed_types
backbones to the carry-all graph route.  When a spin model's backbone
(dpa1 attn_layer=0) crossed to the graph route, its ``call_common`` diverged
from the dense lower interface (``call_common_lower``) used by pt_expt eager
inference -- the carry-all graph keeps neighbors the capped dense nlist
discards.

Fix: ``SpinModel.call_common`` forces ``neighbor_graph_method="legacy"`` on
the backbone call so the spin backbone always runs dense regardless of the
default flip.

This test pins that contract:

1. ``SpinModel.call_common`` energy == backbone energy computed with explicit
   ``neighbor_graph_method="legacy"`` on the spin-doubled coordinate inputs.
2. The backbone in graph mode (``neighbor_graph_method="ase"``) on the same
   doubled inputs gives a DIFFERENT energy, confirming the fixture is
   sel-binding (i.e., the virtual-atom density really does trigger divergence).
"""

import numpy as np
import pytest

from deepmd.dpmodel.model.model import (
    get_spin_model,
)


def _spin_dpa1_config() -> dict:
    """Minimal dpa1-backed spin model config (sel-binding on doubled inputs)."""
    return {
        "type": "standard",
        "type_map": ["Fe", "H"],
        "spin": {
            "use_spin": [True, False],
            "virtual_scale": [0.3],
        },
        "descriptor": {
            "type": "dpa1",
            "rcut": 4.0,
            "rcut_smth": 0.5,
            # Small sel: with 8 real + 8 virtual = 16 atoms in a 6 Å box,
            # sel=8 is well below the average neighbor count → sel-binding.
            "sel": 8,
            "ntypes": 4,  # expanded to 4 by get_spin_model (2 real + 2 virtual)
            "attn_layer": 0,
            "axis_neuron": 2,
            "neuron": [4, 8],
            "seed": 0,
        },
        "fitting_net": {
            "type": "ener",
            "neuron": [4, 4],
            "seed": 0,
        },
    }


def _make_test_frame(rng: np.random.Generator):
    """Return (coord, atype, box) for a small PBC cell with 8 atoms."""
    natoms = 8  # 4 Fe + 4 H
    coord = rng.random((1, natoms, 3)) * 4.0  # 1 frame
    atype = np.array([[0, 0, 0, 0, 1, 1, 1, 1]])
    box = np.eye(3).reshape(1, 9) * 6.0
    return coord, atype, box


def test_spin_model_backbone_routes_legacy() -> None:
    """SpinModel.call_common energy must equal backbone with explicit legacy.

    Procedure:
    - Build the SpinModel and obtain the doubled coords/atypes via
      ``process_spin_input`` (the same transform used internally).
    - Call the backbone directly with ``neighbor_graph_method="legacy"`` on
      those doubled inputs to get the expected dense-path energy.
    - Call ``SpinModel.call_common`` and extract its energy.
    - Assert the two energies are exactly equal.
    - Additionally assert the backbone in ``neighbor_graph_method="ase"`` mode
      on the same doubled inputs gives a DIFFERENT energy (sel-binding guard).
    """
    pytest.importorskip("ase")  # ase builder needed for divergence check

    rng = np.random.default_rng(42)
    coord, atype, box = _make_test_frame(rng)
    spin = np.zeros_like(coord)
    spin[:, :4, 2] = 1.0  # spin-z on Fe atoms only

    model = get_spin_model(_spin_dpa1_config())
    backbone = model.backbone_model

    # --- Get doubled inputs via the model's own transform ---
    coord_doubled, atype_doubled, _corr = model.process_spin_input(coord, atype, spin)

    # --- Backbone with explicit legacy routing ---
    legacy_ret = backbone.call_common(
        coord_doubled, atype_doubled, box, neighbor_graph_method="legacy"
    )
    legacy_energy = np.array(legacy_ret["energy"])

    # --- SpinModel.call_common (must route legacy internally) ---
    spin_ret = model.call_common(coord, atype, spin, box)
    spin_energy = np.array(spin_ret["energy"])

    # ``energy`` here is per-atom energy; the backbone returns all 2*nloc
    # atoms while the spin model truncates to the first nloc real atoms.
    # Compare the total energy (sum over all atoms) which should be equal
    # because the virtual-atom energies are zeroed by the exclusion mask.
    np.testing.assert_allclose(
        float(spin_energy.sum()),
        float(legacy_energy.sum()),
        rtol=0,
        atol=0,
        err_msg=(
            "SpinModel.call_common total energy does not match backbone(legacy) "
            "on doubled inputs; the spin model may be routing through the "
            "carry-all graph instead of the legacy dense path."
        ),
    )

    # --- Sel-binding guard: backbone graph must DIFFER from backbone legacy ---
    graph_ret = backbone.call_common(
        coord_doubled, atype_doubled, box, neighbor_graph_method="ase"
    )
    graph_energy = np.array(graph_ret["energy"])

    assert not np.allclose(legacy_energy.sum(), graph_energy.sum(), rtol=1e-6), (
        "Backbone legacy and graph give the same energy on the doubled spin "
        "system — sel is not binding with these inputs; the regression fixture "
        "is too weak.  Reduce sel or increase atom density."
    )


def test_spin_model_call_common_deterministic() -> None:
    """SpinModel.call_common is deterministic (no stochastic routing)."""
    rng = np.random.default_rng(7)
    coord, atype, box = _make_test_frame(rng)
    spin = np.zeros_like(coord)
    spin[:, :4, 2] = 1.0

    model = get_spin_model(_spin_dpa1_config())

    ret1 = model.call_common(coord, atype, spin, box)
    ret2 = model.call_common(coord, atype, spin, box)

    np.testing.assert_array_equal(
        float(np.array(ret1["energy"]).sum()),
        float(np.array(ret2["energy"]).sum()),
        err_msg="SpinModel.call_common is non-deterministic",
    )
