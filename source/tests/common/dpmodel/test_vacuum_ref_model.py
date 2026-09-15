# SPDX-License-Identifier: LGPL-3.0-or-later
"""Isolated-atom reference at the model level on the dpmodel routes.

With ``vacuum_ref`` in the fitting, an isolated atom contributes exactly the
bias of its type, and the energy of a cluster equals the energy of the same
model without the reference minus, for every atom, the network output of an
isolated atom of its type. The dense neighbor-list route evaluates the vacuum
descriptor on single-atom frames; the graph route carries one reference node
per type through the same descriptor call.
"""

import numpy as np
import pytest

from deepmd.dpmodel.descriptor.dpa1 import (
    DescrptDPA1,
)
from deepmd.dpmodel.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.ener_model import (
    EnergyModel,
)

TYPE_MAP = ["O", "H"]
BIAS = np.array([[-3.0], [0.5]])


def make_model(kind: str, vacuum_ref: bool) -> EnergyModel:
    if kind == "se_e2_a":
        ds = DescrptSeA(
            rcut=4.0, rcut_smth=0.5, sel=[10, 10], neuron=[4, 8], axis_neuron=2, seed=1
        )
    else:
        ds = DescrptDPA1(
            rcut=4.0,
            rcut_smth=0.5,
            sel=[20],
            ntypes=2,
            attn_layer=0,
            axis_neuron=2,
            neuron=[6, 12],
            seed=1,
        )
    ft = InvarFitting(
        "energy",
        2,
        ds.get_dim_out(),
        1,
        neuron=[8, 8],
        mixed_types=ds.mixed_types(),
        vacuum_ref=vacuum_ref,
        seed=1,
    )
    ft["bias_atom_e"] = BIAS.copy()
    return EnergyModel(ds, ft, type_map=TYPE_MAP)


def atom_energies(
    model: EnergyModel, coord: np.ndarray, atype: np.ndarray, method: str
) -> np.ndarray:
    ret = model.call_common(coord, atype, None, neighbor_graph_method=method)
    return ret["energy"][..., 0]


@pytest.mark.parametrize(
    "kind, method",
    [("se_e2_a", "legacy"), ("dpa1", "dense")],  # dense nlist route / graph route
)
def test_isolated_atoms_and_reference_subtraction(kind: str, method: str) -> None:
    rng = np.random.default_rng(0)
    coord = rng.normal(size=(2, 6, 3)) * 1.2
    atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
    iso_coord = np.zeros((2, 1, 3))
    iso_atype = np.array([[0], [1]])
    vac = make_model(kind, True)
    ref = make_model(kind, False)

    # an isolated atom of every type contributes exactly its bias
    np.testing.assert_allclose(
        atom_energies(vac, iso_coord, iso_atype, method),
        BIAS[:, 0][:, None],
        rtol=1e-10,
        atol=1e-10,
    )
    # the reference removes the isolated-atom network output of every atom
    e_vac = atom_energies(vac, coord, atype, method)
    e_ref = atom_energies(ref, coord, atype, method)
    iso_ref = atom_energies(ref, iso_coord, iso_atype, method)[:, 0]
    expected = e_ref - (iso_ref - BIAS[:, 0])[atype]
    np.testing.assert_allclose(e_vac, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "kind, method",
    [("se_e2_a", "legacy"), ("dpa1", "dense")],  # dense nlist route / graph route
)
def test_fold_reproduces_the_referenced_model(kind: str, method: str) -> None:
    """Folding the reference into the bias keeps every energy and drops the rows."""
    rng = np.random.default_rng(1)
    coord = rng.normal(size=(2, 6, 3)) * 1.2
    atype = np.array([[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 1, 0]])
    iso_coord = np.zeros((2, 1, 3))
    iso_atype = np.array([[0], [1]])
    model = make_model(kind, True)
    e_cluster = atom_energies(model, coord, atype, method)
    e_iso = atom_energies(model, iso_coord, iso_atype, method)

    model.fold_vacuum_reference()
    assert not model.atomic_model.fitting_net.vacuum_ref
    np.testing.assert_allclose(
        atom_energies(model, coord, atype, method), e_cluster, rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        atom_energies(model, iso_coord, iso_atype, method),
        e_iso,
        rtol=1e-10,
        atol=1e-10,
    )
    # the folded model is a plain model: its serialization carries no reference
    assert model.serialize()["fitting"]["vacuum_ref"] is False
