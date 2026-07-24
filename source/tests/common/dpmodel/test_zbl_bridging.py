# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel ZBL bridging integration (atomic-layer InterPotential injection).

Twin of pt's ``test_sezm_model.py::test_zbl_adds_energy``: with identical
weights, the ZBL model's energy exceeds the plain model's by a positive
(repulsive) amount on a system with a close pair; the term rides the
NeighborGraph route only (the dense route raises).
"""

import copy

import numpy as np
import pytest

from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.model import (
    get_model,
)

ZBL_CONFIG = {
    "type_map": ["Ni", "O"],
    "descriptor": {
        "type": "dpa4",
        "rcut": 4.0,
        "sel": 8,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 7,
        "random_gamma": False,
    },
    "fitting_net": {"type": "dpa4_ener", "neuron": [8, 8]},
    "bridging_method": "ZBL",
    "bridging_r_inner": 0.8,
    "bridging_r_outer": 1.2,
}


def _models():
    """ZBL model + a plain twin with IDENTICAL weights (incl. InnerClamp).

    Built by deleting only the ``bridging_method`` key from the serialized
    atomic dict, so the descriptor (including the bridging radii's
    InnerClamp/BridgingSwitch) is byte-identical -- the energy difference
    isolates the InterPotential term.
    """
    m_zbl = get_model(copy.deepcopy(ZBL_CONFIG))
    data = m_zbl.serialize()
    assert data["bridging_method"] == "ZBL"
    plain_data = copy.deepcopy(data)
    plain_data.pop("bridging_method")
    m_plain = BaseModel.deserialize(plain_data)
    assert m_plain.atomic_model.inter_potential is None
    return m_zbl, m_plain


def _close_pair_inputs():
    rng = np.random.default_rng(5)
    coord = rng.uniform(1.5, 5.5, size=(1, 6, 3))
    coord[0, 1] = coord[0, 0] + np.array([0.9, 0.0, 0.0])  # close Ni-Ni pair
    atype = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
    box = 8.0 * np.eye(3, dtype=np.float64)[None]
    return coord, atype, box


def test_zbl_adds_positive_energy():
    m_zbl, m_plain = _models()
    coord, atype, box = _close_pair_inputs()
    e_zbl = m_zbl.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    e_plain = m_plain.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    diff = float(np.sum(e_zbl - e_plain))
    assert diff > 1e-3, f"ZBL repulsion missing or non-positive: {diff:.3e}"


def test_zbl_serialize_roundtrip_energy_identical():
    m_zbl, _ = _models()
    coord, atype, box = _close_pair_inputs()
    m2 = BaseModel.deserialize(m_zbl.serialize())
    e1 = m_zbl.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    e2 = m2.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    np.testing.assert_allclose(e1, e2, rtol=1e-12)


def test_dense_route_raises_for_bridging():
    # The dense (nlist) route has no injection site for the term; silently
    # dropping it would be the dangerous direction, so it raises.
    m_zbl, _ = _models()
    coord, atype, box = _close_pair_inputs()
    with pytest.raises(NotImplementedError, match="NeighborGraph route only"):
        m_zbl.call_common(coord, atype, box=box, neighbor_graph_method="legacy")


def test_plain_serialize_has_no_bridging_key():
    # Absent key == disabled keeps pre-existing serialized dicts byte-stable.
    _, m_plain = _models()
    assert "bridging_method" not in m_plain.serialize()
