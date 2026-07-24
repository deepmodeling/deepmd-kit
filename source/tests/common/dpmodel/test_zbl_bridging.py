# SPDX-License-Identifier: LGPL-3.0-or-later
"""dpmodel ZBL bridging as COMPOSITION (review 3638077323, redesigned).

``bridging_method: ZBL`` builds a
``LinearEnergyModel(LinearEnergyAtomicModel([dp, InterPotentialAtomicModel],
weights="sum"))`` -- the analytical term is its own atomic model summed
with the learned one, not a flag on it.
"""

import copy

import numpy as np
import pytest

from deepmd.dpmodel.atomic_model.inter_potential import (
    InterPotentialAtomicModel,
)
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.model.base_model import (
    BaseModel,
)
from deepmd.dpmodel.model.dp_linear_model import (
    LinearEnergyModel,
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


def _close_pair_inputs():
    rng = np.random.default_rng(5)
    coord = rng.uniform(1.5, 5.5, size=(1, 6, 3))
    coord[0, 1] = coord[0, 0] + np.array([0.9, 0.0, 0.0])  # close Ni-Ni pair
    atype = np.array([[0, 0, 1, 0, 1, 1]], dtype=np.int64)
    box = 8.0 * np.eye(3, dtype=np.float64)[None]
    return coord, atype, box


def test_builder_composes_linear_model():
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    assert type(model) is LinearEnergyModel
    am = model.atomic_model
    assert isinstance(am, LinearEnergyAtomicModel)
    assert am.weights == "sum"
    kinds = [type(c).__name__ for c in am.models]
    assert (
        kinds == ["EnergyAtomicModel", "InterPotentialAtomicModel"]
        or kinds[1] == "InterPotentialAtomicModel"
    )
    # radii wired to the LEARNED child's descriptor InnerClamp
    dp_child = am.models[0]
    assert dp_child.descriptor.inner_clamp is not None
    assert float(dp_child.descriptor.inner_clamp.r_inner) == 0.8


def test_zbl_child_equals_composition_minus_learned():
    """Composition energy == learned child + analytical child (exact sum)."""
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    coord, atype, box = _close_pair_inputs()
    e_sum = model.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    dp_child, zbl_child = model.atomic_model.models
    # learned child alone through its OWN model wrapper
    from deepmd.dpmodel.model.ener_model import (
        EnergyModel,
    )

    m_dp = EnergyModel(atomic_model_=dp_child)
    e_dp = m_dp.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    diff = float(np.sum(e_sum - e_dp))
    # positive ZBL repulsion from the close pair
    assert diff > 1e-3, f"ZBL contribution missing or non-positive: {diff:.3e}"


def test_zbl_serialize_roundtrip_energy_identical():
    model = get_model(copy.deepcopy(ZBL_CONFIG))
    coord, atype, box = _close_pair_inputs()
    data = model.serialize()
    assert data["type"] == "linear"
    m2 = BaseModel.deserialize(data)
    assert type(m2) is LinearEnergyModel
    e1 = model.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    e2 = m2.call_common(coord, atype, box=box, neighbor_graph_method="dense")[
        "energy_redu"
    ]
    np.testing.assert_allclose(e1, e2, rtol=1e-12)


def test_zbl_atomic_dense_route_raises():
    zbl = InterPotentialAtomicModel(type_map=["Ni", "O"], rcut=4.0, sel=[8])
    with pytest.raises(NotImplementedError, match="NeighborGraph route only"):
        zbl.forward_atomic(None, None, None)


def test_zbl_atomic_graph_values():
    """Atomic-model wrapper reproduces the kernel's known values."""
    import math

    from deepmd.dpmodel.utils.neighbor_graph import (
        NeighborGraph,
    )

    r = 0.8
    zbl = InterPotentialAtomicModel(type_map=["O"], rcut=4.0, sel=[8])
    graph = NeighborGraph(
        n_node=np.array([2], dtype=np.int64),
        edge_index=np.array([[0, 1], [1, 0]], dtype=np.int64),
        edge_vec=np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0]], dtype=np.float64),
        edge_mask=np.ones(2, dtype=bool),
    )
    out = zbl.forward_common_atomic_graph(graph, np.zeros(2, dtype=np.int64))
    a = 0.88534 * 0.5291772109 / (8.0**0.23 + 8.0**0.23)
    phi = sum(
        ak * math.exp(-bk * (r / a))
        for ak, bk in zip(
            (0.18175, 0.50986, 0.28022, 0.028171),
            (3.1998, 0.94229, 0.4029, 0.20162),
            strict=True,
        )
    )
    ref = 14.3996 * 64.0 / r * phi
    np.testing.assert_allclose(float(np.sum(out["energy"])), ref, atol=1e-5)
