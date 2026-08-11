# SPDX-License-Identifier: LGPL-3.0-or-later
"""Pin the ``pair_exclude_types``/``pair_excl`` construction-path contract
(issue #5897 task 5): ``BaseAtomicModel.__init__`` (via
``reinit_pair_exclude``) always sets both attributes, on every construction
path (direct ``__init__`` and ``deserialize``); ``get_pair_exclude_types()``
is the public accessor, ``pair_excl`` a pinned direct-access attribute.
"""

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)

RCUT = 2.2
RCUT_SMTH = 0.4
SEL = [5, 2]
NTYPES = 2
TYPE_MAP = ["foo", "bar"]


def _make_minimal_atomic_model(
    pair_exclude_types: list[tuple[int, int]],
) -> DPAtomicModel:
    ds = DescrptSeA(
        RCUT,
        RCUT_SMTH,
        SEL,
    )
    ft = InvarFitting(
        "energy",
        NTYPES,
        ds.get_dim_out(),
        1,
        mixed_types=ds.mixed_types(),
    )
    return DPAtomicModel(
        ds,
        ft,
        type_map=TYPE_MAP,
        pair_exclude_types=pair_exclude_types,
    )


def test_pair_excl_exists_after_init_and_deserialize() -> None:
    md0 = _make_minimal_atomic_model(pair_exclude_types=[(0, 1)])
    assert md0.get_pair_exclude_types() == [(0, 1)]
    assert md0.pair_excl is not None
    md1 = type(md0).deserialize(md0.serialize())
    assert md1.get_pair_exclude_types() == [(0, 1)]
    assert md1.pair_excl is not None
    md2 = _make_minimal_atomic_model(pair_exclude_types=[])
    assert md2.get_pair_exclude_types() == []
    assert md2.pair_excl is None
