# SPDX-License-Identifier: LGPL-3.0-or-later
"""Issue #5906 Task 4: export-time questions are atomic-model capabilities.

Compositions must answer by aggregation, never by wrapper type or by
reaching into a single ``.descriptor``.
"""

import copy

import pytest

from deepmd.dpmodel.atomic_model import (
    DPAtomicModel,
)
from deepmd.dpmodel.atomic_model.inter_potential import (
    InterPotentialAtomicModel,
)
from deepmd.dpmodel.atomic_model.linear_atomic_model import (
    LinearEnergyAtomicModel,
)
from deepmd.dpmodel.descriptor import (
    DescrptSeA,
)
from deepmd.dpmodel.descriptor.dpa4 import (
    DescrptDPA4,
)
from deepmd.dpmodel.fitting import (
    InvarFitting,
)
from deepmd.dpmodel.model.model import (
    get_model,
)

from .test_zbl_bridging import (
    ZBL_CONFIG,
)

TYPE_MAP = ["Ni", "O"]


def _inter_potential() -> InterPotentialAtomicModel:
    """Simplest concrete BaseAtomicModel: no descriptor at all."""
    return InterPotentialAtomicModel(type_map=TYPE_MAP, rcut=4.0, sel=[8])


def _dp_atomic_model(descriptor) -> DPAtomicModel:
    fitting = InvarFitting(
        "energy",
        len(TYPE_MAP),
        descriptor.get_dim_out(),
        1,
        mixed_types=descriptor.mixed_types(),
    )
    return DPAtomicModel(descriptor, fitting, type_map=TYPE_MAP)


def _dpa4_descriptor(bridging: bool) -> DescrptDPA4:
    kwargs = {
        "ntypes": len(TYPE_MAP),
        "sel": 8,
        "rcut": 4.0,
        "channels": 16,
        "n_radial": 8,
        "lmax": 2,
        "mmax": 1,
        "n_blocks": 2,
        "precision": "float64",
        "seed": 7,
        "random_gamma": False,
    }
    if bridging:
        kwargs.update(inner_clamp_r_inner=0.8, inner_clamp_r_outer=1.2)
    return DescrptDPA4(**kwargs)


@pytest.mark.parametrize(
    "cap,default",
    [
        ("has_message_passing_across_ranks", False),  # needs-exchange: opt-in
        ("supports_edge_parallel", True),  # nothing to veto by default
        ("dense_lower_supports_comm", True),  # dense comm is the norm (dpa2/dpa3)
        ("uses_compact_edge_pairs", False),  # torch>=2.6 guard: opt-in
        ("supports_graph_export", True),  # only compression restricts export
    ],
)
def test_base_defaults(cap, default) -> None:
    """BaseAtomicModel answers each capability with a concrete default."""
    model = _inter_potential()
    assert getattr(model, cap)() is default


def test_base_graph_edge_dtype_default() -> None:
    """float64 is the model-agnostic edge-geometry ABI."""
    assert _inter_potential().graph_edge_dtype() == "float64"


def test_dp_atomic_model_delegates_to_descriptor() -> None:
    """DPAtomicModel answers every capability from its own descriptor."""
    bridged = _dp_atomic_model(_dpa4_descriptor(bridging=True))
    plain = _dp_atomic_model(_dpa4_descriptor(bridging=False))
    local_only = _dp_atomic_model(DescrptSeA(rcut=4.0, rcut_smth=3.5, sel=[8, 8]))
    assert bridged.has_message_passing_across_ranks() is True
    # bridged is True too since the SFPG cross-rank completion (issue
    # #5906); the ALL-aggregation's False branch is pinned by the stub
    # children in test_linear_aggregation_mixed_children.
    assert bridged.supports_edge_parallel() is True
    assert plain.has_message_passing_across_ranks() is True
    assert plain.supports_edge_parallel() is True
    assert local_only.has_message_passing_across_ranks() is False
    assert local_only.supports_edge_parallel() is True
    # dense-lower comm and graph-export ride the same delegation
    assert plain.dense_lower_supports_comm() is False  # DPA4 dense adapter raises
    assert local_only.dense_lower_supports_comm() is True
    assert plain.graph_edge_dtype() == "float64"
    assert plain.supports_graph_export() is True
    assert plain.uses_compact_edge_pairs() is bool(
        plain.descriptor.uses_compact_edge_pairs()
    )


class _EdgeParallelChild(InterPotentialAtomicModel):
    """Stub child with settable capabilities (test_zbl_bridging.py pattern)."""

    def __init__(
        self,
        *args,
        needs_exchange: bool = False,
        edge_parallel: bool = True,
        compact_pairs: bool = False,
        edge_dtype: str = "float64",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._needs_exchange = needs_exchange
        self._edge_parallel = edge_parallel
        self._compact_pairs = compact_pairs
        self._edge_dtype = edge_dtype

    def has_message_passing_across_ranks(self) -> bool:
        return self._needs_exchange

    def supports_edge_parallel(self) -> bool:
        return self._edge_parallel

    def uses_compact_edge_pairs(self) -> bool:
        return self._compact_pairs

    def graph_edge_dtype(self) -> str:
        return self._edge_dtype


def _stub_linear(**child_kwargs_pair) -> LinearEnergyAtomicModel:
    """Two stub children; child_kwargs_pair maps kwarg -> (child0, child1)."""
    kwargs0 = {k: v[0] for k, v in child_kwargs_pair.items()}
    kwargs1 = {k: v[1] for k, v in child_kwargs_pair.items()}
    return LinearEnergyAtomicModel(
        [
            _EdgeParallelChild(type_map=TYPE_MAP, rcut=4.0, sel=[8], **kwargs0),
            _EdgeParallelChild(type_map=TYPE_MAP, rcut=4.0, sel=[8], **kwargs1),
        ],
        type_map=TYPE_MAP,
        weights="sum",
    )


def test_linear_aggregation_any_all() -> None:
    """Real ZBL composition: the bridged DP child sets needs=True (any);
    edge-parallel aggregates to True since the SFPG cross-rank completion
    (issue #5906).
    """
    am = get_model(copy.deepcopy(ZBL_CONFIG)).atomic_model
    assert isinstance(am, LinearEnergyAtomicModel)
    assert am.has_message_passing_across_ranks() is True
    assert am.supports_edge_parallel() is True
    # ZBL rides the graph route with the learned child
    assert am.supports_graph_export() is True
    assert am.graph_edge_dtype() == "float64"


def test_linear_aggregation_mixed_children() -> None:
    """Both boolean branches of every any/all rule via stub children."""
    # has_message_passing_across_ranks: ANY
    assert (
        _stub_linear(needs_exchange=(True, False)).has_message_passing_across_ranks()
        is True
    )
    assert (
        _stub_linear(needs_exchange=(False, False)).has_message_passing_across_ranks()
        is False
    )
    # supports_edge_parallel: ALL (one veto vetoes all)
    assert _stub_linear(edge_parallel=(True, False)).supports_edge_parallel() is False
    assert _stub_linear(edge_parallel=(True, True)).supports_edge_parallel() is True
    # uses_compact_edge_pairs: ANY
    assert _stub_linear(compact_pairs=(True, False)).uses_compact_edge_pairs() is True
    assert _stub_linear(compact_pairs=(False, False)).uses_compact_edge_pairs() is False
    # graph_edge_dtype: float32 iff ALL children float32
    assert (
        _stub_linear(edge_dtype=("float32", "float64")).graph_edge_dtype() == "float64"
    )
    assert (
        _stub_linear(edge_dtype=("float32", "float32")).graph_edge_dtype() == "float32"
    )


def test_linear_aggregation_dense_comm_and_graph_export() -> None:
    """dense_lower_supports_comm and supports_graph_export are ALL rules."""

    class _NoDenseComm(InterPotentialAtomicModel):
        def dense_lower_supports_comm(self) -> bool:
            return False

    class _NoGraphExport(InterPotentialAtomicModel):
        def supports_graph_export(self) -> bool:
            return False

    plain = _inter_potential()
    mixed_comm = LinearEnergyAtomicModel(
        [_NoDenseComm(type_map=TYPE_MAP, rcut=4.0, sel=[8]), _inter_potential()],
        type_map=TYPE_MAP,
        weights="sum",
    )
    assert mixed_comm.dense_lower_supports_comm() is False
    all_comm = LinearEnergyAtomicModel(
        [plain, _inter_potential()], type_map=TYPE_MAP, weights="sum"
    )
    assert all_comm.dense_lower_supports_comm() is True
    mixed_export = LinearEnergyAtomicModel(
        [_NoGraphExport(type_map=TYPE_MAP, rcut=4.0, sel=[8]), _inter_potential()],
        type_map=TYPE_MAP,
        weights="sum",
    )
    assert mixed_export.supports_graph_export() is False
    assert all_comm.supports_graph_export() is True
