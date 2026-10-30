# SPDX-License-Identifier: LGPL-3.0-or-later
"""Cross-rank ghost exchange of a multi-block DPA4C, on one rank.

Reference = the folded periodic graph, whose edge sources are the owners, so
every source feature is complete by construction. Candidate = the unfolded
LAMMPS layout, owners followed by ghost images with edges for owned
destinations only, completed by a self-send ``border_op`` swap that maps every
ghost slot to its owner. The ghost rows receive no edges, so without the
exchange the second block would read zero moments for every neighbor across
the cell boundary.
"""

import copy
import ctypes

import numpy as np
import pytest
import torch

from deepmd.dpmodel.utils.neighbor_graph import (
    build_neighbor_graph,
)
from deepmd.dpmodel.utils.nlist import (
    build_neighbor_list,
    extend_coord_with_ghosts,
)
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils.comm import (
    ensure_comm_registered,
)

RCUT = 4.0
BOX = 6.0
ATYPE = np.array([[0, 0, 1, 1, 0]], dtype=np.int64)
COORD = np.array(
    [
        [
            [0.3, 3.0, 3.0],
            [5.4, 3.1, 2.8],
            [2.5, 3.0, 3.2],
            [3.6, 4.2, 3.0],
            [1.4, 1.1, 4.7],
        ]
    ],
    dtype=np.float64,
)
CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa4c",
        "rcut": [RCUT, 3.5],
        "n_blocks": 2,
        "channels": 16,
        "lmax": 3,
        "n_radial": 8,
        "radial_modes": 2,
        "precision": "float64",
        "seed": 7,
    },
    "fitting_net": {"neuron": [16, 16], "precision": "float64", "seed": 7},
}
ZBL_CONFIG = {
    **CONFIG,
    "type_map": ["O", "He"],
    "bridging_method": "zbl",
    "bridging_r_inner": 0.8,
    "bridging_r_outer": 1.2,
}
THREE_BLOCK_CONFIG = {
    **ZBL_CONFIG,
    "descriptor": {**CONFIG["descriptor"], "rcut": [RCUT, 3.5, 3.5], "n_blocks": 3},
}


def _perturb(model: torch.nn.Module) -> torch.nn.Module:
    """Move every weight off its initialization, the zero maps included."""
    generator = torch.Generator(device="cpu").manual_seed(11)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                noise = torch.randn(
                    parameter.shape, generator=generator, dtype=parameter.dtype
                )
                parameter.add_(0.3 * noise.to(parameter.device))
    return model


def _self_comm_dict(nloc: int, mapping: np.ndarray, keepalive: list) -> dict:
    """Return a one-swap self-send plan that copies every owner to its ghosts."""
    send_list = np.ascontiguousarray(mapping[nloc:], dtype=np.int32)
    keepalive.append(send_list)
    nghost = len(send_list)
    address = send_list.ctypes.data_as(ctypes.c_void_p).value
    return {
        "send_list": torch.tensor([address], dtype=torch.int64),
        "send_proc": torch.zeros(1, dtype=torch.int32),
        "recv_proc": torch.zeros(1, dtype=torch.int32),
        "send_num": torch.tensor([nghost], dtype=torch.int32),
        "recv_num": torch.tensor([nghost], dtype=torch.int32),
        "communicator": torch.zeros(1, dtype=torch.int64),
        "nlocal": torch.tensor(nloc, dtype=torch.int32),
        "nghost": torch.tensor(nghost, dtype=torch.int32),
    }


def _run_folded(model: torch.nn.Module, coord: np.ndarray) -> tuple:
    graph = build_neighbor_graph(
        coord, ATYPE, (BOX * np.eye(3)).reshape(1, 3, 3), RCUT, canonicalize=True
    )
    n_node = torch.as_tensor(np.asarray(graph.n_node), dtype=torch.int64)
    out = model.forward_common_lower_graph(
        torch.tensor(ATYPE.reshape(-1), dtype=torch.int64),
        n_node,
        n_node,
        torch.as_tensor(np.asarray(graph.edge_index), dtype=torch.int64),
        torch.as_tensor(np.asarray(graph.edge_vec), dtype=torch.float64),
        torch.as_tensor(np.asarray(graph.edge_mask), dtype=torch.bool),
    )
    energy = out["energy_redu"].detach().numpy().reshape(-1)
    force = -out["energy_derv_r"].detach().numpy().reshape(-1, 3)
    return energy, force


def _run_self_comm(model: torch.nn.Module, coord: np.ndarray) -> tuple:
    nloc = ATYPE.shape[1]
    ext_coord, ext_atype, mapping = extend_coord_with_ghosts(
        coord, ATYPE, (BOX * np.eye(3)).reshape(1, 9), RCUT
    )
    nlist = np.asarray(
        build_neighbor_list(
            ext_coord, ext_atype, nloc, RCUT, [256], distinguish_types=False
        )
    )[0]
    ext_coord = np.asarray(ext_coord).reshape(-1, 3)
    mapping = np.asarray(mapping)[0]
    valid = nlist >= 0
    src = nlist[valid].astype(np.int64)
    dst = np.repeat(np.arange(nloc, dtype=np.int64), nlist.shape[1])[valid.ravel()]
    keepalive: list = []
    out = model.forward_common_lower_graph(
        torch.tensor(np.asarray(ext_atype)[0], dtype=torch.int64),
        torch.tensor([ext_coord.shape[0]], dtype=torch.int64),
        torch.tensor([nloc], dtype=torch.int64),
        torch.tensor(np.stack([src, dst]), dtype=torch.int64),
        torch.tensor(ext_coord[src] - ext_coord[dst], dtype=torch.float64),
        torch.ones(len(src), dtype=torch.bool),
        comm_dict=_self_comm_dict(nloc, mapping, keepalive),
    )
    energy = out["energy_redu"].detach().numpy().reshape(-1)
    ghost_force = -out["energy_derv_r"].detach().numpy().reshape(-1, 3)
    force = np.zeros((nloc, 3))
    np.add.at(force, mapping, ghost_force)
    return energy, force


@pytest.mark.parametrize(
    "config",
    [CONFIG, ZBL_CONFIG, THREE_BLOCK_CONFIG],
    ids=["plain", "bridged", "three_blocks"],
)
class TestMultiBlockSelfComm:
    @pytest.fixture(autouse=True)
    def _setup(self, config: dict) -> None:
        ensure_comm_registered()
        self.model = _perturb(get_model(copy.deepcopy(config)).to("cpu").eval())
        # Atoms 0 and 1 face each other across the periodic x boundary at
        # 0.9 Å, inside the window of the bridged model.
        self.coord = COORD

    def test_exchange_matches_the_folded_graph(self) -> None:
        energy, force = _run_folded(self.model, self.coord)
        exchanged_energy, exchanged_force = _run_self_comm(self.model, self.coord)
        np.testing.assert_allclose(exchanged_energy, energy, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(exchanged_force, force, rtol=1e-11, atol=1e-11)

    def test_identity_exchange_diverges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without refreshing the ghost rows the parity breaks."""
        from deepmd.pt_expt.descriptor import dpa4c as module

        monkeypatch.setattr(
            module.DescrptDPA4C,
            "exchange_ghost_rows",
            lambda self, values, comm_dict: values,
        )
        energy, _ = _run_folded(self.model, self.coord)
        exchanged_energy, _ = _run_self_comm(self.model, self.coord)
        assert np.abs(exchanged_energy - energy).max() > 1e-6


def test_source_gate_needs_its_cross_rank_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The close pair straddles the boundary, so its gate partials do too."""
    from deepmd.pt_expt.descriptor import dpa4c as module

    ensure_comm_registered()
    model = _perturb(get_model(copy.deepcopy(ZBL_CONFIG)).to("cpu").eval())
    energy, _ = _run_folded(model, COORD)
    monkeypatch.setattr(
        module.DescrptDPA4C,
        "complete_node_partials",
        lambda self, partials, comm_dict: partials,
    )
    exchanged_energy, _ = _run_self_comm(model, COORD)
    assert np.abs(exchanged_energy - energy).max() > 1e-6


@pytest.mark.parametrize("config", [CONFIG, ZBL_CONFIG], ids=["plain", "bridged"])
def test_with_comm_export_traces_the_exchange(config: dict) -> None:
    """A multi-block model freezes a with-comm program that refreshes ghosts."""
    from deepmd.pt_expt.utils.serialization import (
        _needs_with_comm_artifact,
        _trace_and_export,
    )

    ensure_comm_registered()
    model = _perturb(get_model(copy.deepcopy(config)).to("cpu").eval())
    assert _needs_with_comm_artifact(model, lower_kind="graph")
    exported, _metadata, _model_json, _keys = _trace_and_export(
        {"model": model.serialize()},
        model_json_override=None,
        with_comm_dict=True,
        lower_kind="graph",
    )
    code = str(exported.module().code)
    assert "deepmd_export.border_op" in code
    if "bridging_method" in config:
        assert "deepmd_export.border_op_backward" in code


def test_single_block_needs_no_with_comm_artifact() -> None:
    single = copy.deepcopy(CONFIG)
    single["descriptor"].update(rcut=RCUT, n_blocks=1)
    from deepmd.pt_expt.utils.serialization import (
        _needs_with_comm_artifact,
    )

    assert not _needs_with_comm_artifact(get_model(single), lower_kind="graph")
