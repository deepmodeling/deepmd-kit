# SPDX-License-Identifier: LGPL-3.0-or-later
"""Phase 3 round-trip test for the with-comm AOTInductor artifact.

For a GNN model (DPA3 here), ``deserialize_to_file`` produces a .pt2
archive containing TWO compiled artifacts:
  * the regular forward_lower (no comm), packed at the top of the ZIP.
  * a ``forward_lower_with_comm`` variant nested at
    ``extra/forward_lower_with_comm.pt2``.

This test verifies:
  1. Both artifacts are present in the archive.
  2. ``metadata.json`` carries the ``has_comm_artifact`` flag.
  3. The with-comm artifact loads via ``aoti_load_package`` and runs
     when fed valid comm-dict tensors built via the ctypes pointer
     trick (see ``test_repflow_parallel.py``).
  4. The with-comm artifact's output matches the regular artifact's
     output for a single-rank self-exchange whose effect is identity
     (sendlist mirrors the extended-region mapping, which is what the
     gather in the regular path produces).
"""

from __future__ import (
    annotations,
)

import ctypes
import json
import os
import tempfile
import zipfile

import numpy as np
import pytest
import torch

# Trigger registration of the deepmd_export::border_op opaque wrapper
# (needed by the with-comm artifact at runtime).
import deepmd.pt_expt.utils.comm  # noqa: F401
from deepmd.pt_expt.model.get_model import (
    get_model,
)
from deepmd.pt_expt.utils.serialization import (
    _make_sample_inputs,
    deserialize_to_file,
)

_DPA3_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "dpa3",
        "repflow": {
            "n_dim": 8,
            "e_dim": 6,
            "a_dim": 4,
            "nlayers": 1,
            "e_rcut": 4.0,
            "e_rcut_smth": 0.5,
            "e_sel": 12,
            "a_rcut": 3.5,
            "a_rcut_smth": 0.5,
            "a_sel": 8,
            "axis_neuron": 4,
            "update_angle": False,
        },
        "use_loc_mapping": False,
    },
    "fitting_net": {"neuron": [16, 16], "seed": 1},
}


def _addr_of(np_arr: np.ndarray) -> int:
    return np_arr.ctypes.data_as(ctypes.c_void_p).value


def _build_self_comm_inputs(
    nloc: int,
    nghost: int,
    sendlist_indices: np.ndarray,
    keepalive: list,
) -> tuple[torch.Tensor, ...]:
    """Build runtime comm tensors for a single-rank self-send.

    Clamps the swap count to ``max(1, nghost)`` to mirror the trace-time
    helper in ``serialization.py::_make_comm_sample_inputs``; that
    avoids an empty sendlist pointer when a caller happens to construct
    a fixture with no ghost atoms.
    """
    send_count = max(1, nghost)
    sendlist_indices = np.ascontiguousarray(sendlist_indices, dtype=np.int32)
    if sendlist_indices.size == 0:
        sendlist_indices = np.zeros(send_count, dtype=np.int32)
    keepalive.append(sendlist_indices)
    nswap = 1
    addr = _addr_of(sendlist_indices)
    send_list = torch.tensor([addr], dtype=torch.int64)
    send_proc = torch.zeros(nswap, dtype=torch.int32)
    recv_proc = torch.zeros(nswap, dtype=torch.int32)
    send_num = torch.tensor([send_count], dtype=torch.int32)
    recv_num = torch.tensor([send_count], dtype=torch.int32)
    communicator = torch.zeros(1, dtype=torch.int64)
    nlocal_ts = torch.tensor(nloc, dtype=torch.int32)
    nghost_ts = torch.tensor(nghost, dtype=torch.int32)
    return (
        send_list,
        send_proc,
        recv_proc,
        send_num,
        recv_num,
        communicator,
        nlocal_ts,
        nghost_ts,
    )


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="AOTInductor compile is slow (~30s); run locally only by default.",
)
def test_pt2_dual_artifact_for_gnn(tmp_path) -> None:
    """End-to-end: GNN model produces dual-artifact .pt2; both load."""
    model = get_model(_DPA3_CONFIG)
    model.to("cpu")
    model.eval()

    # Serialize → deserialize_to_file (compiles and packs both artifacts)
    pt2_path = str(tmp_path / "test_dpa3.pt2")
    data = {"model": model.serialize()}
    deserialize_to_file(pt2_path, data)
    assert os.path.exists(pt2_path)

    # 1. ZIP layout sanity. PyTorch 2.11 strict layout puts our sidecars
    # under ``model/extra/`` (PT2_EXTRA_PREFIX); see serialization.py.
    with zipfile.ZipFile(pt2_path, "r") as zf:
        names = set(zf.namelist())
        meta = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
        assert "model/extra/forward_lower_with_comm.pt2" in names, (
            f"with-comm artifact missing; names={sorted(names)}"
        )
    assert meta["has_comm_artifact"] is True

    # 2. Both artifacts load.
    from torch._inductor import (
        aoti_load_package,
    )

    regular = aoti_load_package(pt2_path)

    with tempfile.TemporaryDirectory() as td:
        wc_path = os.path.join(td, "fl_wc.pt2")
        with zipfile.ZipFile(pt2_path, "r") as zf:
            with open(wc_path, "wb") as f:
                f.write(zf.read("model/extra/forward_lower_with_comm.pt2"))
        with_comm = aoti_load_package(wc_path)

    # 3. Run both artifacts with nframes=1 (matches what the with-comm
    # artifact requires; LAMMPS always passes one frame anyway).
    sample = _make_sample_inputs(model, nframes=1, has_spin=False)
    ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam = sample
    nloc = nlist_t.shape[1]
    nall = ext_atype.shape[1]
    nghost = nall - nloc

    out_regular = regular(ext_coord, ext_atype, nlist_t, mapping_t, fparam, aparam)

    # 4. Build runtime comm tensors mirroring the mapping (single-rank
    # self-send: ghost slot ii receives node[mapping[ii]], identical to
    # the gather in the regular path).
    keepalive: list = []
    ghost_sources = mapping_t[0, nloc:].cpu().numpy().astype(np.int32)
    comm_inputs = _build_self_comm_inputs(
        nloc=nloc,
        nghost=nghost,
        sendlist_indices=ghost_sources,
        keepalive=keepalive,
    )

    out_with_comm = with_comm(
        ext_coord,
        ext_atype,
        nlist_t,
        mapping_t,
        fparam,
        aparam,
        *comm_inputs,
    )

    # 5. Outputs must match (parity gate, eager-mode equivalent).
    for key in out_regular:
        np.testing.assert_allclose(
            out_with_comm[key].detach().cpu().numpy(),
            out_regular[key].detach().cpu().numpy(),
            rtol=0,
            atol=1e-10,
            err_msg=f"output[{key}] differs between regular and with-comm",
        )


# ---------------------------------------------------------------------------
# Coverage for previously-untested branches
# ---------------------------------------------------------------------------


def test_make_comm_sample_inputs_clamps_zero_nghost() -> None:
    """``_make_comm_sample_inputs(nghost=0)`` must produce valid tensors.

    The clamp ``send_count = max(1, nghost)`` ensures we never pass an
    empty pointer-array to border_op. This test exercises the
    ``nghost == 0`` branch (a model exported on a system whose entire
    domain fits in one rank with no ghosts) — the trace must still
    produce well-formed comm tensors of shape (1,).
    """
    from deepmd.pt_expt.utils.serialization import (
        _make_comm_sample_inputs,
    )

    comm_inputs = _make_comm_sample_inputs(
        nloc=4,
        nghost=0,
        device=torch.device("cpu"),
    )
    assert len(comm_inputs) == 8
    (
        send_list,
        send_proc,
        recv_proc,
        send_num,
        recv_num,
        communicator,
        nlocal,
        nghost_t,
    ) = comm_inputs
    # nswap stays at 1 (Phase 0: nswap=0 specializes during export).
    assert send_list.shape == (1,)
    assert send_proc.shape == (1,)
    assert recv_proc.shape == (1,)
    assert send_num.shape == (1,)
    assert recv_num.shape == (1,)
    # send_count is clamped to >=1, so send_num is also clamped.
    assert send_num.item() == 1
    assert recv_num.item() == 1
    # Scalar metadata reports the original (un-clamped) values.
    assert nlocal.item() == 4
    assert nghost_t.item() == 0


def test_needs_with_comm_artifact_for_hybrid_with_gnn() -> None:
    """``_needs_with_comm_artifact`` correctly reports True for hybrid
    descriptors whose children include a GNN block needing cross-rank
    message passing.

    The hybrid descriptor delegates ``has_message_passing_across_ranks()``
    to its children — if any child needs cross-rank message passing,
    the hybrid does too. ``_deserialize_to_file_pt2`` uses this gate
    to decide whether to compile the with-comm artifact, so the
    hybrid case must route correctly.
    """
    from deepmd.pt_expt.model.get_model import get_model as get_pt_expt_model
    from deepmd.pt_expt.utils.serialization import (
        _needs_with_comm_artifact,
    )

    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "hybrid",
            "list": [
                # Non-GNN child.
                {
                    "type": "se_e2_a",
                    "sel": [12, 12],
                    "rcut": 4.0,
                    "rcut_smth": 0.5,
                    "neuron": [4, 8],
                    "axis_neuron": 4,
                    "seed": 1,
                },
                # GNN child (DPA3).
                {
                    "type": "dpa3",
                    "repflow": {
                        "n_dim": 4,
                        "e_dim": 4,
                        "a_dim": 4,
                        "nlayers": 1,
                        "e_rcut": 4.0,
                        "e_rcut_smth": 0.5,
                        "e_sel": 8,
                        "a_rcut": 3.5,
                        "a_rcut_smth": 0.5,
                        "a_sel": 4,
                        "axis_neuron": 4,
                        "update_angle": False,
                    },
                    "use_loc_mapping": False,
                },
            ],
        },
        "fitting_net": {"neuron": [8, 8], "seed": 1},
    }
    model = get_pt_expt_model(config)
    model.to("cpu")
    model.eval()
    assert _needs_with_comm_artifact(model) is True, (
        "hybrid model with a use_loc_mapping=False GNN child must "
        "report has_message_passing_across_ranks=True so a with-comm "
        "artifact gets compiled"
    )


def test_pte_with_comm_dict_traces_and_loads(tmp_path) -> None:
    """``_trace_and_export(with_comm_dict=True)`` produces a valid
    ExportedProgram that can be saved as .pte and loaded back.

    .pte is Python-only (the multi-rank consumer is C++/LAMMPS via
    .pt2), so production has no business calling this path. But the
    trace machinery is the same as the .pt2 path, so .pte serves as
    a cheap (no AOTI compile) round-trip test for the with-comm
    export pipeline.
    """
    from deepmd.pt_expt.utils.serialization import (
        _trace_and_export,
    )

    model = get_model(_DPA3_CONFIG)
    model.to("cpu")
    model.eval()
    data = {"model": model.serialize()}

    exported, metadata, _data_for_json, output_keys = _trace_and_export(
        data,
        model_json_override=None,
        with_comm_dict=True,
    )
    # ``_trace_and_export(with_comm_dict=True)`` is the with-comm path
    # by construction; metadata at this layer no longer carries the
    # has_message_passing flag (only ``has_comm_artifact``, written
    # later in _deserialize_to_file_pt2). Sanity-check via output_keys
    # that the trace produced energy outputs.
    # output_keys mirrors what the regular trace would produce; at
    # least one energy-related key must be present.
    assert any(k.startswith("energy") for k in output_keys), (
        f"expected an 'energy*' output key; got {output_keys}"
    )

    # Save as .pte and reload — verifies the ExportedProgram is
    # structurally valid (no broken graph or missing constants).
    pte_path = str(tmp_path / "fl_with_comm.pte")
    torch.export.save(exported, pte_path)
    assert os.path.exists(pte_path)
    loaded = torch.export.load(pte_path)
    # Sanity: the loaded program has the expected number of inputs
    # (6 base + 8 comm = 14).
    spec = loaded.module().graph.find_nodes(op="placeholder")
    assert len(spec) == 14, (
        f"with-comm exported program must accept 14 positional inputs "
        f"(6 base + 8 comm); got {len(spec)}"
    )
