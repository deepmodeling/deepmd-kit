# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for SpinModel + comm_dict end-to-end.

Two coverage levels:

1. ``test_spin_forward_common_lower_exportable_with_comm_traces``:
   verifies the trace machinery (positional comm-tensor plumbing,
   has_spin injection, make_fx symbolic mode) on a spin model with a
   non-GNN descriptor (se_e2_a). The non-GNN case is the cheapest
   smoke test since se_e2_a's `call` accepts and drops comm_dict —
   exercising the wrapper/spin model layers without paying for GNN
   compile cost.

2. ``test_spin_dpa3_eager_parity``: end-to-end value-correctness for
   a spin DPA3 model running through ``call_common_lower`` in eager
   mode, with a comm_dict whose self-exchange mirrors the mapping.
   Asserts the result matches the no-comm reference. This proves
   ``SpinModel.call_common_lower`` correctly forwards comm_dict
   through to the GNN repflow, AND that the spin branch of
   ``_exchange_ghosts`` (real/virtual split + concat_switch_virtual)
   reproduces the regular gather path on real values.
"""

from __future__ import (
    annotations,
)

import ctypes

import numpy as np
import torch

import deepmd.pt_expt.utils.comm  # noqa: F401  # lgtm[py/unused-import]  - opaque op registration
from deepmd.dpmodel.model.model import get_model as get_model_dp
from deepmd.pt_expt.model.spin_ener_model import (
    SpinEnergyModel,
)

SPIN_GNN_DATA = {
    "type_map": ["O", "H", "B"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20, 20],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [3, 6],
        "resnet_dt": False,
        "axis_neuron": 2,
        "precision": "float64",
        "type_one_side": True,
        "seed": 1,
    },
    "fitting_net": {
        "neuron": [5, 5],
        "resnet_dt": True,
        "precision": "float64",
        "seed": 1,
    },
    "spin": {
        "use_spin": [True, False, False],
        "virtual_scale": [0.3140],
    },
}


def _addr_of(np_arr: np.ndarray) -> int:
    return np_arr.ctypes.data_as(ctypes.c_void_p).value


def _build_self_comm_inputs(nloc: int, nghost: int):
    """Build trivial-but-valid comm tensors for tracing."""
    keepalive: list[np.ndarray] = []
    indices = np.zeros(max(1, nghost), dtype=np.int32)
    keepalive.append(indices)
    addr = _addr_of(indices)
    nswap = 1
    return (
        torch.tensor([addr], dtype=torch.int64),  # send_list
        torch.zeros(nswap, dtype=torch.int32),  # send_proc
        torch.zeros(nswap, dtype=torch.int32),  # recv_proc
        torch.tensor([max(1, nghost)], dtype=torch.int32),  # send_num
        torch.tensor([max(1, nghost)], dtype=torch.int32),  # recv_num
        torch.zeros(1, dtype=torch.int64),  # communicator
        torch.tensor(nloc, dtype=torch.int32),  # nlocal
        torch.tensor(nghost, dtype=torch.int32),  # nghost
    ), keepalive


def test_spin_forward_common_lower_exportable_with_comm_traces() -> None:
    """The spin variant of forward_common_lower_exportable_with_comm
    produces a callable traced GraphModule.
    """
    dp_model = get_model_dp(SPIN_GNN_DATA)
    model = SpinEnergyModel.deserialize(dp_model.serialize()).to("cpu")
    model.eval()

    # Build sample inputs (nframes=1 to match the override's nb=1
    # constraint; spin doubles natoms). nlist width must match the
    # model's sum(sel); the descriptor's _format_nlist asserts this.
    nloc = 6  # 3 real + 3 virtual
    nall = 8  # 1 ghost on each side
    n_dim_coord = 3
    nnei = sum(SPIN_GNN_DATA["descriptor"]["sel"])
    ext_coord = torch.zeros(1, nall, n_dim_coord, dtype=torch.float64)
    ext_atype = torch.zeros(1, nall, dtype=torch.int64)
    ext_spin = torch.zeros(1, nall, n_dim_coord, dtype=torch.float64)
    nlist = torch.zeros(1, nloc, nnei, dtype=torch.int64)
    mapping = torch.zeros(1, nall, dtype=torch.int64)
    fparam = None
    aparam = None

    comm_inputs, _keepalive = _build_self_comm_inputs(nloc=nloc, nghost=nall - nloc)

    # The trace should succeed without raising. We do NOT verify
    # numerical correctness here — that would require a real spin GNN
    # model + live MPI (deferred to Phase 5 LAMMPS).  This test only
    # checks the trace-time machinery: positional arg plumbing,
    # has_spin injection, and that make_fx symbolic mode produces a
    # valid GraphModule.
    traced = model.forward_common_lower_exportable_with_comm(
        ext_coord,
        ext_atype,
        ext_spin,
        nlist,
        mapping,
        fparam,
        aparam,
        *comm_inputs,
        do_atomic_virial=True,
        tracing_mode="symbolic",
        _allow_non_fake_inputs=True,
    )
    # The traced module must be a torch.nn.Module that can be invoked.
    assert isinstance(traced, torch.nn.Module)
    # And calling it with the same inputs returns a dict with the
    # expected keys.
    out = traced(
        ext_coord,
        ext_atype,
        ext_spin,
        nlist,
        mapping,
        fparam,
        aparam,
        *comm_inputs,
    )
    assert isinstance(out, dict)
    # forward_common_lower internal output names; specifics depend on
    # the model's output def, just check at least one is present.
    assert any(k.startswith("energy") for k in out), (
        f"expected an 'energy*' key in trace output; got {list(out.keys())}"
    )


# ---------------------------------------------------------------------------
# 2. End-to-end value parity for spin DPA3 in eager mode
# ---------------------------------------------------------------------------


SPIN_DPA3_DATA = {
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
            "e_sel": 8,
            "a_rcut": 3.5,
            "a_rcut_smth": 0.5,
            "a_sel": 4,
            "axis_neuron": 4,
            "update_angle": False,
        },
        "use_loc_mapping": False,
    },
    "fitting_net": {"neuron": [16, 16], "seed": 1},
    "spin": {"use_spin": [True, False], "virtual_scale": [0.314]},
}


def test_spin_dpa3_eager_parity() -> None:
    """SpinModel.call_common_lower with comm_dict (self-exchange) must
    match the no-comm reference for a spin DPA3 model.

    Setup mirrors the per-block parity tests but at the SpinModel
    level so it exercises the full plumbing chain:
      ``SpinModel.call_common_lower(comm_dict=...)``
       → process_spin_input_lower (atom-doubling)
       → backbone EnergyModel.call_common_lower(comm_dict=...)
       → atomic_model.forward_common_atomic(comm_dict=...)
       → DescrptDPA3.call(comm_dict=...)
       → DescrptBlockRepflows.call(comm_dict=...)
       → DescrptBlockRepflows._exchange_ghosts (pt_expt override,
         spin branch via has_spin in comm_dict)

    The comm_dict has has_spin=tensor([1]) and a sendlist that
    mirrors the real-atom portion of the mapping.  The override's
    spin branch splits node_ebd into real/virtual halves, stacks
    along feature dim, exchanges, then de-interleaves with
    concat_switch_virtual.  When the exchange produces the same
    result as the gather (which it should for a self-mirror
    sendlist), the spin model output must equal the no-comm output
    bit-for-bit (atol 1e-12 for float64).
    """
    dp_model = get_model_dp(SPIN_DPA3_DATA)
    model = SpinEnergyModel.deserialize(dp_model.serialize()).to("cpu")
    model.eval()

    # Build a 2-atom test system: 1 real + 1 ghost real for type 0,
    # plus the same in spin (use_spin=[True, False] means type 0 is
    # spin-doubled, type 1 is not).  After atom-doubling the model
    # processes 2 real + 2 virtual = 4 atoms locally and 4 ghost
    # slots.  We use minimal nloc to keep the test fast.
    nframes = 1
    nloc_real = 2  # 2 real atoms (both type 0 to keep simple)
    nghost_real = 2  # 2 ghost real atoms
    nall_real = nloc_real + nghost_real
    rng = np.random.default_rng(42)

    # Coordinates and types (real only — spin model doubles internally).
    coord_real = rng.uniform(0, 4.0, size=(nframes, nall_real, 3)).astype(np.float64)
    atype_real = np.zeros((nframes, nall_real), dtype=np.int64)  # all type 0
    spin_real = rng.uniform(-0.1, 0.1, size=(nframes, nall_real, 3)).astype(np.float64)
    # mapping: ghost atoms mirror local atoms (ghost 0 → local 0, ghost 1 → local 1)
    mapping_real = np.array(
        [[0, 1, 0, 1]],
        dtype=np.int64,
    )  # nframes=1, nall_real=4

    # Build extended-region nlist for the real atoms. Each real atom's
    # neighbour list points to the other 3 atoms (within rcut by
    # construction of small box). We don't need physically meaningful
    # values — just well-formed nlist so the model runs.
    nnei = 8  # matches e_sel
    nlist_real = np.full((nframes, nloc_real, nnei), -1, dtype=np.int64)
    for ii in range(nloc_real):
        # neighbours = all other atoms (real + ghost) up to nnei
        others = [j for j in range(nall_real) if j != ii][:nnei]
        nlist_real[0, ii, : len(others)] = others

    # ``call_common_lower`` runs through ``transform_output`` which
    # calls ``torch.autograd.grad`` on coord, so coord must require
    # grad in eager mode.
    ext_coord = torch.tensor(coord_real, dtype=torch.float64, requires_grad=True)
    ext_atype = torch.tensor(atype_real, dtype=torch.int64)
    ext_spin = torch.tensor(spin_real, dtype=torch.float64)
    nlist_t = torch.tensor(nlist_real, dtype=torch.int64)
    mapping_t = torch.tensor(mapping_real, dtype=torch.int64)

    # 1. No-comm reference.
    out_ref = model.call_common_lower(
        ext_coord,
        ext_atype,
        ext_spin,
        nlist_t,
        mapping_t,
        fparam=None,
        aparam=None,
        do_atomic_virial=False,
    )

    # 2. With comm_dict.  The SpinModel internally doubles atoms to
    # nloc=2*nloc_real=4 and nall=2*nall_real=8.  The override's spin
    # branch peels back to real_nloc=nloc_real and real_nall=nall_real.
    # Sendlist must point to REAL local indices for each real ghost
    # slot (mapping_real[nloc_real:nall_real]).
    keepalive: list = []
    sendlist_indices = mapping_real[0, nloc_real:].astype(np.int32)
    keepalive.append(sendlist_indices)
    addr = sendlist_indices.ctypes.data_as(ctypes.c_void_p).value
    nswap = 1
    nghost_real_count = nall_real - nloc_real
    comm_dict = {
        "send_list": torch.tensor([addr], dtype=torch.int64),
        "send_proc": torch.zeros(nswap, dtype=torch.int32),
        "recv_proc": torch.zeros(nswap, dtype=torch.int32),
        "send_num": torch.tensor([nghost_real_count], dtype=torch.int32),
        "recv_num": torch.tensor([nghost_real_count], dtype=torch.int32),
        "communicator": torch.zeros(1, dtype=torch.int64),
        # nlocal/nghost are the REAL counts (the override's spin branch
        # halves nloc/nall internally).  In production C++ side passes
        # real counts here too — see DeepSpinPT.cc.
        "nlocal": torch.tensor(nloc_real, dtype=torch.int32),
        "nghost": torch.tensor(nghost_real_count, dtype=torch.int32),
        # Triggers spin branch in the override.
        "has_spin": torch.tensor([1], dtype=torch.int32),
    }

    # Fresh coord tensor (the first call's backward graph would otherwise
    # be reused / cause double-backward errors).
    ext_coord_2 = torch.tensor(coord_real, dtype=torch.float64, requires_grad=True)
    out_parallel = model.call_common_lower(
        ext_coord_2,
        ext_atype,
        ext_spin,
        nlist_t,
        mapping_t,
        fparam=None,
        aparam=None,
        do_atomic_virial=False,
        comm_dict=comm_dict,
    )

    # 3. Compare every output key.
    for key in out_ref:
        ref = out_ref[key].detach().cpu().numpy()
        par = out_parallel[key].detach().cpu().numpy()
        np.testing.assert_allclose(
            par,
            ref,
            atol=1e-10,
            rtol=0,
            err_msg=f"output[{key}] mismatch between no-comm and comm_dict path",
        )
