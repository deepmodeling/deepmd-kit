# SPDX-License-Identifier: LGPL-3.0-or-later
"""Eager parity test for the pt_expt RepFlow parallel-mode override.

Verifies that ``DescrptBlockRepflows._exchange_ghosts`` (the pt_expt
override) produces output identical to the dpmodel default
``_exchange_ghosts`` when the supplied ``comm_dict`` describes a
single-rank, self-only MPI exchange whose effect equals the per-layer
gather that the default does via ``mapping``.

This is a Phase 2.5 gate: it exercises the override code path *eagerly*
(no torch.export, no AOTInductor) before we attempt the export round
trip in Phase 3. End-to-end multi-rank validation is deferred to the
Phase 5 LAMMPS test (``test_lammps_dpa3_pt2_mpi``).

Implementation note: the underlying ``torch.ops.deepmd.border_op``
treats ``sendlist_tensor`` as a packed pointer-array (``int**``). We
build that pointer array using numpy contiguous int32 arrays and pack
their addresses into an int64 tensor.  In single-rank mode (no MPI
init) the C++ op enters the ``sendproc == me`` self-send branch and
performs an in-process memcpy from the sendlist-indexed rows into the
ghost slots — no MPI runtime needed.
"""

from __future__ import (
    annotations,
)

import ctypes

import numpy as np
import pytest
import torch

# Trigger registration of the deepmd_export::border_op opaque wrapper.
import deepmd.pt_expt.utils.comm  # noqa: F401
from deepmd.dpmodel.descriptor.dpa3 import (
    RepFlowArgs,
)
from deepmd.pt_expt.descriptor.dpa3 import (
    DescrptDPA3,
)
from deepmd.pt_expt.utils import (
    env,
)
from deepmd.pt_expt.utils.env import (
    PRECISION_DICT,
)

from ...common.test_mixins import (
    TestCaseSingleFrameWithNlist,
    get_tols,
)
from ...seed import (
    GLOBAL_SEED,
)

# ---------------------------------------------------------------------------
# Helpers for building the comm_dict tensors


def _addr_of(np_arr: np.ndarray) -> int:
    """Return the raw int address of a numpy array's data buffer."""
    return np_arr.ctypes.data_as(ctypes.c_void_p).value


def _build_self_comm_dict(
    *,
    nloc: int,
    nghost: int,
    sendlist_indices: np.ndarray,
    device: torch.device,
    keepalive: list,
) -> dict:
    """Build a comm_dict for a single-rank self-exchange.

    Parameters
    ----------
    nloc, nghost
        Atom counts; ``nall = nloc + nghost``.
    sendlist_indices
        int32 array of length ``nghost`` giving local indices to copy
        into successive ghost slots [nloc, nloc+1, ...].
    device
        Target torch device for tensors.
    keepalive
        List into which we store numpy buffers that must outlive the
        forward pass (their addresses are referenced by sendlist_tensor).
    """
    sendlist_indices = np.ascontiguousarray(sendlist_indices, dtype=np.int32)
    keepalive.append(sendlist_indices)
    nswap = 1
    addr = _addr_of(sendlist_indices)
    # int** packed as one int64 entry per swap.
    sendlist_tensor = torch.tensor([addr], dtype=torch.int64, device=device)
    sendproc = torch.zeros(nswap, dtype=torch.int32, device=device)
    recvproc = torch.zeros(nswap, dtype=torch.int32, device=device)
    sendnum = torch.tensor([nghost], dtype=torch.int32, device=device)
    recvnum = torch.tensor([nghost], dtype=torch.int32, device=device)
    communicator = torch.zeros(1, dtype=torch.int64, device=device)
    nlocal_ts = torch.tensor(nloc, dtype=torch.int32, device=device)
    nghost_ts = torch.tensor(nghost, dtype=torch.int32, device=device)
    return {
        "send_list": sendlist_tensor,
        "send_proc": sendproc,
        "recv_proc": recvproc,
        "send_num": sendnum,
        "recv_num": recvnum,
        "communicator": communicator,
        "nlocal": nlocal_ts,
        "nghost": nghost_ts,
    }


# ---------------------------------------------------------------------------


class TestRepflowParallel(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    # ``mapping_at_parallel`` toggles between two scenarios:
    #   - "with-mapping": parallel call still receives the mapping tensor
    #     (matches what pt's DeepPotPT.cc does in production).
    #   - "none-mapping": parallel call receives ``mapping=None`` so the
    #     dpmodel branches that gate on ``mapping is not None`` are
    #     exercised (the regular code path still uses mapping for the
    #     reference, which proves the comm_dict path's correctness
    #     does not depend on mapping when override consumes comm_dict).
    @pytest.mark.parametrize("mapping_at_parallel", ["with-mapping", "none-mapping"])
    @pytest.mark.parametrize(
        "prec", ["float64"]
    )  # precision (single is enough for parity)
    def test_parallel_matches_default(
        self,
        prec: str,
        mapping_at_parallel: str,
    ) -> None:
        """Override with comm_dict matching mapping must match default path."""
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)

        repflow = RepFlowArgs(
            n_dim=8,
            e_dim=6,
            a_dim=4,
            nlayers=2,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            update_angle=False,
            update_style="res_residual",
            update_residual_init="const",
            smooth_edge_update=True,
        )

        dd = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision=prec,
            use_econf_tebd=False,
            type_map=None,
            seed=GLOBAL_SEED,
            use_loc_mapping=False,  # need extended-region indexing for parity
        ).to(self.device)
        dd.repflows.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)

        # use only the first frame to keep the test simple — single rank,
        # one frame, simple mapping ([0, 1, 2, 0]: ghost atom 3 mirrors local 0).
        coord_ext = torch.tensor(
            self.coord_ext[:1],
            dtype=dtype,
            device=self.device,
        )
        atype_ext = torch.tensor(
            self.atype_ext[:1],
            dtype=torch.int64,
            device=self.device,
        )
        nlist = torch.tensor(self.nlist[:1], dtype=torch.int64, device=self.device)
        mapping = torch.tensor(
            self.mapping[:1],
            dtype=torch.int64,
            device=self.device,
        )
        nall = self.nall

        # Default path (comm_dict=None) — uses gather via mapping.
        rd_default, _, _, _, _ = dd(coord_ext, atype_ext, nlist, mapping)

        # Parallel path: build a comm_dict whose sendlist mirrors the
        # extended portion of mapping.  For each ghost slot ii in
        # [nloc, nall), border_op writes node_ebd[sendlist[ii - nloc]],
        # so sendlist must match mapping[nloc:nall].
        keepalive: list = []
        ghost_sources = self.mapping[0, nloc:].astype(np.int32)
        comm_dict = _build_self_comm_dict(
            nloc=nloc,
            nghost=nall - nloc,
            sendlist_indices=ghost_sources,
            device=self.device,
            keepalive=keepalive,
        )

        mapping_for_parallel = (
            mapping if mapping_at_parallel == "with-mapping" else None
        )
        rd_parallel, _, _, _, _ = dd(
            coord_ext,
            atype_ext,
            nlist,
            mapping_for_parallel,
            comm_dict=comm_dict,
        )

        np.testing.assert_allclose(
            rd_parallel.detach().cpu().numpy(),
            rd_default.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )

    def test_use_loc_mapping_with_comm_dict_raises(self) -> None:
        """``use_loc_mapping=True`` + ``comm_dict`` is contradictory.

        The local-mapping codepath skips per-layer ghost exchange
        entirely, so combining it with ``comm_dict`` would silently
        drop the parallel behaviour.  Verify the override raises a
        clear error rather than producing wrong output.
        """
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(rng.normal(size=(self.nt, nnei, 4)))

        repflow = RepFlowArgs(
            n_dim=8,
            e_dim=6,
            a_dim=4,
            nlayers=1,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            update_angle=False,
            update_style="res_residual",
            update_residual_init="const",
            smooth_edge_update=True,
        )
        dd = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision="float64",
            use_econf_tebd=False,
            type_map=None,
            seed=GLOBAL_SEED,
            use_loc_mapping=True,  # contradictory with comm_dict
        ).to(self.device)
        dd.repflows.mean = torch.tensor(davg, dtype=torch.float64, device=self.device)
        dd.repflows.stddev = torch.tensor(dstd, dtype=torch.float64, device=self.device)

        coord_ext = torch.tensor(
            self.coord_ext[:1],
            dtype=torch.float64,
            device=self.device,
        )
        atype_ext = torch.tensor(
            self.atype_ext[:1],
            dtype=torch.int64,
            device=self.device,
        )
        nlist = torch.tensor(self.nlist[:1], dtype=torch.int64, device=self.device)
        mapping = torch.tensor(
            self.mapping[:1],
            dtype=torch.int64,
            device=self.device,
        )

        keepalive: list = []
        ghost_sources = self.mapping[0, nloc:].astype(np.int32)
        comm_dict = _build_self_comm_dict(
            nloc=nloc,
            nghost=self.nall - nloc,
            sendlist_indices=ghost_sources,
            device=self.device,
            keepalive=keepalive,
        )

        with pytest.raises(RuntimeError, match="use_loc_mapping=True"):
            dd(coord_ext, atype_ext, nlist, mapping, comm_dict=comm_dict)

    def test_spin_branch_runs(self) -> None:
        """Structural test for the ``has_spin`` branch of _exchange_ghosts.

        Builds a synthetic input that satisfies the spin path's atom-
        doubling invariant (``nloc`` and ``nall`` even), invokes the
        override directly with ``comm_dict["has_spin"]`` set, and
        verifies the output shape matches the input.  This catches
        regressions in the split-real-virtual + concat_switch_virtual
        code path without requiring a full spin model.
        """
        from deepmd.pt_expt.descriptor.repflows import (
            DescrptBlockRepflows,
        )

        # Build a minimally-initialised block instance via deserialize
        # of a tiny dpmodel block. We just need an instance to call
        # the method on; method behaviour is independent of weights.
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(rng.normal(size=(self.nt, nnei, 4)))

        repflow = RepFlowArgs(
            n_dim=8,
            e_dim=6,
            a_dim=4,
            nlayers=1,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            update_angle=False,
            update_style="res_residual",
            update_residual_init="const",
            smooth_edge_update=True,
        )
        dd = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision="float64",
            use_econf_tebd=False,
            type_map=None,
            seed=GLOBAL_SEED,
            use_loc_mapping=False,
        ).to(self.device)
        dd.repflows.mean = torch.tensor(davg, dtype=torch.float64, device=self.device)
        dd.repflows.stddev = torch.tensor(dstd, dtype=torch.float64, device=self.device)
        block = dd.repflows
        assert isinstance(block, DescrptBlockRepflows)

        # Pseudo-spin shapes: nloc and nall are even; n_dim from the
        # model. The spin path splits along dim 1 into real/virtual
        # halves and concats along dim 2.
        n_dim = block.n_dim
        nloc_spin, nghost_spin = 4, 2
        nall_spin = nloc_spin + nghost_spin
        # node_ebd: (1, nloc_spin, n_dim)
        node_ebd = torch.randn(
            1,
            nloc_spin,
            n_dim,
            dtype=torch.float64,
            device=self.device,
        )

        keepalive: list = []
        # sendlist mirrors local-to-ghost slot for one ghost rank.
        # Real ghost slots are real_nall-real_nloc = 1 atoms -> sendlist
        # has 1 entry. Self-send branch will copy local index 0.
        sendlist_indices = np.array([0], dtype=np.int32)
        comm_dict = _build_self_comm_dict(
            nloc=nloc_spin // 2,
            nghost=nghost_spin // 2,
            sendlist_indices=sendlist_indices,
            device=self.device,
            keepalive=keepalive,
        )
        comm_dict["has_spin"] = torch.tensor(
            [1],
            dtype=torch.int32,
            device=self.device,
        )

        # Direct invocation of _exchange_ghosts on the block.
        out = block._exchange_ghosts(
            node_ebd,
            mapping_tiled=None,
            comm_dict=comm_dict,
            nall=nall_spin,
            nloc=nloc_spin,
        )
        # concat_switch_virtual produces a tensor of shape
        # (1, nall_spin, n_dim) — 4 real + 2 virtual + 2 ghost-real +
        # 2 ghost-virtual interleaved per the helper's contract.
        # The exact structure is: out[1] dim is doubled relative to the
        # real_nall (real_nloc + real_nghost = 3); for nloc_spin=4,
        # nall_spin=6, the helper outputs 2*real_nall = 6 rows.
        assert out.shape[0] == 1
        assert out.shape[2] == n_dim
        # Spin path returns shape (1, 2*real_nall, n_dim) = (1, nall_spin, n_dim).
        assert out.shape[1] == nall_spin
