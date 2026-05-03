# SPDX-License-Identifier: LGPL-3.0-or-later
"""Eager parity test for the pt_expt Repformer parallel-mode override.

Mirror of ``test_repflow_parallel.py`` but for DPA2 (which uses
``DescrptBlockRepformers``).  Same single-rank self-exchange trick:
``sendlist`` mirrors ``mapping[nloc:]`` so the C++ ``border_op``'s
self-send branch reproduces the gather that the dpmodel default does.
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
from deepmd.dpmodel.descriptor.dpa2 import (
    RepformerArgs,
    RepinitArgs,
)
from deepmd.pt_expt.descriptor.dpa2 import (
    DescrptDPA2,
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


def _addr_of(np_arr: np.ndarray) -> int:
    return np_arr.ctypes.data_as(ctypes.c_void_p).value


def _build_self_comm_dict(
    *,
    nloc: int,
    nghost: int,
    sendlist_indices: np.ndarray,
    device: torch.device,
    keepalive: list,
) -> dict:
    """Control tensors must live on CPU because the C++ ``border_op``
    host code dereferences ``data_ptr<int>()`` directly.  Production
    builds them on CPU in
    ``commonPTExpt.h::build_comm_tensors_positional``; on a CUDA build
    a CUDA-device control tensor segfaults the host read.  See
    ``test_repflow_parallel.py::_build_self_comm_dict`` for the full
    rationale.
    """
    del device  # control tensors are always CPU
    sendlist_indices = np.ascontiguousarray(sendlist_indices, dtype=np.int32)
    keepalive.append(sendlist_indices)
    nswap = 1
    addr = _addr_of(sendlist_indices)
    sendlist_tensor = torch.tensor([addr], dtype=torch.int64, device="cpu")
    sendproc = torch.zeros(nswap, dtype=torch.int32, device="cpu")
    recvproc = torch.zeros(nswap, dtype=torch.int32, device="cpu")
    sendnum = torch.tensor([nghost], dtype=torch.int32, device="cpu")
    recvnum = torch.tensor([nghost], dtype=torch.int32, device="cpu")
    communicator = torch.zeros(1, dtype=torch.int64, device="cpu")
    nlocal_ts = torch.tensor(nloc, dtype=torch.int32, device="cpu")
    nghost_ts = torch.tensor(nghost, dtype=torch.int32, device="cpu")
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


class TestRepformerParallel(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    # See test_repflow_parallel.py for rationale on the "none-mapping"
    # variant — exercises dpa2's "skip pre-block gather" branch with
    # mapping=None, which is the realistic LAMMPS multi-rank shape.
    @pytest.mark.parametrize("mapping_at_parallel", ["with-mapping", "none-mapping"])
    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_parallel_matches_default(
        self,
        prec: str,
        mapping_at_parallel: str,
    ) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)
        davg_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = rng.normal(size=(self.nt, nnei // 2, 4))
        dstd_2 = 0.1 + np.abs(dstd_2)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        if prec == "float64":
            atol = 1e-8

        repinit = RepinitArgs(
            rcut=self.rcut,
            rcut_smth=self.rcut_smth,
            nsel=self.sel_mix,
            tebd_input_mode="concat",
            set_davg_zero=True,
        )
        repformer = RepformerArgs(
            rcut=self.rcut / 2,
            rcut_smth=self.rcut_smth,
            nsel=nnei // 2,
            nlayers=2,
            g1_dim=12,
            g2_dim=8,
            axis_neuron=4,
            update_g1_has_conv=True,
            update_g1_has_drrd=True,
            update_g1_has_grrg=True,
            update_g1_has_attn=True,
            update_g2_has_g1g1=True,
            update_g2_has_attn=True,
            update_h2=False,
            attn1_hidden=12,
            attn1_nhead=2,
            attn2_hidden=8,
            attn2_nhead=2,
            attn2_has_gate=False,
            update_style="res_avg",
            set_davg_zero=True,
            use_sqrt_nnei=False,
            g1_out_conv=False,
            g1_out_mlp=False,
        )

        dd = DescrptDPA2(
            self.nt,
            repinit=repinit,
            repformer=repformer,
            smooth=True,
            exclude_types=[],
            add_tebd_to_repinit_out=False,
            precision=prec,
            use_econf_tebd=False,
            type_map=None,
            seed=GLOBAL_SEED,
        ).to(self.device)
        dd.repinit.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd.repinit.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd.repformers.mean = torch.tensor(davg_2, dtype=dtype, device=self.device)
        dd.repformers.stddev = torch.tensor(dstd_2, dtype=dtype, device=self.device)

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

        rd_default, _, _, _, _ = dd(coord_ext, atype_ext, nlist, mapping)

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
