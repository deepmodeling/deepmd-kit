# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor.dpa3 import DescrptDPA3 as DPDescrptDPA3
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
from ..export_helpers import (
    export_save_load_and_compare,
    make_descriptor_dynamic_shapes,
)


class TestDescrptDPA3(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("ua", [True, False])  # update_angle
    @pytest.mark.parametrize("ruri", ["norm", "const"])  # update_residual_init
    @pytest.mark.parametrize("acr", [0, 1])  # a_compress_rate
    @pytest.mark.parametrize("acer", [1, 2])  # a_compress_e_rate
    @pytest.mark.parametrize("acus", [True, False])  # a_compress_use_split
    @pytest.mark.parametrize("nme", [1, 2])  # n_multi_edge_message
    def test_consistency(self, ua, ruri, acr, acer, acus, nme) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        # fixed parameters
        rus = "res_residual"  # update_style
        prec = "float64"  # precision
        ect = False  # use_econf_tebd

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        if prec == "float64":
            atol = 1e-8  # marginal test cases

        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            a_compress_rate=acr,
            a_compress_e_rate=acer,
            a_compress_use_split=acus,
            n_multi_edge_message=nme,
            axis_neuron=4,
            update_angle=ua,
            update_style=rus,
            update_residual_init=ruri,
            smooth_edge_update=True,
        )

        dd0 = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision=prec,
            use_econf_tebd=ect,
            type_map=["O", "H"] if ect else None,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        # serialization round-trip
        dd1 = DescrptDPA3.deserialize(dd0.serialize())
        rd1, _, _, _, _ = dd1(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd1.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        # dp impl
        dd2 = DPDescrptDPA3.deserialize(dd0.serialize())
        rd2, _, _, _, _ = dd2.call(
            self.coord_ext, self.atype_ext, self.nlist, self.mapping
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd2,
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_exportable(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]

        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            update_angle=True,
            update_style="res_residual",
            update_residual_init="const",
            smooth_edge_update=True,
        )

        dd0 = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0 = dd0.eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
            torch.tensor(self.mapping, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("ruri", ["norm", "const"])  # update_residual_init
    @pytest.mark.parametrize("acus", [True, False])  # a_compress_use_split
    @pytest.mark.parametrize("nme", [1, 2])  # n_multi_edge_message
    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_make_fx(self, ruri, acus, nme, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        if prec == "float64":
            atol = 1e-8

        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            a_compress_use_split=acus,
            n_multi_edge_message=nme,
            axis_neuron=4,
            update_angle=True,
            update_style="res_residual",
            update_residual_init=ruri,
            smooth_edge_update=True,
        )

        dd0 = DescrptDPA3(
            self.nt,
            repflow=repflow,
            exclude_types=[],
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)

        dd0.repflows.mean = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.repflows.stddev = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0 = dd0.eval()
        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)
        mapping = torch.tensor(self.mapping, dtype=int, device=self.device)

        def fn(coord_ext, atype_ext, nlist, mapping):
            coord_ext = coord_ext.detach().requires_grad_(True)
            rd = dd0(coord_ext, atype_ext, nlist, mapping)[0]
            grad = torch.autograd.grad(rd.sum(), coord_ext, create_graph=False)[0]
            return rd, grad

        rd_eager, grad_eager = fn(coord_ext, atype_ext, nlist, mapping)
        traced = make_fx(fn)(coord_ext, atype_ext, nlist, mapping)
        rd_traced, grad_traced = traced(coord_ext, atype_ext, nlist, mapping)
        np.testing.assert_allclose(
            rd_eager.detach().cpu().numpy(),
            rd_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(),
            grad_traced.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
        )

        # --- symbolic trace + export + .pte round-trip ---
        dynamic_shapes = make_descriptor_dynamic_shapes(has_mapping=True)
        inputs = (coord_ext, atype_ext, nlist, mapping)
        export_save_load_and_compare(
            fn,
            inputs,
            (rd_eager, grad_eager),
            dynamic_shapes,
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.parametrize("shared_level", [0, 1])  # sharing level
    def test_share_params(self, shared_level) -> None:
        """share_params level 0: share all; level 1: share type_embedding only."""
        rng = np.random.default_rng(GLOBAL_SEED)
        nf, nloc, nnei = self.nlist.shape
        davg0 = rng.normal(size=(self.nt, nnei, 4))
        dstd0 = 0.1 + np.abs(rng.normal(size=(self.nt, nnei, 4)))

        repflow = RepFlowArgs(
            n_dim=20,
            e_dim=10,
            a_dim=8,
            nlayers=3,
            e_rcut=self.rcut,
            e_rcut_smth=self.rcut_smth,
            e_sel=nnei,
            a_rcut=self.rcut - 0.1,
            a_rcut_smth=self.rcut_smth,
            a_sel=nnei - 1,
            axis_neuron=4,
            update_angle=True,
            update_style="res_residual",
            update_residual_init="const",
            smooth_edge_update=True,
        )

        dd0 = DescrptDPA3(
            self.nt, repflow=repflow, exclude_types=[], seed=GLOBAL_SEED
        ).to(self.device)
        dd1 = DescrptDPA3(
            self.nt, repflow=repflow, exclude_types=[], seed=GLOBAL_SEED + 1
        ).to(self.device)
        dd0.repflows.mean = torch.tensor(davg0, dtype=torch.float64, device=self.device)
        dd0.repflows.stddev = torch.tensor(
            dstd0, dtype=torch.float64, device=self.device
        )

        dd1.share_params(dd0, shared_level=shared_level)

        # type_embedding is always shared
        assert dd1._modules["type_embedding"] is dd0._modules["type_embedding"]

        if shared_level == 0:
            assert dd1._modules["repflows"] is dd0._modules["repflows"]
        elif shared_level == 1:
            assert dd1._modules["repflows"] is not dd0._modules["repflows"]

        # invalid level raises
        with pytest.raises(NotImplementedError):
            dd1.share_params(dd0, shared_level=2)


@pytest.mark.parametrize("use_loc_mapping", [True, False])
def test_has_message_passing_across_ranks(use_loc_mapping) -> None:
    """DPA3 always reports message passing; cross-rank only when
    ``use_loc_mapping=False`` (so per-layer node embeddings must flow
    via MPI ghost exchange instead of a local gather).
    """
    import copy

    from deepmd.dpmodel.model.model import (
        get_model,
    )

    config = {
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
            "use_loc_mapping": use_loc_mapping,
        },
        "fitting_net": {"neuron": [16, 16], "seed": 1},
    }
    model = get_model(copy.deepcopy(config))
    desc = model.atomic_model.descriptor
    assert desc.has_message_passing() is True
    assert desc.has_message_passing_across_ranks() is (not use_loc_mapping)
