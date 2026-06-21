# SPDX-License-Identifier: LGPL-3.0-or-later

import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor.hybrid import DescrptHybrid as DPDescrptHybrid
from deepmd.pt_expt.descriptor.hybrid import (
    DescrptHybrid,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.descriptor.se_r import (
    DescrptSeR,
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


class TestDescrptHybrid(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_consistency(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        err_msg = f"prec={prec}"

        ddsub0 = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        )
        ddsub1 = DescrptSeR(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        )
        dd0 = DescrptHybrid(
            list=[ddsub0, ddsub1],
        ).to(self.device)
        # set davg/dstd on sub-descriptors
        dd0.descrpt_list[0].davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.descrpt_list[0].dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0.descrpt_list[1].davg = torch.tensor(
            davg[..., :1], dtype=dtype, device=self.device
        )
        dd0.descrpt_list[1].dstd = torch.tensor(
            dstd[..., :1], dtype=dtype, device=self.device
        )
        rd0, _, _, _, _ = dd0(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        # serialization round-trip
        dd1 = DescrptHybrid.deserialize(dd0.serialize())
        rd1, _, _, _, _ = dd1(
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd1.detach().cpu().numpy(),
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )
        # dp impl
        dd2 = DPDescrptHybrid.deserialize(dd0.serialize())
        rd2, _, _, _, _ = dd2.call(
            self.coord_ext,
            self.atype_ext,
            self.nlist,
        )
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(),
            rd2,
            rtol=rtol,
            atol=atol,
            err_msg=err_msg,
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])  # precision
    def test_exportable(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]

        ddsub0 = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        )
        ddsub1 = DescrptSeR(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        )
        dd0 = DescrptHybrid(
            list=[ddsub0, ddsub1],
        ).to(self.device)
        dd0.descrpt_list[0].davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.descrpt_list[0].dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0.descrpt_list[1].davg = torch.tensor(
            davg[..., :1], dtype=dtype, device=self.device
        )
        dd0.descrpt_list[1].dstd = torch.tensor(
            dstd[..., :1], dtype=dtype, device=self.device
        )
        dd0 = dd0.eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("prec", ["float64"])  # precision
    def test_make_fx(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg = rng.normal(size=(self.nt, nnei, 4))
        dstd = rng.normal(size=(self.nt, nnei, 4))
        dstd = 0.1 + np.abs(dstd)

        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)

        ddsub0 = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        )
        ddsub1 = DescrptSeR(
            self.rcut,
            self.rcut_smth,
            self.sel,
            precision=prec,
            seed=GLOBAL_SEED,
        )
        dd0 = DescrptHybrid(
            list=[ddsub0, ddsub1],
        ).to(self.device)
        dd0.descrpt_list[0].davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.descrpt_list[0].dstd = torch.tensor(dstd, dtype=dtype, device=self.device)
        dd0.descrpt_list[1].davg = torch.tensor(
            davg[..., :1], dtype=dtype, device=self.device
        )
        dd0.descrpt_list[1].dstd = torch.tensor(
            dstd[..., :1], dtype=dtype, device=self.device
        )
        dd0 = dd0.eval()
        coord_ext = torch.tensor(self.coord_ext, dtype=dtype, device=self.device)
        atype_ext = torch.tensor(self.atype_ext, dtype=int, device=self.device)
        nlist = torch.tensor(self.nlist, dtype=int, device=self.device)

        def fn(coord_ext, atype_ext, nlist):
            coord_ext = coord_ext.detach().requires_grad_(True)
            rd = dd0(coord_ext, atype_ext, nlist)[0]
            grad = torch.autograd.grad(rd.sum(), coord_ext, create_graph=False)[0]
            return rd, grad

        rd_eager, grad_eager = fn(coord_ext, atype_ext, nlist)
        traced = make_fx(fn)(coord_ext, atype_ext, nlist)
        rd_traced, grad_traced = traced(coord_ext, atype_ext, nlist)
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
        dynamic_shapes = make_descriptor_dynamic_shapes(has_mapping=False)
        inputs = (coord_ext, atype_ext, nlist)
        export_save_load_and_compare(
            fn,
            inputs,
            (rd_eager, grad_eager),
            dynamic_shapes,
            rtol=rtol,
            atol=atol,
        )

    def test_share_params(self) -> None:
        """share_params level 0: recursively shares all sub-descriptors."""
        rng = np.random.default_rng(GLOBAL_SEED)
        _, _, nnei = self.nlist.shape
        davg4 = rng.normal(size=(self.nt, nnei, 4))
        dstd4 = 0.1 + np.abs(rng.normal(size=(self.nt, nnei, 4)))

        dd0 = DescrptHybrid(
            list=[
                DescrptSeA(self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED),
                DescrptSeR(self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED),
            ]
        ).to(self.device)
        dd1 = DescrptHybrid(
            list=[
                DescrptSeA(self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED + 1),
                DescrptSeR(self.rcut, self.rcut_smth, self.sel, seed=GLOBAL_SEED + 1),
            ]
        ).to(self.device)

        # set stats on dd0's sub-descriptors
        dd0.descrpt_list[0].davg = torch.tensor(
            davg4, dtype=torch.float64, device=self.device
        )
        dd0.descrpt_list[0].dstd = torch.tensor(
            dstd4, dtype=torch.float64, device=self.device
        )
        dd0.descrpt_list[1].davg = torch.tensor(
            davg4[..., :1], dtype=torch.float64, device=self.device
        )
        dd0.descrpt_list[1].dstd = torch.tensor(
            dstd4[..., :1], dtype=torch.float64, device=self.device
        )

        dd1.share_params(dd0, shared_level=0)

        # each sub-descriptor's modules/buffers are shared
        for ii in range(len(dd0.descrpt_list)):
            for key in dd0.descrpt_list[ii]._modules:
                assert (
                    dd1.descrpt_list[ii]._modules[key]
                    is dd0.descrpt_list[ii]._modules[key]
                )
            for key in dd0.descrpt_list[ii]._buffers:
                assert (
                    dd1.descrpt_list[ii]._buffers[key]
                    is dd0.descrpt_list[ii]._buffers[key]
                )

        # invalid level raises
        with pytest.raises(NotImplementedError):
            dd1.share_params(dd0, shared_level=1)


def _se_e2_a_child() -> dict:
    return {
        "type": "se_e2_a",
        "rcut": 6.0,
        "rcut_smth": 0.5,
        "sel": [20, 20],
        "neuron": [2, 4],
        "axis_neuron": 2,
        "type_one_side": True,
        "precision": "float64",
        "seed": 1,
    }


def _dpa3_child(use_loc_mapping: bool) -> dict:
    return {
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
    }


@pytest.mark.parametrize(
    "child_factory,expected_hmp,expected_hmp_ar",
    [
        (_se_e2_a_child, False, False),
        (lambda: _dpa3_child(use_loc_mapping=True), True, False),
        (lambda: _dpa3_child(use_loc_mapping=False), True, True),
    ],
    ids=["se_e2_a-only", "dpa3-ulm-true", "dpa3-ulm-false"],
)
def test_has_message_passing_across_ranks(
    child_factory, expected_hmp, expected_hmp_ar
) -> None:
    """Hybrid descriptor recurses into its children; cross-rank message
    passing is required iff any child needs it. Closes the structural
    side of catalog Tier-1 #1.
    """
    import copy

    from deepmd.dpmodel.model.model import (
        get_model,
    )

    config = {
        "type_map": ["O", "H"],
        "descriptor": {
            "type": "hybrid",
            "list": [child_factory()],
        },
        "fitting_net": {"neuron": [4, 4], "seed": 1},
    }
    desc = get_model(copy.deepcopy(config)).atomic_model.descriptor
    assert desc.has_message_passing() is expected_hmp
    assert desc.has_message_passing_across_ranks() is expected_hmp_ar
