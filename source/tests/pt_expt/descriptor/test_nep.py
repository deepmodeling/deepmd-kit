# SPDX-License-Identifier: LGPL-3.0-or-later
import numpy as np
import pytest
import torch
from torch.fx.experimental.proxy_tensor import (
    make_fx,
)

from deepmd.dpmodel.descriptor import DescrptNep as DPDescrptNep
from deepmd.pt_expt.descriptor.nep import (
    DescrptNep,
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


class TestDescrptNep(TestCaseSingleFrameWithNlist):
    def setup_method(self) -> None:
        TestCaseSingleFrameWithNlist.setUp(self)
        self.device = env.DEVICE

    def _make(self, prec: str) -> DescrptNep:
        return DescrptNep(
            rcut_radial=self.rcut,
            rcut_angular=self.rcut * 0.8,
            sel=self.sel,
            n_max_radial=2,
            n_max_angular=2,
            basis_size_radial=3,
            basis_size_angular=3,
            l_max=2,
            l_max_4body=2,
            l_max_5body=1,
            precision=prec,
            seed=GLOBAL_SEED,
        ).to(self.device)

    @pytest.mark.parametrize("prec", ["float64", "float32"])
    def test_consistency(self, prec) -> None:
        rng = np.random.default_rng(GLOBAL_SEED)
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        err_msg = f"prec={prec}"

        dd0 = self._make(prec)
        dim = dd0.get_dim_out()
        davg = rng.normal(size=(dim,))
        dstd = 0.1 + np.abs(rng.normal(size=(dim,)))
        dd0.davg = torch.tensor(davg, dtype=dtype, device=self.device)
        dd0.dstd = torch.tensor(dstd, dtype=dtype, device=self.device)

        args = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        rd0, _, _, _, sw0 = dd0(*args)

        # serialize / deserialize round-trip
        dd1 = DescrptNep.deserialize(dd0.serialize())
        rd1, _, _, _, sw1 = dd1(*args)
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy(), rd1.detach().cpu().numpy(),
            rtol=rtol, atol=atol, err_msg=err_msg,
        )

        # permutation equivariance (frame 1 is a permutation of frame 0)
        np.testing.assert_allclose(
            rd0.detach().cpu().numpy()[0][self.perm[: self.nloc]],
            rd0.detach().cpu().numpy()[1],
            rtol=rtol, atol=atol, err_msg=err_msg,
        )

        # consistency with the dpmodel reference
        dd2 = DPDescrptNep.deserialize(dd0.serialize())
        rd2, _, _, _, sw2 = dd2.call(self.coord_ext, self.atype_ext, self.nlist)
        np.testing.assert_allclose(
            rd1.detach().cpu().numpy(), rd2, rtol=rtol, atol=atol, err_msg=err_msg
        )
        np.testing.assert_allclose(
            sw1.detach().cpu().numpy(), sw2, rtol=rtol, atol=atol, err_msg=err_msg
        )

    @pytest.mark.parametrize("prec", ["float64", "float32"])
    def test_exportable(self, prec) -> None:
        dtype = PRECISION_DICT[prec]
        dd0 = self._make(prec).eval()
        inputs = (
            torch.tensor(self.coord_ext, dtype=dtype, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        torch.export.export(dd0, inputs)

    @pytest.mark.parametrize("prec", ["float64"])
    def test_make_fx(self, prec) -> None:
        """Verify make_fx traces the forward pass together with autograd."""
        dtype = PRECISION_DICT[prec]
        rtol, atol = get_tols(prec)
        dd0 = self._make(prec).eval()

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
            rd_eager.detach().cpu().numpy(), rd_traced.detach().cpu().numpy(),
            rtol=rtol, atol=atol,
        )
        np.testing.assert_allclose(
            grad_eager.detach().cpu().numpy(), grad_traced.detach().cpu().numpy(),
            rtol=rtol, atol=atol,
        )

        # symbolic trace + export + .pte round-trip
        dynamic_shapes = make_descriptor_dynamic_shapes(has_mapping=False)
        export_save_load_and_compare(
            fn,
            (coord_ext, atype_ext, nlist),
            (rd_eager, grad_eager),
            dynamic_shapes,
            rtol=rtol,
            atol=atol,
        )

    def test_share_params(self) -> None:
        """share_params level 0 shares all coefficient modules and buffers."""
        rng = np.random.default_rng(GLOBAL_SEED)
        dd0 = self._make("float64")
        dd1 = DescrptNep(
            rcut_radial=self.rcut,
            rcut_angular=self.rcut * 0.8,
            sel=self.sel,
            n_max_radial=2,
            n_max_angular=2,
            basis_size_radial=3,
            basis_size_angular=3,
            l_max=2,
            l_max_4body=2,
            l_max_5body=1,
            precision="float64",
            seed=GLOBAL_SEED + 1,
        ).to(self.device)
        dim = dd0.get_dim_out()
        dd0.dstd = torch.tensor(
            0.1 + np.abs(rng.normal(size=(dim,))), dtype=torch.float64, device=self.device
        )
        dd1.share_params(dd0, shared_level=0)
        for key in dd0._modules:
            assert dd1._modules[key] is dd0._modules[key]
        for key in dd0._buffers:
            assert dd1._buffers[key] is dd0._buffers[key]
        inputs = (
            torch.tensor(self.coord_ext, dtype=torch.float64, device=self.device),
            torch.tensor(self.atype_ext, dtype=int, device=self.device),
            torch.tensor(self.nlist, dtype=int, device=self.device),
        )
        np.testing.assert_allclose(
            dd0(*inputs)[0].detach().cpu().numpy(),
            dd1(*inputs)[0].detach().cpu().numpy(),
        )
        with pytest.raises(NotImplementedError):
            dd1.share_params(dd0, shared_level=1)
