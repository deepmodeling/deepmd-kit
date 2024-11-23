# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.dpmodel.output_def import (
    OutputVariableCategory,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.model import (
    EnergyModel,
    make_hessian_model,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
    to_torch_tensor,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = torch.float64


def finite_hessian(f, x, delta=1e-6):
    in_shape = x.shape
    assert len(in_shape) == 1
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape + in_shape)
    for iidx in np.ndindex(*in_shape):
        for jidx in np.ndindex(*in_shape):
            i0 = np.zeros(in_shape)
            i1 = np.zeros(in_shape)
            i2 = np.zeros(in_shape)
            i3 = np.zeros(in_shape)
            i0[iidx] += delta
            i2[iidx] += delta
            i1[iidx] -= delta
            i3[iidx] -= delta
            i0[jidx] += delta
            i1[jidx] += delta
            i2[jidx] -= delta
            i3[jidx] -= delta
            y0 = f(x + i0)
            y1 = f(x + i1)
            y2 = f(x + i2)
            y3 = f(x + i3)
            res[(Ellipsis, *iidx, *jidx)] = (y0 + y3 - y1 - y2) / (4 * delta**2.0)
    return res


class HessianTest:
    def test(
        self,
    ) -> None:
        # setup test case
        places = 6
        delta = 1e-3
        natoms = self.nloc
        nf = self.nf
        nv = self.nv
        generator = torch.Generator(device=env.DEVICE).manual_seed(GLOBAL_SEED)
        cell0 = torch.rand([3, 3], dtype=dtype, device=env.DEVICE, generator=generator)
        cell0 = 1.0 * (cell0 + cell0.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        cell1 = torch.rand([3, 3], dtype=dtype, device=env.DEVICE, generator=generator)
        cell1 = 1.0 * (cell1 + cell1.T) + 5.0 * torch.eye(3, device=env.DEVICE)
        cell = torch.stack([cell0, cell1])
        coord = torch.rand(
            [nf, natoms, 3], dtype=dtype, device=env.DEVICE, generator=generator
        )
        coord = torch.matmul(coord, cell)
        cell = cell.view([nf, 9])
        coord = coord.view([nf, natoms * 3])
        atype = (
            torch.stack(
                [
                    torch.IntTensor([0, 0, 1]),
                    torch.IntTensor([1, 0, 1]),
                ]
            )
            .view([nf, natoms])
            .to(env.DEVICE)
        )
        nfp, nap = 2, 3
        fparam = torch.rand(
            [nf, nfp], dtype=dtype, device=env.DEVICE, generator=generator
        )
        aparam = torch.rand(
            [nf, natoms * nap], dtype=dtype, device=env.DEVICE, generator=generator
        )
        # forward hess and value models
        ret_dict0 = self.model_hess.forward_common(
            coord, atype, box=cell, fparam=fparam, aparam=aparam
        )
        ret_dict1 = self.model_valu.forward_common(
            coord, atype, box=cell, fparam=fparam, aparam=aparam
        )
        # compare hess and value models
        torch.testing.assert_close(ret_dict0["energy"], ret_dict1["energy"])
        ana_hess = ret_dict0["energy_derv_r_derv_r"]

        # compute finite difference
        fnt_hess = []
        for ii in range(nf):

            def np_infer(
                xx,
            ):
                ret = self.model_valu.forward_common(
                    to_torch_tensor(xx).unsqueeze(0),
                    atype[ii].unsqueeze(0),
                    box=cell[ii].unsqueeze(0),
                    fparam=fparam[ii].unsqueeze(0),
                    aparam=aparam[ii].unsqueeze(0),
                )
                # detach
                ret = {kk: to_numpy_array(ret[kk]) for kk in ret}
                return ret

            def ff(xx):
                return np_infer(xx)["energy_redu"]

            xx = to_numpy_array(coord[ii])
            fnt_hess.append(finite_hessian(ff, xx, delta=delta).squeeze())

        # compare finite difference with autodiff
        fnt_hess = np.stack(fnt_hess).reshape([nf, nv, natoms * 3, natoms * 3])
        np.testing.assert_almost_equal(
            fnt_hess, to_numpy_array(ana_hess), decimal=places
        )


class TestDPModel(unittest.TestCase, HessianTest):
    def setUp(self) -> None:
        torch.manual_seed(2)
        self.nf = 2
        self.nloc = 3
        self.rcut = 4.0
        self.rcut_smth = 3.0
        self.sel = [10, 10]
        self.nt = 2
        self.nv = 2
        ds = DescrptSeA(
            self.rcut,
            self.rcut_smth,
            self.sel,
            neuron=[2, 4, 8],
            axis_neuron=2,
        ).to(env.DEVICE)
        ft0 = InvarFitting(
            "energy",
            self.nt,
            ds.get_dim_out(),
            self.nv,
            mixed_types=ds.mixed_types(),
            do_hessian=True,
            neuron=[4, 4, 4],
        ).to(env.DEVICE)
        type_map = ["foo", "bar"]
        self.model_hess = make_hessian_model(EnergyModel)(
            ds, ft0, type_map=type_map
        ).to(env.DEVICE)
        self.model_valu = EnergyModel.deserialize(self.model_hess.serialize())
        self.model_hess.requires_hessian("energy")

    def test_output_def(self) -> None:
        self.assertTrue(self.model_hess.atomic_output_def()["energy"].r_hessian)
        self.assertFalse(self.model_valu.atomic_output_def()["energy"].r_hessian)
        self.assertTrue(self.model_hess.model_output_def()["energy"].r_hessian)
        self.assertEqual(
            self.model_hess.model_output_def()["energy_derv_r_derv_r"].category,
            OutputVariableCategory.DERV_R_DERV_R,
        )
