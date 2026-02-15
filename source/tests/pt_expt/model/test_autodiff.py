# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np
import torch

from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    InvarFitting,
)
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.utils import (
    env,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = torch.float64


def finite_difference(f, x, delta=1e-6):
    in_shape = x.shape
    y0 = f(x)
    out_shape = y0.shape
    res = np.empty(out_shape + in_shape)
    for idx in np.ndindex(*in_shape):
        diff = np.zeros(in_shape)
        diff[idx] += delta
        y1p = f(x + diff)
        y1n = f(x - diff)
        res[(Ellipsis, *idx)] = (y1p - y1n) / (2 * delta)
    return res


def stretch_box(old_coord, old_box, new_box):
    ocoord = old_coord.reshape(-1, 3)
    obox = old_box.reshape(3, 3)
    nbox = new_box.reshape(3, 3)
    ncoord = ocoord @ np.linalg.inv(obox) @ nbox
    return ncoord.reshape(old_coord.shape)


def eval_model(model, coord, cell, atype):
    """Evaluate the pt_expt EnergyModel.

    Parameters
    ----------
    model : EnergyModel
        The model to evaluate.
    coord : torch.Tensor
        Coordinates, shape [nf, natoms, 3].
    cell : torch.Tensor
        Cell, shape [nf, 3, 3].
    atype : torch.Tensor
        Atom types, shape [natoms].

    Returns
    -------
    dict
        Model predictions with keys: energy, force, virial.
    """
    nframes = coord.shape[0]
    if len(atype.shape) == 1:
        atype = atype.unsqueeze(0).expand(nframes, -1)
    coord_input = coord.to(dtype=dtype, device=env.DEVICE)
    cell_input = cell.reshape(nframes, 9).to(dtype=dtype, device=env.DEVICE)
    atype_input = atype.to(dtype=torch.long, device=env.DEVICE)
    coord_input.requires_grad_(True)
    result = model(coord_input, atype_input, cell_input)
    return result


class ForceTest:
    def test(self) -> None:
        places = 5
        delta = 1e-5
        natoms = 5
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        coord = coord.numpy()

        def np_infer_coord(coord):
            result = eval_model(
                self.model,
                torch.tensor(coord, device=env.DEVICE).unsqueeze(0),
                cell.unsqueeze(0),
                atype,
            )
            ret = {
                key: result[key].squeeze(0).detach().cpu().numpy()
                for key in ["energy", "force", "virial"]
            }
            return ret

        def ff_coord(_coord):
            return np_infer_coord(_coord)["energy"]

        fdf = -finite_difference(ff_coord, coord, delta=delta).squeeze()
        rff = np_infer_coord(coord)["force"]
        np.testing.assert_almost_equal(fdf, rff, decimal=places)


class VirialTest:
    def test(self) -> None:
        places = 5
        delta = 1e-4
        natoms = 5
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        coord = coord.numpy()
        cell = cell.numpy()

        def np_infer(new_cell):
            result = eval_model(
                self.model,
                torch.tensor(
                    stretch_box(coord, cell, new_cell), device="cpu"
                ).unsqueeze(0),
                torch.tensor(new_cell, device="cpu").unsqueeze(0),
                atype,
            )
            ret = {
                key: result[key].squeeze(0).detach().cpu().numpy()
                for key in ["energy", "force", "virial"]
            }
            return ret

        def ff(bb):
            return np_infer(bb)["energy"]

        fdv = (
            -(finite_difference(ff, cell, delta=delta).transpose(0, 2, 1) @ cell)
            .squeeze()
            .reshape(9)
        )
        rfv = np_infer(cell)["virial"]
        np.testing.assert_almost_equal(fdv, rfv, decimal=places)


class TestEnergyModelSeAForce(unittest.TestCase, ForceTest):
    def setUp(self) -> None:
        ds = DescrptSeA(4.0, 0.5, [8, 6]).to(env.DEVICE)
        ft = InvarFitting(
            "energy",
            2,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        self.model = EnergyModel(ds, ft, type_map=["foo", "bar"]).to(env.DEVICE)
        self.model.eval()


class TestEnergyModelSeAVirial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        ds = DescrptSeA(4.0, 0.5, [8, 6]).to(env.DEVICE)
        ft = InvarFitting(
            "energy",
            2,
            ds.get_dim_out(),
            1,
            mixed_types=ds.mixed_types(),
            seed=GLOBAL_SEED,
        ).to(env.DEVICE)
        self.model = EnergyModel(ds, ft, type_map=["foo", "bar"]).to(env.DEVICE)
        self.model.eval()


if __name__ == "__main__":
    unittest.main()
