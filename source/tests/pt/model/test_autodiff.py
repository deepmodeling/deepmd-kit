# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import unittest

import numpy as np
import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.utils import (
    to_numpy_array,
)

from ...seed import (
    GLOBAL_SEED,
)

dtype = torch.float64

from ..common import (
    eval_model,
)
from .test_permutation import (
    model_dpa1,
    model_dpa2,
    model_hybrid,
    model_se_e2_a,
    model_spin,
    model_zbl,
)


# from deepmd-kit repo
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


def finite_difference_cell_energy(
    energy_func,
    cell: np.ndarray,
    delta: float = 1e-4,
) -> np.ndarray:
    """
    Compute cell energy derivatives via central finite differences.

    Parameters
    ----------
    energy_func : callable
        Function that returns a scalar energy for a given cell.
    cell : np.ndarray
        Cell matrix with shape (3, 3).
    delta : float
        Perturbation size in unitless.

    Returns
    -------
    np.ndarray
        Energy derivatives dE/dh with shape (3, 3).
    """
    deriv = np.zeros_like(cell)
    for ii in range(3):
        for jj in range(3):
            perturb = np.zeros_like(cell)
            perturb[ii, jj] = delta
            ep = energy_func(cell + perturb)
            em = energy_func(cell - perturb)
            deriv[ii, jj] = (ep - em) / (2.0 * delta)
    return deriv


class ForceTest:
    def test(
        self,
    ) -> None:
        places = 5
        delta = 1e-5
        natoms = 5
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        spin = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        # assumes input to be numpy tensor
        coord = coord.numpy()
        spin = spin.numpy()
        test_spin = getattr(self, "test_spin", False)
        if not test_spin:
            test_keys = ["energy", "force", "virial"]
        else:
            test_keys = ["energy", "force", "force_mag", "virial"]

        def np_infer_coord(
            coord,
        ):
            result = eval_model(
                self.model,
                torch.tensor(coord, device=env.DEVICE).unsqueeze(0),
                cell.unsqueeze(0),
                atype,
                spins=torch.tensor(spin, device=env.DEVICE).unsqueeze(0),
            )
            # detach
            ret = {key: to_numpy_array(result[key].squeeze(0)) for key in test_keys}
            return ret

        def np_infer_spin(
            spin,
        ):
            result = eval_model(
                self.model,
                torch.tensor(coord, device=env.DEVICE).unsqueeze(0),
                cell.unsqueeze(0),
                atype,
                spins=torch.tensor(spin, device=env.DEVICE).unsqueeze(0),
            )
            # detach
            ret = {key: to_numpy_array(result[key].squeeze(0)) for key in test_keys}
            return ret

        def ff_coord(_coord):
            return np_infer_coord(_coord)["energy"]

        def ff_spin(_spin):
            return np_infer_spin(_spin)["energy"]

        if not test_spin:
            fdf = -finite_difference(ff_coord, coord, delta=delta).squeeze()
            rff = np_infer_coord(coord)["force"]
            np.testing.assert_almost_equal(fdf, rff, decimal=places)
        else:
            # real force
            fdf = -finite_difference(ff_coord, coord, delta=delta).squeeze()
            rff = np_infer_coord(coord)["force"]
            np.testing.assert_almost_equal(fdf, rff, decimal=places)
            # magnetic force
            fdf = -finite_difference(ff_spin, spin, delta=delta).squeeze()
            rff = np_infer_spin(spin)["force_mag"]
            np.testing.assert_almost_equal(fdf, rff, decimal=places)


class VirialTest:
    def test(
        self,
    ) -> None:
        places = 5
        delta = 1e-4
        natoms = 5
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        spin = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        # assumes input to be numpy tensor
        coord = coord.numpy()
        spin = spin.numpy()
        cell = cell.numpy()
        test_spin = getattr(self, "test_spin", False)
        if not test_spin:
            test_keys = ["energy", "force", "virial"]
        else:
            test_keys = ["energy", "force", "force_mag", "virial"]

        def np_infer(
            new_cell,
        ):
            result = eval_model(
                self.model,
                torch.tensor(
                    stretch_box(coord, cell, new_cell), device="cpu"
                ).unsqueeze(0),
                torch.tensor(new_cell, device="cpu").unsqueeze(0),
                atype,
                spins=torch.tensor(spin, device=env.DEVICE).unsqueeze(0),
            )
            # detach
            ret = {key: to_numpy_array(result[key].squeeze(0)) for key in test_keys}
            # detach
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
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSeAVirial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_se_e2_a)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1Force(unittest.TestCase, ForceTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA1Virial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa1)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPA2Force(unittest.TestCase, ForceTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelDPAUniVirial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_dpa2)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelHybridForce(unittest.TestCase, ForceTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelHybridVirial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_hybrid)
        self.type_split = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelZBLForce(unittest.TestCase, ForceTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelZBLVirial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_zbl)
        self.type_split = False
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinSeAForce(unittest.TestCase, ForceTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_spin)
        self.type_split = False
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinSeAVirial(unittest.TestCase, VirialTest):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_spin)
        self.type_split = False
        self.test_spin = True
        self.model = get_model(model_params).to(env.DEVICE)


class TestEnergyModelSpinSeAVirialShear(unittest.TestCase):
    def setUp(self) -> None:
        model_params = copy.deepcopy(model_spin)
        self.model = get_model(model_params).to(env.DEVICE)

    def test(self) -> None:
        places = 5
        delta = 1e-4
        natoms = 5
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        atype = np.array([0, 0, 0, 1, 1], dtype=np.int32)

        # === Step 1. Prepare inputs ===
        cell = torch.rand([3, 3], dtype=dtype, device="cpu", generator=generator)
        cell = (cell + cell.T) + 5.0 * torch.eye(3, device="cpu")
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        spin = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = coord.numpy()
        spin = spin.numpy()
        cell = cell.numpy()

        # === Step 2. Define energy and virial evaluators ===
        def np_infer(new_cell):
            coord_new = stretch_box(coord, cell, new_cell)
            result = eval_model(
                self.model,
                coord_new.reshape(1, natoms, 3),
                new_cell.reshape(1, 3, 3),
                atype,
                spins=spin.reshape(1, natoms, 3),
            )
            return result

        def energy_func(new_cell):
            return np_infer(new_cell)["energy"].reshape(-1)[0]

        # === Step 3. Finite-difference virial by shear perturbations ===
        dE_dh = finite_difference_cell_energy(energy_func, cell, delta=delta)
        virial_fd = -(dE_dh.T @ cell).reshape(9)

        # === Step 4. Compare with analytic virial ===
        virial_ref = np_infer(cell)["virial"].reshape(9)
        np.testing.assert_almost_equal(virial_fd, virial_ref, decimal=places)
