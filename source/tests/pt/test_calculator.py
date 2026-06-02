# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest
from copy import (
    deepcopy,
)
from importlib.util import (
    find_spec,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.utils.ase_calc import (
    DPCalculator,
)

from ..seed import (
    GLOBAL_SEED,
)

dtype = torch.float64


class TestCalculator(unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()

        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        _, _, more_loss = trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)

        self.calculator = DPCalculator("model.pt")

    def test_calculator(self) -> None:
        from ase import (
            Atoms,
        )

        natoms = 5
        cell = torch.eye(3, dtype=dtype, device="cpu") * 10
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        atype = torch.IntTensor([0, 0, 0, 1, 1])
        atomic_numbers = [1, 1, 1, 8, 8]
        idx_perm = [1, 0, 4, 3, 2]

        # Convert tensors to numpy for ASE compatibility
        cell_np = cell.numpy()
        coord_np = coord.numpy()

        prec = 1e-10
        low_prec = 1e-4

        ase_atoms0 = Atoms(
            numbers=atomic_numbers,
            positions=coord_np,
            # positions=[tuple(item) for item in coordinate],
            cell=cell_np,
            calculator=self.calculator,
            pbc=True,
        )
        e0, f0 = ase_atoms0.get_potential_energy(), ase_atoms0.get_forces()
        s0, v0 = (
            ase_atoms0.get_stress(voigt=True),
            -ase_atoms0.get_stress(voigt=False) * ase_atoms0.get_volume(),
        )

        ase_atoms1 = Atoms(
            numbers=[atomic_numbers[i] for i in idx_perm],
            positions=coord_np[idx_perm, :],
            # positions=[tuple(item) for item in coordinate],
            cell=cell_np,
            calculator=self.calculator,
            pbc=True,
        )
        e1, f1 = ase_atoms1.get_potential_energy(), ase_atoms1.get_forces()
        s1, v1 = (
            ase_atoms1.get_stress(voigt=True),
            -ase_atoms1.get_stress(voigt=False) * ase_atoms1.get_volume(),
        )

        assert isinstance(e0, float)
        assert f0.shape == (natoms, 3)
        assert v0.shape == (3, 3)
        np.testing.assert_allclose(e0, e1, rtol=low_prec, atol=prec)
        np.testing.assert_allclose(f0[idx_perm, :], f1, rtol=low_prec, atol=prec)
        np.testing.assert_allclose(s0, s1, rtol=low_prec, atol=prec)
        np.testing.assert_allclose(v0, v1, rtol=low_prec, atol=prec)


class TestCalculatorWithFparamAparam(unittest.TestCase):
    def setUp(self) -> None:
        input_json = str(Path(__file__).parent / "water/se_atten.json")
        with open(input_json) as f:
            self.config = json.load(f)
        self.config["training"]["numb_steps"] = 1
        self.config["model"]["fitting_net"]["numb_fparam"] = 2
        self.config["model"]["fitting_net"]["numb_aparam"] = 1
        self.config["training"]["save_freq"] = 1
        data_file = [str(Path(__file__).parent / "water/data/data_0")]
        self.config["training"]["training_data"]["systems"] = data_file
        self.config["training"]["validation_data"]["systems"] = [
            str(Path(__file__).parent / "water/data/single")
        ]
        self.input_json = "test_dp_test.json"
        with open(self.input_json, "w") as fp:
            json.dump(self.config, fp, indent=4)

        trainer = get_trainer(deepcopy(self.config))
        trainer.run()

        with torch.device("cpu"):
            input_dict, label_dict, _ = trainer.get_data(is_train=False)
        _, _, more_loss = trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)

        self.calculator = DPCalculator("model.pt")

    def test_calculator(self) -> None:
        from ase import (
            Atoms,
        )

        natoms = 5
        cell = torch.eye(3, dtype=dtype, device="cpu") * 10
        generator = torch.Generator(device="cpu").manual_seed(GLOBAL_SEED)
        coord = torch.rand([natoms, 3], dtype=dtype, device="cpu", generator=generator)
        coord = torch.matmul(coord, cell)
        fparam = torch.IntTensor([1, 2]).numpy()
        aparam = torch.IntTensor([[1], [0], [2], [1], [0]]).numpy()
        atomic_numbers = [1, 1, 1, 8, 8]
        idx_perm = [1, 0, 4, 3, 2]

        # Convert tensors to numpy for ASE compatibility
        cell_np = cell.numpy()
        coord_np = coord.numpy()

        prec = 1e-10
        low_prec = 1e-4

        ase_atoms0 = Atoms(
            numbers=atomic_numbers,
            positions=coord_np,
            # positions=[tuple(item) for item in coordinate],
            cell=cell_np,
            calculator=self.calculator,
            pbc=True,
        )
        ase_atoms0.info.update({"fparam": fparam, "aparam": aparam})
        e0, f0 = ase_atoms0.get_potential_energy(), ase_atoms0.get_forces()
        s0, v0 = (
            ase_atoms0.get_stress(voigt=True),
            -ase_atoms0.get_stress(voigt=False) * ase_atoms0.get_volume(),
        )

        ase_atoms1 = Atoms(
            numbers=[atomic_numbers[i] for i in idx_perm],
            positions=coord_np[idx_perm, :],
            # positions=[tuple(item) for item in coordinate],
            cell=cell_np,
            calculator=self.calculator,
            pbc=True,
        )
        ase_atoms1.info.update({"fparam": fparam, "aparam": aparam[idx_perm, :]})
        e1, f1 = ase_atoms1.get_potential_energy(), ase_atoms1.get_forces()
        s1, v1 = (
            ase_atoms1.get_stress(voigt=True),
            -ase_atoms1.get_stress(voigt=False) * ase_atoms1.get_volume(),
        )

        assert isinstance(e0, float)
        assert f0.shape == (natoms, 3)
        assert v0.shape == (3, 3)
        np.testing.assert_allclose(e0, e1, rtol=low_prec, atol=prec)
        np.testing.assert_allclose(f0[idx_perm, :], f1, rtol=low_prec, atol=prec)
        np.testing.assert_allclose(s0, s1, rtol=low_prec, atol=prec)
        np.testing.assert_allclose(v0, v1, rtol=low_prec, atol=prec)


_CALC_CONFIG = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [20, 20],
        "rcut_smth": 0.5,
        "rcut": 4.0,
        "neuron": [8],
        "resnet_dt": False,
        "axis_neuron": 4,
        "seed": 1,
    },
    "fitting_net": {"neuron": [8], "resnet_dt": True, "seed": 1},
}


@unittest.skipUnless(find_spec("vesin") is not None, "vesin not installed")
class TestCalculatorNlistBackend(unittest.TestCase):
    """The ASE DP calculator must thread nlist_backend to DeepPot and give
    backend-independent results.
    """

    @classmethod
    def setUpClass(cls) -> None:
        from deepmd.pt.model.model import (
            get_model,
        )

        torch.manual_seed(1)
        cls.model_file = "calc_model_nlist_backend.pth"
        torch.jit.script(get_model(deepcopy(_CALC_CONFIG))).save(cls.model_file)

    @classmethod
    def tearDownClass(cls) -> None:
        import os

        if os.path.isfile(cls.model_file):
            os.remove(cls.model_file)

    def test_calculator_nlist_backend(self) -> None:
        from ase import (
            Atoms,
        )

        from deepmd.calculator import (
            DP,
        )

        rng = np.random.default_rng(7)
        atoms = Atoms(
            numbers=[8, 1, 1, 8, 1, 1],
            positions=rng.random((6, 3)) * 8.0,
            cell=np.eye(3) * 8.0,
            pbc=True,
        )
        calc_native = DP(self.model_file, nlist_backend="native")
        calc_vesin = DP(self.model_file, nlist_backend="vesin")
        # the kwarg must reach the underlying DeepPot backend
        self.assertEqual(calc_native.dp.deep_eval.nlist_backend, "native")
        self.assertTrue(calc_vesin.dp.deep_eval._use_vesin)

        a_native = atoms.copy()
        a_native.calc = calc_native
        a_vesin = atoms.copy()
        a_vesin.calc = calc_vesin
        np.testing.assert_allclose(
            a_native.get_potential_energy(),
            a_vesin.get_potential_energy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            a_native.get_forces(), a_vesin.get_forces(), rtol=1e-10, atol=1e-10
        )
