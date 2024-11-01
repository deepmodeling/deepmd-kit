# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import paddle

from deepmd.pd.entrypoints.main import (
    get_trainer,
)
from deepmd.pd.utils.ase_calc import (
    DPCalculator,
)

from ..seed import (
    GLOBAL_SEED,
)

dtype = paddle.float64

paddle.framework.core.set_prim_eager_enabled(True)
paddle.framework.core._set_prim_all_enabled(True)


class TestCalculator(unittest.TestCase):
    def setUp(self):
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

        device = paddle.get_device()
        paddle.set_device("cpu")
        input_dict, label_dict, _ = trainer.get_data(is_train=False)
        paddle.set_device(device)
        _, _, more_loss = trainer.wrapper(**input_dict, label=label_dict, cur_lr=1.0)

        self.calculator = DPCalculator("model.pd")

    def test_calculator(self):
        from ase import (
            Atoms,
        )

        natoms = 5
        cell = paddle.eye(3, dtype=dtype).to(device="cpu") * 10
        paddle.seed(GLOBAL_SEED)
        coord = paddle.rand([natoms, 3], dtype=dtype).to(device="cpu")
        coord = paddle.matmul(coord, cell)
        atype = paddle.to_tensor([0, 0, 0, 1, 1])
        atomic_numbers = [1, 1, 1, 8, 8]
        idx_perm = [1, 0, 4, 3, 2]

        prec = 1e-10
        low_prec = 1e-4

        ase_atoms0 = Atoms(
            numbers=atomic_numbers,
            positions=coord,
            # positions=[tuple(item) for item in coordinate],
            cell=cell,
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
            positions=coord[idx_perm, :],
            # positions=[tuple(item) for item in coordinate],
            cell=cell,
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
