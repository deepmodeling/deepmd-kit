# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)

import numpy as np
import torch

from deepmd.entrypoints.test import test as dp_test
from deepmd.infer.deep_density import (
    DeepDensity,
)
from deepmd.infer.deep_eval import (
    DeepEval,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.utils.argcheck import (
    normalize,
)

model_density = {
    "type_map": ["O", "H"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [8, 8],
        "rcut_smth": 0.50,
        "rcut": 4.00,
        "neuron": [8, 16],
        "resnet_dt": False,
        "axis_neuron": 4,
        "seed": 1,
    },
    "fitting_net": {
        "type": "density",
        "neuron": [8, 8],
        "seed": 1,
    },
}


class TestDPTestDensity(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        root = Path(cls.tmpdir.name)

        # write a synthetic density system
        rng = np.random.default_rng(42)
        cls.nframes, natoms, cls.ngrid = 4, 5, 8
        cls.system = root / "system"
        set_dir = cls.system / "set.000"
        set_dir.mkdir(parents=True)
        np.save(set_dir / "coord.npy", rng.random((cls.nframes, natoms * 3)) * 8.0)
        np.save(
            set_dir / "box.npy",
            np.tile(np.eye(3).reshape(-1) * 10.0, (cls.nframes, 1)),
        )
        np.save(set_dir / "grid.npy", rng.random((cls.nframes, cls.ngrid, 3)) * 8.0)
        np.save(set_dir / "density.npy", rng.random((cls.nframes, cls.ngrid, 1)))
        np.savetxt(cls.system / "type.raw", [0, 0, 1, 1, 1], fmt="%d")
        np.savetxt(cls.system / "type_map.raw", ["O", "H"], fmt="%s")

        cls.config = {
            "model": deepcopy(model_density),
            "learning_rate": {
                "type": "exp",
                "start_lr": 0.001,
                "stop_lr": 1e-8,
                "decay_steps": 10,
            },
            "optimizer": {"type": "Adam"},
            "loss": {
                "type": "grid_density",
                "start_pref_d": 1.0,
                "limit_pref_d": 1.0,
            },
            "training": {
                "training_data": {
                    "systems": [str(cls.system)],
                    "batch_size": 1,
                },
                "validation_data": {
                    "systems": [str(cls.system)],
                    "batch_size": 1,
                },
                "numb_steps": 1,
                "seed": 1,
                "disp_file": os.devnull,
                "save_freq": 100,
            },
        }

        # build and freeze a tiny density model through the trainer path
        trainer = get_trainer(normalize(deepcopy(cls.config)))
        with torch.device("cpu"):
            input_dict, _, _ = trainer.get_data(is_train=False)
        # the density model takes grid instead of spin as the extra input
        input_dict.pop("spin", None)
        trainer.model(**input_dict)
        model = torch.jit.script(trainer.model)
        tmp_fd, cls.model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(model, cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        os.unlink(cls.model_path)
        cls.tmpdir.cleanup()

    def test_model_type_dispatch(self) -> None:
        dp = DeepEval(self.model_path)
        self.assertIsInstance(dp, DeepDensity)

    def test_eval_shape(self) -> None:
        dp = DeepDensity(self.model_path)
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        grid = np.load(set_dir / "grid.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        out = dp.eval(coord, box, atype, grid=grid)
        self.assertEqual(out.shape, (2, self.ngrid))

    def test_dp_test(self) -> None:
        detail_file = os.path.join(self.tmpdir.name, "detail")
        dp_test(
            model=self.model_path,
            system=str(self.system),
            datafile=None,
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        # one detail file per frame, each holding ngrid [label, pred] pairs
        for frame in range(self.nframes):
            detail = np.loadtxt(f"{detail_file}.density.out.{frame}", skiprows=1)
            self.assertEqual(detail.shape, (self.ngrid, 2))

    def test_dp_test_shuffle(self) -> None:
        # grid/density must be shuffled together with the frames
        dp_test(
            model=self.model_path,
            system=str(self.system),
            datafile=None,
            numb_test=2,
            rand_seed=42,
            shuffle_test=True,
            detail_file=None,
            atomic=False,
        )


if __name__ == "__main__":
    unittest.main()
