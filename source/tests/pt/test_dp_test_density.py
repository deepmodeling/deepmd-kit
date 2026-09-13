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
        # grid points may coincide with atoms; without protection the
        # env matrix would divide by zero and produce NaN densities
        "env_protection": 1e-6,
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
        # keep the live torch model and a data sample for the gradient check
        cls.torch_model = trainer.model
        cls.input_dict = input_dict
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

    def test_eval_no_auto_batch(self) -> None:
        # auto_batch_size=False bypasses execute_all's single-tuple unwrapping;
        # the density result must still be normalized to a bare ndarray
        dp = DeepDensity(self.model_path, auto_batch_size=False)
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        grid = np.load(set_dir / "grid.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        out = dp.eval(coord, box, atype, grid=grid)
        self.assertEqual(out.shape, (2, self.ngrid))
        # and match the default-batching result numerically
        out_default = DeepDensity(self.model_path).eval(coord, box, atype, grid=grid)
        np.testing.assert_allclose(out, out_default)

    def test_grid_at_atoms_finite(self) -> None:
        # grid points coincident with atoms are legitimate inputs; with
        # env_protection > 0 (set in the test config) the predictions and
        # gradients must stay finite instead of turning NaN via 1/r terms
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        grid_at_atoms = coord.reshape(coord.shape[0], -1, 3)
        out = DeepDensity(self.model_path).eval(coord, box, atype, grid=grid_at_atoms)
        self.assertTrue(np.isfinite(out).all())

        # gradients wrt the grid input and all parameters stay finite
        input_dict = {
            kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
            for kk, vv in self.input_dict.items()
        }
        coord_t = input_dict["coord"]
        grid_t = (
            coord_t.reshape(coord_t.shape[0], -1, 3).detach().clone().requires_grad_()
        )
        input_dict["grid"] = grid_t
        model_out = self.torch_model(**input_dict)
        self.assertTrue(torch.isfinite(model_out["density"]).all())
        model_out["density"].sum().backward()
        self.assertIsNotNone(grid_t.grad)
        self.assertTrue(torch.isfinite(grid_t.grad).all())
        for param in self.torch_model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.isfinite(param.grad).all())

    def test_grid_periodic_translation(self) -> None:
        # grids shifted by integer cell vectors are periodically equivalent
        # and must produce the same density as the wrapped originals
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        grid = np.load(set_dir / "grid.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        dp = DeepDensity(self.model_path)
        out = dp.eval(coord, box, atype, grid=grid)
        cell_vec = box.reshape(box.shape[0], 3, 3)[:, 0]
        out_shift = dp.eval(coord, box, atype, grid=grid + 2.0 * cell_vec[:, None, :])
        np.testing.assert_allclose(out_shift, out, rtol=1e-5, atol=1e-5)

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
        # grid/density must be shuffled together with the frames: every
        # (label, pred) row written by the shuffled run must match a row
        # from the unshuffled reference run
        detail_ref = os.path.join(self.tmpdir.name, "detail_ref")
        dp_test(
            model=self.model_path,
            system=str(self.system),
            datafile=None,
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_ref,
            atomic=False,
        )
        ref_rows = set()
        for frame in range(self.nframes):
            detail = np.loadtxt(f"{detail_ref}.density.out.{frame}", skiprows=1)
            for row in detail:
                ref_rows.add(tuple(np.round(row, decimals=5)))

        detail_shuf = os.path.join(self.tmpdir.name, "detail_shuf")
        dp_test(
            model=self.model_path,
            system=str(self.system),
            datafile=None,
            numb_test=2,
            rand_seed=42,
            shuffle_test=True,
            detail_file=detail_shuf,
            atomic=False,
        )
        shuf_rows = []
        for frame in range(2):
            detail = np.loadtxt(f"{detail_shuf}.density.out.{frame}", skiprows=1)
            shuf_rows.extend(tuple(np.round(row, decimals=5)) for row in detail)
        self.assertEqual(len(shuf_rows), 2 * self.ngrid)
        for row in shuf_rows:
            self.assertIn(row, ref_rows)


if __name__ == "__main__":
    unittest.main()
