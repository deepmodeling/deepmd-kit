# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
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
from deepmd.utils.path import (
    DPPath,
)

model_density = {
    # the last entry is the reserved grid point type; its sel is 0 because
    # grid points are never neighbors, only centers
    "type_map": ["O", "H", "X"],
    "descriptor": {
        "type": "se_e2_a",
        "sel": [8, 8, 0],
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
        np.savetxt(cls.system / "type_map.raw", ["O", "H", "X"], fmt="%s")

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

    def test_change_out_bias_noop(self) -> None:
        # fine-tuning calls the model-level change_out_bias, whose default
        # change-by-statistic mode would run the grid-less stat wrapper;
        # for density models it must be a no-op instead
        model = self.torch_model
        bias_before = model.atomic_model.out_bias.detach().clone()
        model.change_out_bias(self.input_dict)
        model.change_out_bias(self.input_dict, bias_adjust_mode="set-by-statistic")
        torch.testing.assert_close(model.atomic_model.out_bias, bias_before)

    def test_atom_excl_does_not_mask_grid(self) -> None:
        # grid points carry the reserved type X; excluding a real atom type
        # must not zero the density predictions (it would if grid_type were 0)
        model = self.torch_model
        model.atomic_model.reinit_atom_exclude([0])
        try:
            input_dict = {
                kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
                for kk, vv in self.input_dict.items()
            }
            out = model(**input_dict)
            self.assertFalse(bool((out["density"] == 0).all()))
        finally:
            model.atomic_model.reinit_atom_exclude([])

    def test_model_type_dispatch(self) -> None:
        dp = DeepEval(self.model_path)
        self.assertIsInstance(dp, DeepDensity)

    def _run_training(self, config: dict, workdir: str) -> None:
        """Run a few real training steps (checkpoints land in workdir)."""
        cwd = os.getcwd()
        os.chdir(workdir)
        try:
            trainer = get_trainer(normalize(deepcopy(config)))
            trainer.run()
        finally:
            os.chdir(cwd)
        self.assertTrue(os.path.exists(os.path.join(workdir, "model.ckpt.pt")))

    def _training_config(
        self, system: Path, start: float = 1.0, limit: float = 1.0
    ) -> dict:
        config = deepcopy(self.config)
        config["training"]["training_data"]["systems"] = [str(system)]
        config["training"]["validation_data"]["systems"] = [str(system)]
        config["training"]["training_data"]["batch_size"] = 2
        config["training"]["validation_data"]["batch_size"] = 2
        config["training"]["numb_steps"] = 3
        config["loss"]["start_pref_d"] = start
        config["loss"]["limit_pref_d"] = limit
        return config

    def test_training_steps(self) -> None:
        # exercises GridDensityLoss.forward on the main path
        workdir = tempfile.mkdtemp(dir=self.tmpdir.name)
        self._run_training(self._training_config(self.system), workdir)

    def test_training_without_density_label(self) -> None:
        # density.npy absent with must=False: the find_density == 0 branch
        # must skip the residual but keep the graph connected
        system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "system"
        shutil.copytree(self.system, system)
        (system / "set.000" / "density.npy").unlink()
        workdir = tempfile.mkdtemp(dir=self.tmpdir.name)
        self._run_training(self._training_config(system), workdir)

    def test_training_zero_prefactor(self) -> None:
        # start_pref_d = limit_pref_d = 0 disables the density term;
        # backward() must still work on the graph-connected zero loss
        workdir = tempfile.mkdtemp(dir=self.tmpdir.name)
        self._run_training(
            self._training_config(self.system, start=0.0, limit=0.0), workdir
        )

    def test_env_protection_default(self) -> None:
        # a density model built without an explicit env_protection gets
        # 1e-6 by default (with a warning), and the recorded def script
        # agrees with the built model
        config = deepcopy(self.config)
        config["model"]["descriptor"].pop("env_protection")
        trainer = get_trainer(normalize(config))
        descriptor = trainer.model.atomic_model.descriptor
        self.assertEqual(descriptor.get_env_protection(), 1e-6)
        recorded = json.loads(trainer.model.model_def_script)
        self.assertEqual(recorded["descriptor"]["env_protection"], 1e-6)

    def test_stat_file_grid_row_writeback(self) -> None:
        # the patched grid-type row is written back to the stat cache, and a
        # complete cache then takes the fast path without the injected pass
        stat_dir = DPPath(tempfile.mkdtemp(dir=self.tmpdir.name), "w")
        atomic_model = self.torch_model.atomic_model

        def sampler() -> list:
            return [dict(self.input_dict)]

        atomic_model.compute_or_load_stat(sampler, stat_file_path=stat_dir)
        grid_type = len(self.config["model"]["type_map"]) - 1
        r_x = list(stat_dir.rglob(f"r_{grid_type}"))
        self.assertTrue(r_x, "no grid-type stat item written to the cache")
        for path in r_x:
            self.assertNotEqual(
                float(path.load_numpy()[0]),
                0.0,
                f"{path} still holds zero samples (placeholder row)",
            )
        # second call with the complete cache: the injected pass must not run
        original_inject = atomic_model._inject_grid_samples

        def boom(_sampled: list) -> list:
            raise AssertionError("injected pass should not run on a complete cache")

        atomic_model._inject_grid_samples = boom  # type: ignore[method-assign]
        try:
            atomic_model.compute_or_load_stat(sampler, stat_file_path=stat_dir)
        finally:
            atomic_model._inject_grid_samples = original_inject  # type: ignore[method-assign]

    def test_grid_type_statistics(self) -> None:
        # the reserved grid type X gets real input statistics from the
        # injected grid samples, not the descriptor's placeholder defaults
        descriptor = self.torch_model.atomic_model.descriptor
        dstd = descriptor.sea["dstd"].detach().cpu().numpy()
        self.assertEqual(dstd.shape[0], 3)
        self.assertTrue(np.isfinite(dstd).all())
        # placeholder default is 0.1; real statistics differ from it
        self.assertFalse(
            np.allclose(dstd[-1], 0.1, atol=1e-3),
            f"grid type still has placeholder statistics: {dstd[-1]}",
        )

    def test_grid_type_statistics_dpa2(self) -> None:
        # DPA-2 carries several stat blocks (repinit, repformers,
        # repinit_three_body); the grid-type row must be patched in all of
        # them, not just the first one
        config = deepcopy(self.config)
        config["model"]["descriptor"] = {
            "type": "dpa2",
            "repinit": {
                "tebd_dim": 4,
                "rcut": 4.0,
                "rcut_smth": 0.5,
                "nsel": 16,
                "neuron": [8, 16],
                "axis_neuron": 4,
                "activation_function": "tanh",
                "use_three_body": True,
                "three_body_sel": 8,
                "three_body_rcut": 2.0,
                "three_body_rcut_smth": 1.0,
            },
            "repformer": {
                "rcut": 2.0,
                "rcut_smth": 1.5,
                "nsel": 8,
                "nlayers": 2,
                "g1_dim": 16,
                "g2_dim": 8,
                "attn2_hidden": 8,
                "attn2_nhead": 2,
                "attn1_hidden": 16,
                "attn1_nhead": 2,
                "axis_neuron": 4,
            },
            "env_protection": 1e-6,
            "seed": 1,
        }
        trainer = get_trainer(normalize(config))
        descriptor = trainer.model.atomic_model.descriptor
        for name in ("repinit", "repformers", "repinit_three_body"):
            block = getattr(descriptor, name)
            dstd = block["dstd"].detach().cpu().numpy()
            self.assertTrue(np.isfinite(dstd).all(), name)
            self.assertFalse(
                np.allclose(dstd[-1], 0.1, atol=1e-3),
                f"{name} still has placeholder statistics for the grid type",
            )
        # the reserved grid type X gets real input statistics from the
        # injected grid samples, not the descriptor's placeholder defaults
        descriptor = self.torch_model.atomic_model.descriptor
        dstd = descriptor.sea["dstd"].detach().cpu().numpy()
        self.assertEqual(dstd.shape[0], 3)
        self.assertTrue(np.isfinite(dstd).all())
        # placeholder default is 0.1; real statistics differ from it
        self.assertFalse(
            np.allclose(dstd[-1], 0.1, atol=1e-3),
            f"grid type still has placeholder statistics: {dstd[-1]}",
        )

    def test_eval_shape(self) -> None:
        dp = DeepDensity(self.model_path)
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        grid = np.load(set_dir / "grid.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        out = dp.eval(coord, box, atype, grid=grid)
        self.assertEqual(out.shape, (2, self.ngrid))

    def test_eval_requires_grid(self) -> None:
        # evaluating a density model without grid must fail with a clear
        # error instead of a bare KeyError on the output name table
        dp = DeepDensity(self.model_path)
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        with self.assertRaisesRegex(ValueError, "grid is required"):
            dp.deep_eval.eval(coord, box, atype)

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
