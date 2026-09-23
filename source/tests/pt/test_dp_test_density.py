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
from deepmd.pt.model.atomic_model.density_atomic_model import (
    DPDensityAtomicModel,
)
from deepmd.pt.model.descriptor.hybrid import (
    DescrptHybrid,
)
from deepmd.pt.model.descriptor.se_a import (
    DescrptSeA,
)
from deepmd.pt.model.descriptor.se_r import (
    DescrptSeR,
)
from deepmd.pt.model.task.density import (
    DensityFittingNet,
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

    def test_pair_excl_involving_grid_type_applies_in_forward(self) -> None:
        # exclusions involving the grid type must drop the neighbor in the
        # directional grid-to-atom list too, not just in the atom nlist
        model = self.torch_model
        grid_type = len(self.config["model"]["type_map"]) - 1
        input_dict = {
            kk: vv.clone() if isinstance(vv, torch.Tensor) else vv
            for kk, vv in self.input_dict.items()
        }
        model.atomic_model.reinit_pair_exclude([(0, grid_type)])
        try:
            out_excl = model(**input_dict)["density"]
        finally:
            model.atomic_model.reinit_pair_exclude([])
        out_none = model(**input_dict)["density"]
        self.assertFalse(torch.allclose(out_excl, out_none))

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

    def test_property_named_density_dispatches_to_property(self) -> None:
        # the dispatch keys on the grid capability, not the output name, so
        # a property fitting with property_name "density" is not hijacked
        from deepmd.infer.deep_property import (
            DeepProperty,
        )
        from deepmd.pt.model.model.property_model import (
            PropertyModel,
        )
        from deepmd.pt.model.task.property import (
            PropertyFittingNet,
        )

        descriptor = DescrptSeA(rcut=4.0, rcut_smth=0.5, sel=[8, 8])
        fitting = PropertyFittingNet(
            descriptor.get_ntypes(),
            descriptor.get_dim_out(),
            "density",
            neuron=[8, 8],
        )
        model = PropertyModel(descriptor, fitting, type_map=["O", "H"])
        fd, path = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        try:
            with torch.device("cpu"):
                torch.jit.save(torch.jit.script(model), path)
            dp = DeepEval(path)
            self.assertIsInstance(dp, DeepProperty)
            # the property must stay evaluable: the eval guard keys on the
            # grid capability, not the output-variable name
            set_dir = self.system / "set.000"
            coord = np.load(set_dir / "coord.npy")[:2]
            box = np.load(set_dir / "box.npy")[:2]
            atype = np.loadtxt(self.system / "type.raw", dtype=int)
            result = dp.eval(coord, box, atype)
            self.assertEqual(result[0].shape[0], 2)
        finally:
            os.unlink(path)

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

    def test_eval_single_frame_grid_2d(self) -> None:
        # a natural single-frame grid of shape (ngrid, 3) must be carried
        # through with its frame dimension, not sliced to one grid point
        dp = DeepDensity(self.model_path)
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:1]
        box = np.load(set_dir / "box.npy")[:1]
        grid = np.load(set_dir / "grid.npy")[0]  # (ngrid, 3): no frame dim
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        out = dp.eval(coord, box, atype, grid=grid)
        self.assertEqual(out.shape, (1, self.ngrid))

    def test_eval_grid_frame_mismatch(self) -> None:
        dp = DeepDensity(self.model_path)
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        grid = np.load(set_dir / "grid.npy")[:1]  # fewer frames than coord
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        with self.assertRaisesRegex(ValueError, "frames"):
            dp.eval(coord, box, atype, grid=grid)

    def test_fitting_rejects_aparam(self) -> None:
        # grid descriptor rows have no per-atom parameters
        descriptor = DescrptSeA(rcut=4.0, rcut_smth=0.5, sel=[8, 8, 0])
        with self.assertRaisesRegex(ValueError, "aparam"):
            DensityFittingNet(
                descriptor.get_ntypes(),
                descriptor.get_dim_out(),
                numb_aparam=2,
            )

    def test_loss_serialization(self) -> None:
        # GridDensityLoss round-trips through serialize/deserialize
        from deepmd.pt.loss.charge import (
            GridDensityLoss,
        )

        loss_fn = GridDensityLoss(
            starter_learning_rate=0.001, start_pref_d=2.0, limit_pref_d=0.5
        )
        data = loss_fn.serialize()
        self.assertEqual(data["@class"], "GridDensityLoss")
        restored = GridDensityLoss.deserialize(data)
        self.assertEqual(restored.serialize(), data)

    def test_loss_inference_mode(self) -> None:
        # inference=True reports the metrics even with zero prefactors
        from deepmd.pt.loss.charge import (
            GridDensityLoss,
        )
        from deepmd.pt.utils import (
            env,
        )

        loss_fn = GridDensityLoss(inference=True)

        class FakeModel(torch.nn.Module):
            def forward(self, **kwargs):
                return {
                    "density": torch.tensor([[[2.0], [4.0]]], device=env.DEVICE),
                    "mask": torch.tensor(
                        [[1, 1]], dtype=torch.int32, device=env.DEVICE
                    ),
                }

        label = {
            "density": torch.tensor([[[1.0], [2.0]]], device=env.DEVICE),
            "find_density": 1.0,
        }
        _, loss, more_loss = loss_fn({}, FakeModel(), label, 1, 1.0)
        self.assertIn("rmse_d", more_loss)
        self.assertIn("mae_d", more_loss)
        self.assertAlmostEqual(more_loss["rmse_d"].item(), ((1.0 + 4.0) / 2) ** 0.5)

    def test_loss_masks_excluded_grid_points(self) -> None:
        # excluded grid points (mask == 0) must not contribute to the
        # residual, which is normalised by the per-frame mask sum
        from deepmd.pt.loss.charge import (
            GridDensityLoss,
        )
        from deepmd.pt.utils import (
            env,
        )

        loss_fn = GridDensityLoss(start_pref_d=1.0, limit_pref_d=1.0)

        class FakeModel(torch.nn.Module):
            def forward(self, **kwargs):
                return {
                    "density": torch.tensor(
                        [[[1.0], [2.0], [3.0], [4.0]]], device=env.DEVICE
                    ),
                    "mask": torch.tensor(
                        [[1, 1, 0, 0]], dtype=torch.int32, device=env.DEVICE
                    ),
                }

        label = {
            "density": torch.tensor(
                [[[1.0], [1.0], [100.0], [100.0]]], device=env.DEVICE
            ),
            "find_density": 1.0,
        }
        _, loss, more_loss = loss_fn({}, FakeModel(), label, 1, 1.0)
        # only the two unmasked points count: (0^2 + 1^2) / 2
        self.assertAlmostEqual(loss.item(), 0.5)
        self.assertAlmostEqual(more_loss["rmse_d"].item(), 0.5**0.5)

    def test_training_steps(self) -> None:
        # exercises GridDensityLoss.forward on the main path
        workdir = tempfile.mkdtemp(dir=self.tmpdir.name)
        self._run_training(self._training_config(self.system), workdir)

    def test_training_requires_density_label(self) -> None:
        # density is the only supervision signal: a missing density.npy must
        # abort training instead of silently optimising nothing
        system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "system"
        shutil.copytree(self.system, system)
        (system / "set.000" / "density.npy").unlink()
        with self.assertRaisesRegex(RuntimeError, "not found"):
            get_trainer(normalize(self._training_config(system)))

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

    def test_hybrid_descriptor_stat_blocks(self) -> None:
        # hybrid descriptors keep their blocks in descrpt_list: the block
        # discovery must walk the module tree instead of a hard-coded
        # attribute list, and a stat file must not abort the run
        descriptor = DescrptHybrid(
            [
                DescrptSeA(rcut=4.0, rcut_smth=0.5, sel=[8, 8, 0]),
                DescrptSeR(rcut=2.0, rcut_smth=1.0, sel=[4, 4, 0]),
            ]
        )
        fitting = DensityFittingNet(descriptor.get_ntypes(), descriptor.get_dim_out())
        model = DPDensityAtomicModel(descriptor, fitting, type_map=["O", "H", "X"])
        blocks = model._descriptor_stat_blocks()
        self.assertEqual(len(blocks), 2)
        # a stat file must take the soft-failure path, not abort
        stat_dir = DPPath(tempfile.mkdtemp(dir=self.tmpdir.name), "w")

        def sampler() -> list:
            return [dict(self.input_dict)]

        model.compute_or_load_stat(sampler, stat_file_path=stat_dir)

    def test_grid_type_requires_reserved_slot(self) -> None:
        # a real element in the reserved slot must be rejected, not
        # silently used as the grid type
        descriptor = DescrptSeA(rcut=4.0, rcut_smth=0.5, sel=[8, 8])
        fitting = DensityFittingNet(descriptor.get_ntypes(), descriptor.get_dim_out())
        with self.assertRaisesRegex(ValueError, "reserved grid type"):
            DPDensityAtomicModel(descriptor, fitting, type_map=["O", "H"])
        # non-element reserved slot is accepted (all other tests use X)
        DPDensityAtomicModel(descriptor, fitting, type_map=["O", "X"])

    def test_env_protection_enforced_at_model_level(self) -> None:
        # models constructed directly with env_protection == 0.0 get the
        # guard set on the descriptor block, not just a warning
        descriptor = DescrptSeA(
            rcut=4.0, rcut_smth=0.5, sel=[8, 8, 0], env_protection=0.0
        )
        fitting = DensityFittingNet(descriptor.get_ntypes(), descriptor.get_dim_out())
        model = DPDensityAtomicModel(descriptor, fitting, type_map=["O", "H", "X"])
        self.assertEqual(model.descriptor.get_env_protection(), 1e-6)

    def test_env_protection_hybrid(self) -> None:
        # hybrid descriptors have no top-level env_protection: the default
        # must reach every sub-descriptor
        config = deepcopy(self.config)
        descriptor = config["model"].pop("descriptor")
        # the sub-descriptor must not carry an explicit value: the asserted
        # 1e-6 has to be an effect of the default, not of the input config
        descriptor.pop("env_protection", None)
        config["model"]["descriptor"] = {"type": "hybrid", "list": [descriptor]}
        normalized = normalize(config)
        self.assertEqual(
            normalized["model"]["descriptor"]["list"][0]["env_protection"],
            1e-6,
        )

    def test_injected_pass_honors_pair_exclude_types(self) -> None:
        # the injected stat pass must see the model's pair_exclude_types,
        # which are only written by the wrapped sampler
        atomic_model = self.torch_model.atomic_model
        atomic_model.reinit_pair_exclude([(0, 1)])
        seen: list = []
        original = atomic_model.descriptor.compute_input_stats

        def spy(merged, path=None):
            samples = merged() if callable(merged) else merged
            seen.extend(samples)
            original(merged, path)

        atomic_model.descriptor.compute_input_stats = spy  # type: ignore[method-assign]
        try:

            def sampler() -> list:
                return [dict(self.input_dict)]

            atomic_model.compute_or_load_stat(sampler, stat_file_path=None)
        finally:
            atomic_model.descriptor.compute_input_stats = original  # type: ignore[method-assign]
            atomic_model.reinit_pair_exclude([])
        # injected samples carry the grid pseudo-atoms (more rows than atoms)
        natoms = len(np.loadtxt(self.system / "type.raw", dtype=int))
        injected = [s for s in seen if s["atype"].shape[1] > natoms and "grid" in s]
        self.assertTrue(injected, "no injected samples reached the descriptor")
        for sample in injected:
            excluded = [tuple(pair) for pair in sample["pair_exclude_types"]]
            self.assertIn((0, 1), excluded, "model exclusions not honoured")
            self.assertIn((2, 2), excluded, "grid-grid pairs not excluded")

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
