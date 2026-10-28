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
from unittest.mock import (
    patch,
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
from deepmd.infer.model_test import (
    build_tester,
)
from deepmd.pt.entrypoints.main import (
    get_trainer,
)
from deepmd.pt.model.atomic_model.density_atomic_model import (
    DPDensityAtomicModel,
)
from deepmd.pt.model.atomic_model.dp_atomic_model import (
    DPAtomicModel,
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
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.model.model.density_model import (
    GridDensityModel,
)
from deepmd.pt.model.task.density import (
    DensityFittingNet,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.data import (
    DeepmdData,
)
from deepmd.utils.path import (
    DPPath,
)
from deepmd.utils.stat_file import (
    StatFileSpec,
    open_stat_file,
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
        # Keep both no-op overrides compatible with their base APIs.
        model.change_out_bias(self.input_dict, "set-by-statistic", {})
        model.atomic_model.change_out_bias(
            self.input_dict, None, "set-by-statistic", None, None
        )
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

    def test_eval_rejects_invalid_grid_inputs(self) -> None:
        dp = DeepDensity(self.model_path)
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:2]
        box = np.load(set_dir / "box.npy")[:2]
        grid = np.load(set_dir / "grid.npy")[:2]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        invalid = (
            ("frame mismatch", grid[:1], {}, "frames"),
            ("ambiguous rank two", grid[0], {}, "ambiguous"),
            ("wrong coordinate width", np.zeros((2, self.ngrid, 2)), {}, "shape"),
            ("rank four", np.zeros((2, self.ngrid, 3, 1)), {}, "shape"),
            ("empty", np.zeros((2, 0, 3)), {}, "at least one point"),
            ("invalid batch size", grid, {"grid_batch_size": 0}, "grid_batch_size"),
        )
        for case, invalid_grid, kwargs, message in invalid:
            with self.subTest(case=case):
                with self.assertRaisesRegex(ValueError, message):
                    dp.eval(coord, box, atype, grid=invalid_grid, **kwargs)

    def test_fitting_rejects_aparam(self) -> None:
        # grid descriptor rows have no per-atom parameters
        descriptor = DescrptSeA(rcut=4.0, rcut_smth=0.5, sel=[8, 8, 0])
        with self.assertRaisesRegex(ValueError, "aparam"):
            DensityFittingNet(
                descriptor.get_ntypes(),
                descriptor.get_dim_out(),
                numb_aparam=2,
            )

    def test_fitting_model_serialization_roundtrip(self) -> None:
        # GeneralFitting.serialize() emits @version 5; the density fitting
        # and the whole model must round-trip through serialize/deserialize,
        # which is what model freezing relies on. The model must go through
        # BaseModel.deserialize, which dispatches on the serialized fitting
        # type ("density") and is what dp convert-backend uses.
        from deepmd.pt.loss.charge import (
            GridDensityLoss,
        )
        from deepmd.pt.model.model.model import (
            BaseModel,
        )

        fitting = self.torch_model.atomic_model.fitting_net
        restored_fit = DensityFittingNet.deserialize(fitting.serialize())
        self._assert_state_dict_equal(fitting, restored_fit)
        model = self.torch_model
        restored_model = BaseModel.deserialize(model.serialize())
        self.assertIsInstance(restored_model, GridDensityModel)
        self._assert_state_dict_equal(model, restored_model)

        loss_fn = GridDensityLoss(
            starter_learning_rate=0.001, start_pref_d=2.0, limit_pref_d=0.5
        )
        data = loss_fn.serialize()
        self.assertEqual(data["@class"], "GridDensityLoss")
        restored_loss = GridDensityLoss.deserialize(data)
        self.assertEqual(restored_loss.serialize(), data)

    def _assert_state_dict_equal(
        self, module0: torch.nn.Module, module1: torch.nn.Module
    ) -> None:
        sd0 = module0.state_dict()
        sd1 = module1.state_dict()
        self.assertEqual(set(sd0), set(sd1))
        for key in sd0:
            torch.testing.assert_close(sd0[key], sd1[key])

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

    def test_training_with_disp_avg(self) -> None:
        # TaskLoss.training_metric_names is consulted before training starts
        # whenever averaged metrics are requested.
        config = self._training_config(self.system)
        config["training"]["disp_avg"] = True
        self._run_training(config, tempfile.mkdtemp(dir=self.tmpdir.name))

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

    def test_env_protection_hybrid_defaults(self) -> None:
        cases = (
            ("all omitted", [None], [1e-6]),
            ("shared explicit value", [0.1, None], [0.1, 0.1]),
        )
        for case, values, expected in cases:
            config = deepcopy(self.config)
            base = config["model"].pop("descriptor")
            descriptors = []
            for value in values:
                descriptor = deepcopy(base)
                descriptor.pop("env_protection", None)
                if value is not None:
                    descriptor["env_protection"] = value
                descriptors.append(descriptor)
            config["model"]["descriptor"] = {
                "type": "hybrid",
                "list": descriptors,
            }
            with self.subTest(case=case):
                normalized = normalize(config)
                actual = [
                    sub["env_protection"]
                    for sub in normalized["model"]["descriptor"]["list"]
                ]
                self.assertEqual(actual, expected)

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

    def test_stat_cache_rejects_zero_sample_angular_grid_row(self) -> None:
        stat_dir = DPPath(tempfile.mkdtemp(dir=self.tmpdir.name), "w")
        atomic_model = self.torch_model.atomic_model

        def sampler() -> list:
            return [dict(self.input_dict)]

        atomic_model.compute_or_load_stat(sampler, stat_file_path=stat_dir)
        grid_type = len(self.config["model"]["type_map"]) - 1
        angular_rows = list(stat_dir.rglob(f"a_{grid_type}"))
        self.assertTrue(angular_rows, "no angular grid-type statistic in cache")
        for path in angular_rows:
            value = path.load_numpy()
            value[0] = 0.0
            path.save_numpy(value)

        self.assertTrue(atomic_model._grid_stat_missing(stat_dir))

    def test_stat_without_grid_warns_and_uses_defaults(self) -> None:
        # Statistics can be requested before a sampler is augmented with the
        # density input; this must take the documented fallback rather than
        # assuming every sample contains grid coordinates.
        atomic_model = get_model(deepcopy(model_density)).atomic_model
        sample = dict(self.input_dict)
        sample.pop("grid")
        with self.assertLogs(
            "deepmd.pt.model.atomic_model.density_atomic_model", level="WARNING"
        ) as logs:
            atomic_model.compute_or_load_stat(lambda: [sample])
        self.assertIn("No grid data", "\n".join(logs.output))

    def test_stat_writeback_skips_missing_cache_directory(self) -> None:
        atomic_model = get_model(deepcopy(model_density)).atomic_model
        grid_stats = [
            (block["davg"][-1].detach().clone(), block["dstd"][-1].detach().clone())
            for block in atomic_model._descriptor_stat_blocks()
        ]
        stat_dir = DPPath(tempfile.mkdtemp(dir=self.tmpdir.name), "w")
        # A cache can lack one descriptor block (for example, after an older
        # partial run); that block is deliberately skipped during write-back.
        atomic_model._write_back_grid_stat(stat_dir, grid_stats)

    def test_read_only_stat_cache_patches_grid_row_in_memory(self) -> None:
        # Older complete caches can contain a zero-sample placeholder for the
        # reserved grid type. Read mode may patch that row in memory, but must
        # never attempt the write-back that update mode performs.
        stat_file = Path(self.tmpdir.name) / "density-stat.hdf5"
        model = get_model(deepcopy(model_density))
        atomic_model = model.atomic_model

        def sampler() -> list:
            return [dict(self.input_dict)]

        with open_stat_file(StatFileSpec(str(stat_file))) as update_path:
            self.assertIsNotNone(update_path)
            # The parent pass writes a complete cache with the ordinary
            # zero-sample placeholder for the reserved grid type.
            DPAtomicModel.compute_or_load_stat(atomic_model, sampler, update_path)
        original = stat_file.read_bytes()

        with open_stat_file(StatFileSpec(str(stat_file), "read")) as read_path:
            self.assertIsNotNone(read_path)
            with self.assertLogs(
                "deepmd.pt.model.atomic_model.density_atomic_model", level="WARNING"
            ) as logs:
                atomic_model.compute_or_load_stat(sampler, read_path)
        self.assertIn("read-only", "\n".join(logs.output))
        self.assertEqual(stat_file.read_bytes(), original)

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

    def test_eval_without_periodic_box(self) -> None:
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")[:1]
        grid = np.load(set_dir / "grid.npy")[:1]
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        out = DeepDensity(self.model_path).eval(coord, None, atype, grid=grid)
        self.assertEqual(out.shape, (1, self.ngrid))
        self.assertTrue(np.isfinite(out).all())

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

    def test_dp_test_without_detail_file(self) -> None:
        dp = DeepDensity(self.model_path)
        data = DeepmdData(
            str(self.system),
            set_prefix="set",
            shuffle_test=False,
            type_map=dp.get_type_map(),
            sort_atoms=False,
        )
        errors = build_tester(dp, atomic=False).run(
            data,
            str(self.system),
            numb_test=self.nframes,
            detail_file=None,
        )
        self.assertEqual(set(errors), {"mae_density", "rmse_density"})

    def test_dp_test_chunking_matches_whole_run(self) -> None:
        def run(chunk_size: int, detail_file: str) -> dict:
            dp = DeepDensity(self.model_path)
            data = DeepmdData(
                str(self.system),
                set_prefix="set",
                shuffle_test=False,
                type_map=dp.get_type_map(),
                sort_atoms=False,
            )
            with patch.dict(
                os.environ, {"DP_TEST_CHUNK_ATOMS": str(chunk_size)}, clear=False
            ):
                return build_tester(dp, atomic=False).run(
                    data,
                    str(self.system),
                    numb_test=self.nframes,
                    detail_file=detail_file,
                )

        whole_detail = os.path.join(self.tmpdir.name, "detail_whole")
        chunk_detail = os.path.join(self.tmpdir.name, "detail_chunk")
        whole = run(1_000_000, whole_detail)
        chunked = run(self.ngrid, chunk_detail)
        self.assertEqual(whole.keys(), chunked.keys())
        for key in whole:
            np.testing.assert_allclose(whole[key], chunked[key])
        for frame in range(self.nframes):
            np.testing.assert_allclose(
                np.loadtxt(f"{whole_detail}.density.out.{frame}"),
                np.loadtxt(f"{chunk_detail}.density.out.{frame}"),
            )

    def test_dp_test_mixed_type(self) -> None:
        system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "mixed-system"
        shutil.copytree(self.system, system)
        atom_types = np.array(
            [
                [0, 0, 1, 1, 1],
                [0, 1, 0, 1, 1],
                [1, 0, 0, 1, 1],
                [1, 1, 0, 0, 1],
            ],
            dtype=np.int32,
        )
        np.save(system / "set.000" / "real_atom_types.npy", atom_types)
        detail_file = os.path.join(self.tmpdir.name, "detail_mixed")
        dp_test(
            model=self.model_path,
            system=str(system),
            datafile=None,
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        set_dir = system / "set.000"
        ref = DeepDensity(self.model_path).eval(
            np.load(set_dir / "coord.npy"),
            np.load(set_dir / "box.npy"),
            atom_types,
            grid=np.load(set_dir / "grid.npy"),
            mixed_type=True,
        )
        for frame in range(self.nframes):
            detail = np.loadtxt(f"{detail_file}.density.out.{frame}", skiprows=1)
            np.testing.assert_allclose(detail[:, 1], ref[frame], rtol=1e-5, atol=1e-6)

    def test_dp_test_forwards_fparam(self) -> None:
        params = deepcopy(model_density)
        params["fitting_net"]["numb_fparam"] = 1
        model = get_model(params)
        fd, model_path = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "fparam-system"
        shutil.copytree(self.system, system)
        fparam = np.arange(1, self.nframes + 1, dtype=np.float64).reshape(-1, 1)
        np.save(system / "set.000" / "fparam.npy", fparam)
        try:
            torch.jit.save(torch.jit.script(model), model_path)
            detail_file = os.path.join(self.tmpdir.name, "detail_fparam")
            dp_test(
                model=model_path,
                system=str(system),
                datafile=None,
                numb_test=0,
                rand_seed=None,
                shuffle_test=False,
                detail_file=detail_file,
                atomic=False,
            )
            set_dir = system / "set.000"
            dp = DeepDensity(model_path)
            args = (
                np.load(set_dir / "coord.npy"),
                np.load(set_dir / "box.npy"),
                np.loadtxt(system / "type.raw", dtype=int),
            )
            grid = np.load(set_dir / "grid.npy")
            ref = dp.eval(*args, grid=grid, fparam=fparam)
            without_fparam_effect = dp.eval(
                *args, grid=grid, fparam=np.zeros_like(fparam)
            )
            self.assertFalse(np.allclose(ref, without_fparam_effect))
            for frame in range(self.nframes):
                detail = np.loadtxt(f"{detail_file}.density.out.{frame}", skiprows=1)
                np.testing.assert_allclose(
                    detail[:, 1], ref[frame], rtol=1e-5, atol=1e-6
                )
        finally:
            os.unlink(model_path)

    def test_dp_test_uses_default_fparam(self) -> None:
        params = deepcopy(model_density)
        params["fitting_net"].update(numb_fparam=1, default_fparam=[0.5])
        fd, model_path = tempfile.mkstemp(suffix=".pth")
        os.close(fd)
        default_system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "default"
        explicit_system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "explicit"
        shutil.copytree(self.system, default_system)
        shutil.copytree(self.system, explicit_system)
        np.save(
            explicit_system / "set.000" / "fparam.npy",
            np.full((self.nframes, 1), 0.5),
        )
        try:
            torch.jit.save(torch.jit.script(get_model(params)), model_path)
            detail_paths = []
            for name, system in (
                ("default", default_system),
                ("explicit", explicit_system),
            ):
                detail_file = os.path.join(self.tmpdir.name, f"detail_fparam_{name}")
                dp_test(
                    model=model_path,
                    system=str(system),
                    datafile=None,
                    numb_test=0,
                    rand_seed=None,
                    shuffle_test=False,
                    detail_file=detail_file,
                    atomic=False,
                )
                detail_paths.append(detail_file)
            for frame in range(self.nframes):
                np.testing.assert_allclose(
                    np.loadtxt(f"{detail_paths[0]}.density.out.{frame}"),
                    np.loadtxt(f"{detail_paths[1]}.density.out.{frame}"),
                )
        finally:
            os.unlink(model_path)

    def test_dp_test_without_periodic_box(self) -> None:
        system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "nopbc-system"
        shutil.copytree(self.system, system)
        (system / "set.000" / "box.npy").unlink()
        (system / "nopbc").touch()
        detail_file = os.path.join(self.tmpdir.name, "detail_nopbc")
        dp_test(
            model=self.model_path,
            system=str(system),
            datafile=None,
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        set_dir = system / "set.000"
        ref = DeepDensity(self.model_path).eval(
            np.load(set_dir / "coord.npy"),
            None,
            np.loadtxt(system / "type.raw", dtype=int),
            grid=np.load(set_dir / "grid.npy"),
        )
        for frame in range(self.nframes):
            detail = np.loadtxt(f"{detail_file}.density.out.{frame}", skiprows=1)
            np.testing.assert_allclose(detail[:, 1], ref[frame], rtol=1e-5, atol=1e-6)

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

    def test_dp_test_flattened_grid(self) -> None:
        # the loader also accepts a flattened (nframes, ngrid*3) grid; the
        # tester must normalize it to the (nframes, ngrid, 3) contract that
        # DeepDensity.eval expects, so both layouts give the same details
        system = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "system"
        shutil.copytree(self.system, system)
        set_dir = system / "set.000"
        grid3d = np.load(set_dir / "grid.npy")
        np.save(set_dir / "grid.npy", grid3d.reshape(self.nframes, -1))
        detail_file = os.path.join(self.tmpdir.name, "detail_flat")
        dp_test(
            model=self.model_path,
            system=str(system),
            datafile=None,
            numb_test=0,
            rand_seed=None,
            shuffle_test=False,
            detail_file=detail_file,
            atomic=False,
        )
        coord = np.load(set_dir / "coord.npy")
        box = np.load(set_dir / "box.npy")
        atype = np.loadtxt(system / "type.raw", dtype=int)
        ref = DeepDensity(self.model_path).eval(coord, box, atype, grid=grid3d)
        for frame in range(self.nframes):
            detail = np.loadtxt(f"{detail_file}.density.out.{frame}", skiprows=1)
            self.assertEqual(detail.shape, (self.ngrid, 2))
            np.testing.assert_allclose(detail[:, 1], ref[frame], rtol=1e-5, atol=1e-6)


model_density_dpa3_chg_spin = {
    "type_map": ["O", "H", "X"],
    "descriptor": {
        "type": "dpa3",
        "repflow": {
            "n_dim": 8,
            "e_dim": 6,
            "a_dim": 4,
            "nlayers": 1,
            "e_rcut": 4.0,
            "e_rcut_smth": 0.5,
            "e_sel": 12,
            "a_rcut": 3.5,
            "a_rcut_smth": 0.5,
            "a_sel": 8,
            "axis_neuron": 4,
            "update_angle": False,
        },
        "env_protection": 1e-6,
        "add_chg_spin_ebd": True,
        "default_chg_spin": [1.0, 0.0],
        "seed": 1,
    },
    "fitting_net": {
        "type": "density",
        "neuron": [8, 8],
        "seed": 1,
    },
}


class TestDPTestDensityChgSpin(unittest.TestCase):
    """charge_spin forwarding on the density inference and dp test paths."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        root = Path(cls.tmpdir.name)
        rng = np.random.default_rng(7)
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
        # non-default charge/spin conditions, so forwarding has a visible effect
        np.save(set_dir / "charge_spin.npy", np.tile([2.0, 1.0], (cls.nframes, 1)))
        np.savetxt(cls.system / "type.raw", [0, 0, 1, 1, 1], fmt="%d")
        np.savetxt(cls.system / "type_map.raw", ["O", "H", "X"], fmt="%s")

        model = get_model(deepcopy(model_density_dpa3_chg_spin))
        scripted = torch.jit.script(model)
        tmp_fd, cls.model_path = tempfile.mkstemp(suffix=".pth")
        os.close(tmp_fd)
        torch.jit.save(scripted, cls.model_path)

    @classmethod
    def tearDownClass(cls) -> None:
        os.unlink(cls.model_path)
        cls.tmpdir.cleanup()

    def _inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        set_dir = self.system / "set.000"
        coord = np.load(set_dir / "coord.npy")
        box = np.load(set_dir / "box.npy")
        grid = np.load(set_dir / "grid.npy")
        atype = np.loadtxt(self.system / "type.raw", dtype=int)
        return coord, box, grid, atype

    def test_eval_forwards_charge_spin(self) -> None:
        dp = DeepDensity(self.model_path)
        self.assertTrue(dp.has_chg_spin_ebd())
        self.assertTrue(dp.has_default_chg_spin())
        coord, box, grid, atype = self._inputs()
        # without explicit charge_spin the model falls back to its default
        out_default = dp.eval(coord, box, atype, grid=grid)
        out_explicit_default = dp.eval(
            coord,
            box,
            atype,
            grid=grid,
            charge_spin=np.tile([1.0, 0.0], (self.nframes, 1)),
        )
        np.testing.assert_allclose(
            out_default, out_explicit_default, rtol=1e-6, atol=1e-8
        )
        # a non-default condition must change the prediction: the value has
        # to reach the descriptor, not be dropped on the eval path
        out_other = dp.eval(
            coord,
            box,
            atype,
            grid=grid,
            charge_spin=np.tile([2.0, 1.0], (self.nframes, 1)),
        )
        self.assertFalse(np.allclose(out_default, out_other))

    def test_eval_without_charge_spin_or_default(self) -> None:
        # A charge/spin-conditioned descriptor needs either an explicit
        # condition or a configured default; fail clearly before the
        # descriptor's internal assertion.
        config = deepcopy(model_density_dpa3_chg_spin)
        config["descriptor"].pop("default_chg_spin")
        path = os.path.join(self.tmpdir.name, "no-default-charge-spin.pth")
        with torch.device("cpu"):
            torch.jit.save(torch.jit.script(get_model(config)), path)
        coord, box, grid, atype = self._inputs()
        with self.assertRaisesRegex(torch.jit.Error, "charge_spin is required"):
            DeepDensity(path).eval(coord, box, atype, grid=grid)

    def test_eval_grid_chunking_dpa3_matches_full_grid(self) -> None:
        coord, box, grid, atype = self._inputs()
        dp = DeepDensity(self.model_path)

        full = dp.eval(coord, box, atype, grid=grid, grid_batch_size=None)
        chunked = dp.eval(coord, box, atype, grid=grid, grid_batch_size=3)

        np.testing.assert_allclose(chunked, full, rtol=1e-6, atol=1e-8)

    def test_dp_test_charge_spin_data_and_default(self) -> None:
        coord, box, grid, atype = self._inputs()
        dp = DeepDensity(self.model_path)
        explicit_ref = dp.eval(
            coord,
            box,
            atype,
            grid=grid,
            charge_spin=np.tile([2.0, 1.0], (self.nframes, 1)),
        )
        default_ref = dp.eval(coord, box, atype, grid=grid)
        self.assertFalse(
            np.allclose(explicit_ref, default_ref),
            "the test is vacuous: charge_spin has no effect on this model",
        )

        system_without_data = Path(tempfile.mkdtemp(dir=self.tmpdir.name)) / "system"
        shutil.copytree(self.system, system_without_data)
        (system_without_data / "set.000" / "charge_spin.npy").unlink()
        cases = (
            ("explicit data", self.system, explicit_ref),
            ("model default", system_without_data, default_ref),
        )
        for case, system, reference in cases:
            detail_file = os.path.join(self.tmpdir.name, f"detail_cs_{case}")
            with self.subTest(case=case):
                dp_test(
                    model=self.model_path,
                    system=str(system),
                    datafile=None,
                    numb_test=0,
                    rand_seed=None,
                    shuffle_test=False,
                    detail_file=detail_file,
                    atomic=False,
                )
                for frame in range(self.nframes):
                    detail = np.loadtxt(
                        f"{detail_file}.density.out.{frame}", skiprows=1
                    )
                    self.assertEqual(detail.shape, (self.ngrid, 2))
                    np.testing.assert_allclose(
                        detail[:, 1], reference[frame], rtol=1e-5, atol=1e-6
                    )


if __name__ == "__main__":
    unittest.main()
