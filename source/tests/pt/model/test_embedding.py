# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for embedding extraction (descriptor, atomic and structural features)."""

import os
import shutil
import tempfile
import unittest
from types import (
    SimpleNamespace,
)

import h5py
import numpy as np
import torch
from packaging.version import parse as parse_version

from deepmd.infer.deep_pot import (
    DeepPot,
)
from deepmd.pt.infer.deep_eval import DeepEval as PTDeepEval
from deepmd.pt.model.model import (
    get_model,
    get_sezm_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt.utils import (
    env,
)

# The SeZM compile path is validated on torch 2.11.x / 2.12.x only.
_TORCH_VERSION = parse_version(torch.__version__)
_SKIP_COMPILE = (_TORCH_VERSION.major, _TORCH_VERSION.minor) not in {(2, 11), (2, 12)}
_SKIP_COMPILE_REASON = (
    "SeZM's torch.compile path is only supported on torch 2.11.x and 2.12.x."
)


def _sezm_params() -> dict:
    """Return a small SeZM/DPA4 model configuration for fast tests."""
    return {
        "type": "SeZM",
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "SeZM",
            "sel": [2, 2],
            "rcut": 3.0,
            "channels": 4,
            "n_focus": 1,
            "n_radial": 3,
            "radial_mlp": [6],
            "use_env_seed": True,
            "l_schedule": [1, 0],
            "mmax": 1,
            "so2_layers": 1,
            "n_atten_head": 1,
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "use_amp": False,
            "precision": "float32",
            "seed": 7,
        },
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float32",
            "seed": 7,
        },
    }


def _se_e2_a_params() -> dict:
    """Return a small standard ``se_e2_a`` energy model configuration."""
    return {
        "type_map": ["A", "B"],
        "descriptor": {
            "type": "se_e2_a",
            "sel": [4, 4],
            "rcut_smth": 0.5,
            "rcut": 3.0,
            "neuron": [4, 8],
            "axis_neuron": 4,
            "seed": 1,
        },
        "fitting_net": {
            "neuron": [8],
            "seed": 1,
        },
    }


def _randomize(model: torch.nn.Module, seed: int = 1234) -> None:
    """Fill parameters with small random values to expose masked paths."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn_like(param) * 0.1)


def _make_frame(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one deterministic 7-atom frame (coord, atype, box)."""
    coord = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.3, 0.0],
                [0.2, 1.5, 0.4],
                [1.7, 1.2, 0.2],
                [2.3, 0.1, 1.0],
                [0.8, 2.2, 1.1],
                [2.6, 1.8, 1.5],
            ],
        ],
        device=device,
        dtype=torch.float32,
    )
    atype = torch.tensor([[0, 1, 0, 1, 0, 1, 0]], device=device, dtype=torch.int32)
    box = torch.tensor(
        [[8.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 8.0]],
        device=device,
        dtype=torch.float32,
    )
    return coord, atype, box


class TestSeZMEmbeddingForward(unittest.TestCase):
    """Validate ``forward_embedding`` against the independent energy forward."""

    def setUp(self) -> None:
        torch.manual_seed(2024)
        self.device = env.DEVICE
        self.model = get_sezm_model(_sezm_params()).to(self.device)
        _randomize(self.model)
        self.model.eval()
        self.coord, self.atype, self.box = _make_frame(self.device)

    def test_embedding_reconstructs_energy(self) -> None:
        # Projecting the atomic / structural features through the fitting output
        # layer reproduces the per-atom / total energy of the independent energy
        # forward. With the per-type bias zeroed and no excluded atoms, the
        # output layer is linear, so this validates both the atomic feature (it
        # is the last hidden activation) and the structural feature (it is the
        # atom-pooled feature).
        fitting = self.model.atomic_model.fitting_net
        with torch.no_grad():
            fitting.bias_atom_e.zero_()
        out = self.model(self.coord, self.atype, box=self.box)
        emb = self.model.forward_embedding(self.coord, self.atype, box=self.box)
        output_layer = fitting.filter_layers.networks[0].output_layer

        recon_atom = output_layer(emb["atomic_feature"].to(fitting.prec))
        torch.testing.assert_close(
            recon_atom.double(),
            out["atom_energy"].double(),
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
        )
        recon_total = output_layer(emb["structural_feature"].to(fitting.prec))
        torch.testing.assert_close(
            recon_total.flatten().double(),
            out["energy"].flatten().double(),
            atol=1e-4,
            rtol=1e-4,
            check_dtype=False,
        )

    @unittest.skipIf(_SKIP_COMPILE, _SKIP_COMPILE_REASON)
    def test_compiled_matches_eager(self) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"DP_COMPILE_INFER": "1"}, clear=False
        ):
            model_cmp = get_sezm_model(_sezm_params()).to(self.device)
        model_cmp.load_state_dict(self.model.state_dict())
        model_cmp.eval()
        self.assertTrue(model_cmp.should_use_compile())

        eager = self.model.forward_embedding(self.coord, self.atype, box=self.box)
        compiled = model_cmp.forward_embedding(self.coord, self.atype, box=self.box)
        # Inductor reductions can differ from eager by ~1e-3 in float32 on GPU.
        atol = 1e-5 if self.device == torch.device("cpu") else 2e-3
        rtol = 1e-5 if self.device == torch.device("cpu") else 3e-3
        for key in ("descriptor", "atomic_feature", "structural_feature"):
            torch.testing.assert_close(
                eager[key], compiled[key], atol=atol, rtol=rtol, msg=key
            )


class TestEmbeddingDeepEvalAPI(unittest.TestCase):
    """Validate the ``DeepEval`` embedding API and the unsupported boundary."""

    def setUp(self) -> None:
        torch.manual_seed(2024)
        self.device = env.DEVICE
        self._tmp = tempfile.mkdtemp()
        coord, atype, box = _make_frame(self.device)
        self.coord_np = coord.cpu().numpy()
        self.cell_np = box.cpu().numpy()
        self.atype_np = atype[0].cpu().numpy()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _save_checkpoint(self, model: torch.nn.Module, params: dict, name: str) -> str:
        path = os.path.join(self._tmp, name)
        wrapper = ModelWrapper(model, model_params=params)
        torch.save({"model": wrapper.state_dict()}, path)
        return path

    def test_sezm_eval_embedding(self) -> None:
        params = _sezm_params()
        model = get_sezm_model(params)
        _randomize(model)
        path = self._save_checkpoint(model, params, "sezm.pt")

        dp = DeepPot(path)
        descriptor, atomic_feature, structural_feature = dp.eval_embedding(
            self.coord_np, self.cell_np, self.atype_np
        )

        natoms = int(self.atype_np.shape[0])
        self.assertEqual(descriptor.shape[:2], (1, natoms))
        self.assertEqual(atomic_feature.shape[:2], (1, natoms))
        self.assertEqual(structural_feature.shape, (1, atomic_feature.shape[2]))
        # Embeddings are returned as float32.
        self.assertEqual(descriptor.dtype, np.float32)
        self.assertEqual(atomic_feature.dtype, np.float32)
        self.assertEqual(structural_feature.dtype, np.float32)

    def test_standard_model_embedding(self) -> None:
        # Standard energy models reuse the base-class embedding path; the
        # descriptor and atomic feature flow through the same neighbor list as
        # the energy forward.
        params = _se_e2_a_params()
        model = get_model(params)
        _randomize(model)
        path = self._save_checkpoint(model, params, "se_e2_a.pt")

        dp = DeepPot(path)
        descriptor, atomic_feature, structural_feature = dp.eval_embedding(
            self.coord_np, self.cell_np, self.atype_np
        )

        natoms = int(self.atype_np.shape[0])
        self.assertEqual(descriptor.shape[:2], (1, natoms))
        self.assertEqual(atomic_feature.shape[:2], (1, natoms))
        self.assertEqual(structural_feature.shape, (1, atomic_feature.shape[2]))
        # Embeddings are returned as float32.
        self.assertEqual(descriptor.dtype, np.float32)
        self.assertEqual(atomic_feature.dtype, np.float32)
        self.assertEqual(structural_feature.dtype, np.float32)
        # The structural feature is the atom-sum of the atomic feature.
        np.testing.assert_allclose(
            structural_feature[0],
            atomic_feature[0].sum(axis=0),
            rtol=1e-4,
            atol=1e-5,
        )
        # eval_descriptor / eval_fitting_last_layer are thin wrappers that slice
        # the embedding output (independent forward passes match to round-off).
        np.testing.assert_allclose(
            dp.eval_descriptor(
                self.coord_np, self.cell_np, self.atype_np, dtype="fp32"
            ),
            descriptor,
            rtol=1e-10,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            dp.eval_fitting_last_layer(
                self.coord_np, self.cell_np, self.atype_np, dtype="fp32"
            ),
            atomic_feature,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_standard_model_embedding_without_hidden_layers(self) -> None:
        params = _se_e2_a_params()
        params["fitting_net"]["neuron"] = []
        model = get_model(params)
        _randomize(model)
        path = self._save_checkpoint(model, params, "se_e2_a_no_hidden.pt")

        dp = DeepPot(path)
        descriptor, atomic_feature, structural_feature = dp.eval_embedding(
            self.coord_np, self.cell_np, self.atype_np
        )

        self.assertEqual(atomic_feature.shape, descriptor.shape)
        self.assertEqual(structural_feature.shape, (1, atomic_feature.shape[2]))
        self.assertEqual(atomic_feature.dtype, np.float32)

    def test_eval_embedding_dtype_fp64(self) -> None:
        params = _se_e2_a_params()
        model = get_model(params)
        path = self._save_checkpoint(model, params, "se_e2_a_fp64.pt")

        dp = DeepPot(path)
        descriptor, atomic_feature, structural_feature = dp.eval_embedding(
            self.coord_np, self.cell_np, self.atype_np, dtype="fp64"
        )

        self.assertEqual(descriptor.dtype, np.float64)
        self.assertEqual(atomic_feature.dtype, np.float64)
        self.assertEqual(structural_feature.dtype, np.float64)
        self.assertEqual(
            dp.eval_descriptor(
                self.coord_np, self.cell_np, self.atype_np, dtype="fp64"
            ).dtype,
            np.float64,
        )

    def test_legacy_frozen_model_uses_baked_in_hook(self) -> None:
        # Frozen ``.pth`` files predating ``forward_embedding`` still carry the
        # descriptor / fitting hooks baked into the TorchScript module. The
        # deprecated ``eval_descriptor`` / ``eval_fitting_last_layer`` must drive
        # those hooks instead of raising, so existing artifacts keep working.
        toggles: list[tuple[str, bool]] = []
        with torch.device("cpu"):
            desc = torch.arange(6, dtype=torch.float64).reshape(1, 2, 3)
            fit = torch.arange(8, dtype=torch.float64).reshape(1, 2, 4)

        class LegacyModel:
            def set_eval_descriptor_hook(self, enable: bool) -> None:
                toggles.append(("descriptor", enable))

            def eval_descriptor(self) -> torch.Tensor:
                return desc

            def set_eval_fitting_last_layer_hook(self, enable: bool) -> None:
                toggles.append(("fitting", enable))

            def eval_fitting_last_layer(self) -> torch.Tensor:
                return fit

        backend = object.__new__(PTDeepEval)
        backend.auto_batch_size = None
        backend.dp = SimpleNamespace(model={"Default": LegacyModel()})
        # The hook caches are populated by the (here mocked) forward pass.
        backend.eval = lambda *args, **kwargs: None

        np.testing.assert_array_equal(
            PTDeepEval.eval_descriptor(
                backend, self.coord_np, self.cell_np, self.atype_np
            ),
            desc.numpy(),
        )
        np.testing.assert_array_equal(
            PTDeepEval.eval_fitting_last_layer(
                backend, self.coord_np, self.cell_np, self.atype_np
            ),
            fit.numpy(),
        )
        # Each hook is enabled for the forward pass and disabled afterwards.
        self.assertEqual(
            toggles,
            [
                ("descriptor", True),
                ("descriptor", False),
                ("fitting", True),
                ("fitting", False),
            ],
        )


class TestEmbeddingEntrypoint(unittest.TestCase):
    """Validate the HDF5 entrypoint without requiring on-disk systems."""

    def setUp(self) -> None:
        self._tmp = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_embedding_writes_hdf5_from_datafile(self) -> None:
        import importlib

        embedding_module = importlib.import_module("deepmd.entrypoints.embedding")
        datafile = os.path.join(self._tmp, "systems.txt")
        output = os.path.join(self._tmp, "embedding.hdf5")
        with open(datafile, "w") as fp:
            fp.write("/tmp/a/sys\n\n/tmp/b/sys\n")

        class FakeData:
            mixed_type = False
            pbc = False

            def __init__(self, *args, **kwargs) -> None:
                pass

            def get_test(self) -> dict[str, np.ndarray]:
                return {
                    "coord": np.zeros((1, 2, 3), dtype=np.float64),
                    "box": np.zeros((1, 9), dtype=np.float64),
                    "type": np.array([[0, 1]], dtype=np.int32),
                }

        descriptor = np.zeros((1, 2, 3), dtype=np.float64)
        atomic_feature = np.ones((1, 2, 4), dtype=np.float64)
        structural_feature = atomic_feature.sum(axis=1)
        dp = unittest.mock.Mock()
        dp.get_type_map.return_value = ["A", "B"]
        dp.get_dim_fparam.return_value = 0
        dp.get_dim_aparam.return_value = 0
        dp.eval_embedding.return_value = (
            descriptor,
            atomic_feature,
            structural_feature,
        )

        with (
            unittest.mock.patch.object(embedding_module, "DeepEval", return_value=dp),
            unittest.mock.patch.object(embedding_module, "DeepmdData", FakeData),
        ):
            embedding_module.embedding(
                model="model.pt",
                system=".",
                datafile=datafile,
                output=output,
                dtype="fp64",
            )

        self.assertEqual(dp.eval_embedding.call_count, 2)
        self.assertIsNone(dp.eval_embedding.call_args.args[1])
        self.assertEqual(dp.eval_embedding.call_args.kwargs["dtype"], "fp64")
        with h5py.File(output, "r") as h5file:
            self.assertEqual(set(h5file.keys()), {"sys", "sys_1"})
            self.assertEqual(h5file["sys"]["descriptor"].dtype, np.float64)
            self.assertEqual(h5file["sys_1"].attrs["system"], "/tmp/b/sys")


if __name__ == "__main__":
    unittest.main()
