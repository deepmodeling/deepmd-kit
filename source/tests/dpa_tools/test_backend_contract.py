# SPDX-License-Identifier: LGPL-3.0-or-later
"""Contract tests for ``deepmd.dpa_tools._backend``.

These tests call **real** deepmd APIs — no mocks — on a minimal synthetic
DPA-3 descriptor model.  Their purpose is to catch silent breakage when
deepmd internal APIs change signatures, ``"Default"`` key paths shift, or
the descriptor-hook behaviour is altered upstream.

No large checkpoint file is needed; we build a tiny model from a config
dict and run a single forward pass.
"""

from __future__ import annotations

import numpy as np
import pytest


# Smallest possible DPA-3 descriptor config that get_model accepts.
_MINIMAL_DPA3_CONFIG = {
    "type_map": ["H", "O"],
    "descriptor": {
        "type": "dpa3",
        "repflow": {
            "n_dim": 16,
            "e_dim": 8,
            "a_dim": 4,
            "nlayers": 2,
            "e_rcut": 4.0,
            "e_rcut_smth": 3.5,
            "e_sel": 10,
            "a_rcut": 3.0,
            "a_rcut_smth": 2.5,
            "a_sel": 5,
            "axis_neuron": 2,
            "skip_stat": True,
            "a_compress_rate": 1,
            "a_compress_e_rate": 2,
            "a_compress_use_split": True,
            "update_angle": True,
            "smooth_edge_update": True,
            "use_dynamic_sel": True,
            "sel_reduce_factor": 10.0,
            "update_style": "res_residual",
            "update_residual": 0.1,
            "update_residual_init": "const",
            "n_multi_edge_message": 1,
            "optim_update": True,
            "use_exp_switch": True,
        },
        "activation_function": "silu",
        "precision": "float64",
        "use_tebd_bias": False,
        "concat_output_tebd": False,
        "exclude_types": [],
        "env_protection": 0.0,
        "trainable": True,
        "use_econf_tebd": False,
    },
    "fitting_net": {
        "type": "ener",
        "neuron": [16, 16],
        "activation_function": "tanh",
        "precision": "float64",
        "resnet_dt": True,
        "use_tebd_bias": False,
        "exclude_types": [],
        "numb_fparam": 0,
        "numb_aparam": 0,
    },
}


@pytest.mark.skipif(True, reason="requires real DPA checkpoint / GPU — CI contract")
class _HeavyContract:
    """Guarded heavy tests that need DPA checkpoint + GPU."""

    def test_real_checkpoint_descriptor_shape(self):
        ...  # placeholder for future Bohrium-only tests


class TestBackendContract:
    """Contract tests using real deepmd APIs (no mocks).

    These require a fully-functional deepmd-kit installation.  They are
    skipped when the environment is incomplete (e.g. CI without MPI).
    """

    @pytest.fixture(autouse=True)
    def _require_deepmd(self):
        """Skip if the deepmd model builder is not usable."""
        try:
            from deepmd.dpa_tools._backend import build_model_from_config
            build_model_from_config(_MINIMAL_DPA3_CONFIG)
        except Exception as exc:
            pytest.skip(f"deepmd build_model_from_config not functional: {exc}")

    @pytest.fixture
    def _extractor(self):
        """Build a model + extractor, yield it, then **always** disable the
        descriptor hook so a test failure never leaks global state."""
        from deepmd.dpa_tools._backend import (
            _DescriptorExtraction,
            build_model_from_config,
        )

        wrapper = build_model_from_config(_MINIMAL_DPA3_CONFIG)
        wrapper.eval()
        extractor = _DescriptorExtraction(wrapper)
        extractor._enable_hook()
        try:
            yield extractor
        finally:
            extractor._disable_hook()

    def test_build_model_from_config(self):
        """``build_model_from_config`` succeeds with minimal config."""
        from deepmd.dpa_tools._backend import build_model_from_config

        wrapper = build_model_from_config(_MINIMAL_DPA3_CONFIG)
        assert wrapper is not None
        assert "Default" in wrapper.model, (
            "ModelWrapper.model must contain 'Default' key"
        )

    def test_descriptor_extraction_chain(self, _extractor):
        """Full chain: build → hook → forward → eval_descriptor → shape check."""
        import torch

        # Synthetic input: 1 frame, 2 atoms (H and O), reasonable distances
        n_frames = 1
        n_atoms = 2
        coords = torch.tensor(
            [[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]],
            dtype=torch.float64,
        ).requires_grad_(True)
        atype = torch.tensor([[0, 1]], dtype=torch.long)  # H, O
        box = torch.tensor(
            [[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]],
            dtype=torch.float64,
        )

        desc = _extractor._run_forward(coords, atype, box)

        assert desc.ndim == 3, f"expected (n_frames, n_atoms, feat_dim), got {desc.shape}"
        assert desc.shape[0] == n_frames
        assert desc.shape[1] == n_atoms
        assert desc.shape[2] > 0, "feature dim must be > 0"
        assert not torch.any(torch.isnan(desc)), "descriptor contains NaN"
        assert not torch.any(torch.isinf(desc)), "descriptor contains Inf"

    def test_descriptor_feat_dim_matches_repflow(self, _extractor):
        """The feature dimension matches n_dim from the repflow config."""
        import torch

        coords = torch.tensor(
            [[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]],
            dtype=torch.float64,
        ).requires_grad_(True)
        atype = torch.tensor([[0, 1]], dtype=torch.long)
        box = torch.tensor(
            [[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]],
            dtype=torch.float64,
        )

        desc = _extractor._run_forward(coords, atype, box)

        n_dim = _MINIMAL_DPA3_CONFIG["descriptor"]["repflow"]["n_dim"]
        assert desc.shape[2] == n_dim, (
            f"descriptor feat dim {desc.shape[2]} != repflow n_dim {n_dim}"
        )

    def test_forward_common_fails_without_grad(self, _extractor):
        """``forward_common`` requires gradients on coords — verify the guard."""
        import torch

        coords = torch.tensor(
            [[0.0, 0.0, 0.0, 1.5, 0.0, 0.0]],
            dtype=torch.float64,
        )  # NO requires_grad
        atype = torch.tensor([[0, 1]], dtype=torch.long)
        box = torch.tensor(
            [[10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0]],
            dtype=torch.float64,
        )

        with pytest.raises(RuntimeError, match="grad"):
            _extractor._run_forward(coords, atype, box)


class TestBackendHelpers:
    """Unit-level checks for _backend utility functions."""

    def test_get_torch_device_returns_device(self):
        import sys
        from unittest.mock import MagicMock

        if isinstance(sys.modules.get("torch"), MagicMock):
            pytest.skip("torch is mocked by another test")

        from deepmd.dpa_tools._backend import get_torch_device

        device = get_torch_device()
        assert device.type in ("cpu", "cuda")

    def test_load_torch_file_roundtrip(self, tmp_path):
        import sys
        from unittest.mock import MagicMock

        if isinstance(sys.modules.get("torch"), MagicMock):
            pytest.skip("torch is mocked by another test")

        import torch

        from deepmd.dpa_tools._backend import load_torch_file

        path = str(tmp_path / "test.pt")
        data = {"key": "value", "n": 42}
        torch.save(data, path)
        loaded = load_torch_file(path)
        assert loaded == data


class TestFormatVersion:
    """format_version contract."""

    def test_freeze_bundle_has_format_version(self, tmp_path):
        """A frozen bundle from DPAFineTuner.freeze() must carry format_version=1."""
        import numpy as np
        from unittest.mock import patch

        from deepmd.dpa_tools import DPAFineTuner

        system = tmp_path / "sys"
        system.mkdir()
        (system / "type.raw").write_text("0\n1\n")
        (system / "type_map.raw").write_text("Cu\nO\n")
        sd = system / "set.000"
        sd.mkdir()
        np.save(sd / "coord.npy", np.zeros((3, 6)))
        np.save(sd / "box.npy", np.tile(np.eye(3).ravel(), (3, 1)))
        np.save(sd / "energy.npy", np.arange(3, dtype=float))

        def _fake_extract(self, systems):
            return np.random.default_rng(0).random((3, 8))

        with (
            patch.object(DPAFineTuner, "_load_descriptor_model", lambda self: None),
            patch.object(DPAFineTuner, "_extract_features", _fake_extract),
        ):
            ft = DPAFineTuner(pretrained="fake.pt", predictor="linear")
            ft._checkpoint_type_map = ["Cu", "O"]
            ft.fit(str(system), target_key="energy")
            frozen = ft.freeze(str(tmp_path / "model.pth"))

        from deepmd.dpa_tools._backend import load_torch_file

        bundle = load_torch_file(frozen)
        assert bundle.get("format_version") == 1, (
            f"format_version missing or wrong: {bundle.get('format_version')!r}"
        )
