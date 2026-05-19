# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for SeZM's AOTInductor ``.pt2`` freeze pipeline.

Layout mirrors ``source/tests/pt_expt/model/test_export_pipeline.py``:
a tiny fp64 SeZM model is built on the fly, so the tests are fully
self-contained and have no external-artefact dependency.
"""

from __future__ import (
    annotations,
)

import contextlib
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import (
    Path,
)
from typing import (
    TYPE_CHECKING,
)
from unittest import (
    mock,
)

import numpy as np
import torch

from deepmd.pt.entrypoints.freeze_pt2 import (
    _build_dynamic_shapes,
    _collect_metadata,
    _make_sample_inputs,
    _resolve_nframes,
    freeze_sezm_to_pt2,
    is_sezm_checkpoint,
)
from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.train.wrapper import (
    ModelWrapper,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

# Tracing and numerical parity always run on CPU — see module docstring
# of deepmd/pt/entrypoints/freeze_pt2.py for why.
_CPU = torch.device("cpu")

_REQUIRED_OUTPUT_KEYS = {
    "energy",
    "energy_redu",
    "energy_derv_r",
    "energy_derv_c",
    "energy_derv_c_redu",
}


def _tiny_sezm_model_params() -> dict:
    """Minimal fp64 SeZM config for self-contained export tests.

    ``precision="float64"`` is what unlocks the ``rtol=1e-10, atol=1e-10``
    parity pt_expt enforces; fp32 accumulation alone drifts in the 1e-6
    range.  All other knobs are tuned to keep ``make_fx`` tracing time
    in the low-single-digit seconds.
    """
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
            "so2_norm": False,
            "so2_layers": 1,
            "n_atten_head": 1,
            "sandwich_norm": [True, False, True, False],
            "ffn_neurons": 8,
            "ffn_blocks": 1,
            "s2_activation": [False, True],
            "mlp_bias": False,
            "layer_scale": False,
            "use_amp": False,
            "activation_function": "silu",
            "glu_activation": True,
            "precision": "float64",
            "seed": 7,
        },
        "fitting_net": {
            "neuron": [8],
            "activation_function": "silu",
            "precision": "float64",
            "seed": 7,
        },
        "use_compile": False,
    }


def _tiny_sezm_spin_model_params() -> dict:
    """Minimal fp64 SeZM spin config for freeze routing tests."""
    params = copy.deepcopy(_tiny_sezm_model_params())
    params["type_map"] = ["O", "H"]
    params["spin"] = {
        "use_spin": [True, False],
        "virtual_scale": 0.2,
    }
    return params


def _build_tiny_sezm_model() -> torch.nn.Module:
    """Fresh tiny SeZM model on CPU, in eval mode."""
    model = get_model(_tiny_sezm_model_params())
    model.eval()
    model.to(_CPU)
    return model


def _write_tiny_sezm_checkpoint(tmp_path: Path, params: dict) -> Path:
    """Serialise a tiny SeZM model to a ``.pt`` in the trainer's layout.

    ``ModelWrapper`` populates ``state_dict["_extra_state"]`` from its
    ``get_extra_state`` hook, which is exactly the shape
    :func:`freeze_sezm_to_pt2` expects.
    """
    model = get_model(params)
    model.eval()
    model.to(_CPU)
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(params))
    ckpt_path = tmp_path / "tiny_sezm.pt"
    torch.save({"model": wrapper.state_dict()}, ckpt_path)
    return ckpt_path


def _make_sample(model: torch.nn.Module, *, nloc: int, start: int) -> tuple:
    """Build a forward_common_lower sample on CPU via the freeze helper."""
    _, sample = _resolve_nframes(model, nloc=nloc, device=_CPU, start=start)
    return sample


@contextlib.contextmanager
def _clear_default_device() -> Iterator[None]:
    """Clear the pt-test ``cuda:9999999`` sentinel default device.

    ``source/tests/pt/__init__.py`` sets the default device to an
    invalid ``"cuda:9999999"`` so that tests relying on implicit
    placement fail loudly.  The AOTI / export pipeline in PyTorch 2.11
    allocates unnamed tensors (e.g. inside ``PhiloxStateTracker``)
    without an explicit device and would trip the guard.  Matches the
    pattern used by ``pt_expt/test_change_bias.py``.
    """
    saved = torch.get_default_device()
    torch.set_default_device(None)
    try:
        yield
    finally:
        torch.set_default_device(saved)


def _eager_forward(
    model: torch.nn.Module,
    sample_inputs: tuple,
) -> dict[str, torch.Tensor]:
    """Mirror the trace closure: fresh leaf coord + ``requires_grad=True``."""
    ext_coord, ext_atype, nlist, mapping, fparam, aparam = sample_inputs
    eager_coord = ext_coord.detach().clone().requires_grad_(True)
    return model.forward_common_lower(
        eager_coord,
        ext_atype,
        nlist,
        mapping=mapping,
        fparam=fparam,
        aparam=aparam,
        do_atomic_virial=True,
        extra_nlist_sort=model.need_sorted_nlist_for_lower(),
    )


class TestSeZMExportPipeline(unittest.TestCase):
    """Bitwise trace / export / ``.pte`` round-trip parity (``rtol=1e-10``).

    The ExportedProgram is a pure FX graph (no Inductor codegen), so
    it must reproduce the eager result exactly.  Drift here implies a
    bug in ``forward_common_lower_exportable`` or the dynamic-shape
    spec, not in AOTI.  The pipeline is built once per class because
    ``make_fx`` and ``.pte`` round-trip dominate wall time.
    """

    @classmethod
    def setUpClass(cls) -> None:
        with _clear_default_device():
            cls.model = _build_tiny_sezm_model()
            cls.sample_inputs = _make_sample(cls.model, nloc=7, start=2)
            cls.traced, cls.loaded, cls._pte_tmp = cls._build_pipeline(
                cls.model, cls.sample_inputs
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._pte_tmp.close()

    def setUp(self) -> None:
        self._device_ctx = _clear_default_device()
        self._device_ctx.__enter__()

    def tearDown(self) -> None:
        self._device_ctx.__exit__(None, None, None)

    @staticmethod
    def _build_pipeline(
        model: torch.nn.Module,
        sample_inputs: tuple,
    ) -> tuple[
        torch.fx.GraphModule,
        torch.nn.Module,
        tempfile._TemporaryFileWrapper,
    ]:
        traced = model.forward_common_lower_exportable(
            *sample_inputs,
            do_atomic_virial=True,
        )
        exported = torch.export.export(
            traced,
            sample_inputs,
            dynamic_shapes=_build_dynamic_shapes(sample_inputs),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
        # Keep the tempfile alive for the class lifetime so the loaded
        # module can lazily reference its backing bytes.
        pte_tmp = tempfile.NamedTemporaryFile(suffix=".pte", delete=True)
        torch.export.save(exported, pte_tmp.name)
        loaded = torch.export.load(pte_tmp.name).module()
        return traced, loaded, pte_tmp

    def _assert_dict_allclose(
        self,
        ref: dict[str, torch.Tensor],
        test_dict: dict[str, torch.Tensor] | object,
        *,
        context: str,
    ) -> None:
        test_pairs = (
            list(test_dict.items())
            if hasattr(test_dict, "items")
            else list(zip(ref.keys(), test_dict, strict=True))
        )
        for key, test_val in test_pairs:
            self.assertIn(key, ref, msg=f"{context}: unexpected output key {key!r}")
            ref_val = ref[key]
            self.assertEqual(
                tuple(ref_val.shape),
                tuple(test_val.shape),
                msg=(
                    f"{context} ({key}): shape mismatch "
                    f"ref={tuple(ref_val.shape)} vs test={tuple(test_val.shape)}"
                ),
            )
            np.testing.assert_allclose(
                ref_val.detach().cpu().numpy(),
                test_val.detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"{context}: {key}",
            )

    def test_traced_matches_eager_same_shape(self) -> None:
        eager_out = _eager_forward(self.model, self.sample_inputs)
        traced_out = self.traced(*self.sample_inputs)
        self._assert_dict_allclose(
            eager_out, traced_out, context="traced vs eager (trace shape)"
        )

    def test_loaded_pte_matches_eager_same_shape(self) -> None:
        eager_out = _eager_forward(self.model, self.sample_inputs)
        loaded_out = self.loaded(*self.sample_inputs)
        self._assert_dict_allclose(
            eager_out, loaded_out, context="loaded (.pte) vs eager (trace shape)"
        )

    def test_loaded_pte_matches_eager_different_shape(self) -> None:
        # start=3 retargets the nframes symbol away from the trace
        # value of 2; nloc=11 exercises the nloc symbol.
        infer_inputs = _make_sample(self.model, nloc=11, start=3)
        eager_out = _eager_forward(self.model, infer_inputs)
        loaded_out = self.loaded(*infer_inputs)
        self._assert_dict_allclose(
            eager_out, loaded_out, context="loaded (.pte) vs eager (infer shape)"
        )


class _FrozenPt2Fixture:
    """Shared setUp/tearDown: freeze a tiny SeZM checkpoint to ``.pt2`` once.

    AOTInductor compilation costs a few seconds; classes that share this
    fixture avoid paying that cost twice.  ``cls.ckpt_path`` / ``cls.out_path``
    / ``cls.params`` are populated and live for the lifetime of the class.
    """

    params: dict
    ckpt_path: Path
    out_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory()
        tmp_root = Path(cls._tmpdir.name)
        cls.params = _tiny_sezm_model_params()
        with _clear_default_device():
            cls.ckpt_path = _write_tiny_sezm_checkpoint(tmp_root, cls.params)
            cls.out_path = tmp_root / "frozen_sezm.pt2"
            freeze_sezm_to_pt2(str(cls.ckpt_path), str(cls.out_path), device=_CPU)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        self._device_ctx = _clear_default_device()
        self._device_ctx.__enter__()

    def tearDown(self) -> None:
        self._device_ctx.__exit__(None, None, None)


class TestSeZMExportArchive(_FrozenPt2Fixture, unittest.TestCase):
    """AOTI ``.pt2`` archive structure + load-and-run smoke.

    Numerical parity of the compiled ``.pt2`` is covered by the
    pipeline class through the ``.pte`` round-trip; here we only
    verify the archive layout and the C++ consumer contract.
    """

    def test_detector_recognises_sezm(self) -> None:
        self.assertTrue(is_sezm_checkpoint(str(self.ckpt_path)))

    def test_archive_metadata(self) -> None:
        """ZIP layout + metadata fields match the DeepPotPTExpt contract."""
        self.assertTrue(zipfile.is_zipfile(str(self.out_path)))
        with zipfile.ZipFile(str(self.out_path), "r") as zf:
            names = zf.namelist()
            self.assertIn("model/extra/metadata.json", names)
            self.assertIn("model/extra/model_def_script.json", names)
            metadata = json.loads(zf.read("model/extra/metadata.json").decode("utf-8"))
            mds = json.loads(
                zf.read("model/extra/model_def_script.json").decode("utf-8")
            )

        for key in (
            "type_map",
            "ntypes",
            "rcut",
            "sel",
            "dim_fparam",
            "dim_aparam",
            "dim_chg_spin",
            "mixed_types",
            "has_default_fparam",
            "default_chg_spin",
            "output_keys",
            "fitting_output_defs",
            "sel_type",
            "is_spin",
        ):
            self.assertIn(key, metadata)

        self.assertEqual(metadata["type_map"], self.params["type_map"])
        self.assertEqual(metadata["ntypes"], len(self.params["type_map"]))
        self.assertEqual(metadata["rcut"], self.params["descriptor"]["rcut"])
        self.assertEqual(list(metadata["sel"]), list(self.params["descriptor"]["sel"]))
        self.assertTrue(metadata["mixed_types"])
        self.assertFalse(metadata["is_spin"])
        self.assertEqual(metadata["dim_fparam"], 0)
        self.assertEqual(metadata["dim_aparam"], 0)
        self.assertEqual(metadata["dim_chg_spin"], 0)
        self.assertIsNone(metadata["default_chg_spin"])
        # sel_type must agree with the eager SeZM model — this is the
        # field DeepEval._init_from_metadata reads when no model.json is
        # present.  DPA4 / SeZM's dpa4_ener fitting head enumerates every type,
        # so the list is non-empty in general.
        probe = _build_tiny_sezm_model()
        self.assertEqual(list(metadata["sel_type"]), list(probe.get_sel_type()))
        self.assertTrue(_REQUIRED_OUTPUT_KEYS.issubset(set(metadata["output_keys"])))

        # model_def_script preserves the training params verbatim.
        self.assertEqual(str(mds.get("type", "")).lower(), "sezm")
        self.assertEqual(mds.get("use_compile"), self.params["use_compile"])

    def test_aoti_load_and_run_returns_finite_outputs(self) -> None:
        from torch._inductor import (
            aoti_load_package,
        )

        loader = aoti_load_package(str(self.out_path))
        probe = _build_tiny_sezm_model()
        sample_inputs = _make_sample(probe, nloc=5, start=2)
        outs = loader(*sample_inputs)

        # AOTICompiledModel returns an immutable_dict on PyTorch ≥2.11
        # and a flat tuple on older versions; normalise both.
        with zipfile.ZipFile(str(self.out_path), "r") as zf:
            output_keys = json.loads(
                zf.read("model/extra/metadata.json").decode("utf-8")
            )["output_keys"]
        if hasattr(outs, "items"):
            out_map = dict(outs.items())
            self.assertEqual(list(out_map.keys()), output_keys)
        else:
            self.assertEqual(len(outs), len(output_keys))
            out_map = dict(zip(output_keys, outs, strict=True))

        for key in ("energy_redu", "energy_derv_r", "energy_derv_c_redu"):
            self.assertIn(key, out_map)
            self.assertTrue(torch.isfinite(out_map[key]).all().item())


class TestSeZMViaDeepPot(_FrozenPt2Fixture, unittest.TestCase):
    """Integration through the standard :class:`deepmd.infer.DeepPot` entry.

    Locks in the contract that makes ``dp test -m frozen.pt2`` and the
    deepmd ASE calculator work on a SeZM-produced archive.  Everything
    here goes through the public backend-agnostic API —
    :class:`DeepPot` dispatches ``.pt2`` to
    :class:`deepmd.pt_expt.infer.deep_eval.DeepEval`, which since the
    metadata-only patch no longer needs ``extra/model.json``.

    Numerical tolerance is looser than the ``.pte`` pipeline tests
    because AOTInductor fuses pointwise / reduction kernels and the
    fused accumulation order differs from eager; the intent here is
    contract parity, not bitwise parity.
    """

    RTOL = 1e-5
    ATOL = 1e-7

    @classmethod
    def setUpClass(cls) -> None:
        # The ``.pt2`` archive is compiled on CPU by the fixture; AOTI
        # packages are device-locked, so ``pt_expt.DeepEval``'s input
        # preparation must also place tensors on CPU — otherwise
        # ``_pt2_runner(...)`` segfaults on dtype/device mismatch.
        # ``_prepare_inputs`` does a function-local
        # ``from deepmd.pt_expt.utils.env import DEVICE``, so patching
        # the module attribute is enough (no rebinding required).
        import deepmd.pt_expt.utils.env as _pt_expt_env

        cls._orig_pt_expt_device = _pt_expt_env.DEVICE
        _pt_expt_env.DEVICE = _CPU

        super().setUpClass()

        # Late import: building the deepmd Backend registry is cheap, but
        # doing it at collection time conflicts with the conftest
        # default-device sentinel used elsewhere in this package.
        from deepmd.infer import (
            DeepPot,
        )

        cls.dp = DeepPot(str(cls.out_path))

        # A deterministic bulk sample; coord is centred in a cubic box
        # well inside the periodic image, and the atype distribution
        # exercises both type-0 and type-1 slots of sel=[2, 2].
        rng = np.random.default_rng(2026)
        cls.natoms = 5
        cls.atype = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        box_edge = cls.params["descriptor"]["rcut"] * 3.0
        cls.coord = (
            rng.random((1, cls.natoms, 3), dtype=np.float64) * box_edge * 0.4
            + box_edge * 0.3
        )
        cls.cell = (np.eye(3, dtype=np.float64) * box_edge).reshape(1, 9)

    @classmethod
    def tearDownClass(cls) -> None:
        import deepmd.pt_expt.utils.env as _pt_expt_env

        _pt_expt_env.DEVICE = cls._orig_pt_expt_device
        super().tearDownClass()

    def _eager_energy_force_virial(self) -> tuple[np.ndarray, ...]:
        """Run the eager SeZMModel forward and return arrays shaped like DeepPot."""
        model = _build_tiny_sezm_model()
        wrapper = ModelWrapper(model, model_params=copy.deepcopy(self.params))
        raw = torch.load(self.ckpt_path, map_location=_CPU, weights_only=False)
        wrapper.load_state_dict(raw["model"])
        model.eval()

        coord_t = torch.tensor(self.coord, dtype=torch.float64).requires_grad_(True)
        atype_t = torch.tensor(self.atype, dtype=torch.int64).unsqueeze(0)
        box_t = torch.tensor(self.cell, dtype=torch.float64)
        out = model.forward(coord_t, atype_t, box_t, do_atomic_virial=True)
        return (
            out["energy"].detach().cpu().numpy(),
            out["force"].detach().cpu().numpy(),
            out["virial"].detach().cpu().numpy(),
            out["atom_energy"].detach().cpu().numpy(),
        )

    # ---- metadata accessors ----------------------------------------

    def test_deeppot_metadata_accessors(self) -> None:
        dp = self.dp
        self.assertEqual(list(dp.deep_eval.get_type_map()), self.params["type_map"])
        self.assertEqual(dp.deep_eval.get_ntypes(), len(self.params["type_map"]))
        self.assertAlmostEqual(
            dp.deep_eval.get_rcut(), self.params["descriptor"]["rcut"]
        )
        self.assertEqual(dp.deep_eval.get_dim_fparam(), 0)
        self.assertEqual(dp.deep_eval.get_dim_aparam(), 0)
        # get_sel_type() must agree with the eager model; SeZM's
        # ``dpa4_ener`` fitting head selects every type by enumerating them,
        # so the concrete value is ``list(range(ntypes))`` rather than ``[]``
        # — both are valid DeepPot conventions for "all types selected".
        eager = _build_tiny_sezm_model()
        self.assertEqual(list(dp.deep_eval.get_sel_type()), list(eager.get_sel_type()))
        self.assertFalse(dp.deep_eval.get_has_spin())

    def test_deeppot_is_metadata_only(self) -> None:
        """SeZM's .pt2 omits model.json, so the loader must take the fallback."""
        self.assertIsNone(self.dp.deep_eval._dpmodel)

    # ---- numeric parity against eager -------------------------------

    def test_deeppot_eval_matches_eager(self) -> None:
        e_ref, f_ref, v_ref, atom_e_ref = self._eager_energy_force_virial()
        e, f, v = self.dp.eval(self.coord, self.cell, self.atype, atomic=False)[:3]
        np.testing.assert_allclose(
            e,
            e_ref.reshape(e.shape),
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg="energy mismatch (DeepPot vs eager)",
        )
        np.testing.assert_allclose(
            f,
            f_ref.reshape(f.shape),
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg="force mismatch (DeepPot vs eager)",
        )
        np.testing.assert_allclose(
            v,
            v_ref.reshape(v.shape),
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg="virial mismatch (DeepPot vs eager)",
        )

    def test_deeppot_eval_atomic_matches_eager(self) -> None:
        """``atomic=True`` additionally returns atom_energy and atom_virial."""
        e_ref, _, _, atom_e_ref = self._eager_energy_force_virial()
        out = self.dp.eval(self.coord, self.cell, self.atype, atomic=True)
        e, _, _, atom_e, _ = out
        np.testing.assert_allclose(
            e,
            e_ref.reshape(e.shape),
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg="energy mismatch (atomic path)",
        )
        np.testing.assert_allclose(
            atom_e,
            atom_e_ref.reshape(atom_e.shape),
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg="atom_energy mismatch (atomic path)",
        )


class TestSeZMFreezeGuards(unittest.TestCase):
    """Error paths: detector rejections and CLI-level ``NotImplementedError``s."""

    def test_metadata_records_ntypes_when_type_map_is_empty(self) -> None:
        """Metadata-only loaders need ntypes even when no type names are exported."""
        model = _build_tiny_sezm_model()
        with mock.patch.object(model, "get_type_map", return_value=[]):
            metadata = _collect_metadata(model, ["energy"])

        self.assertEqual(metadata["type_map"], [])
        self.assertEqual(metadata["ntypes"], model.get_descriptor().get_ntypes())

    def test_charge_spin_export_sample_has_runtime_input_slot(self) -> None:
        """Charge/spin-conditioned exports should not bake defaults into the graph."""
        params = _tiny_sezm_model_params()
        params["descriptor"]["add_chg_spin_ebd"] = True
        params["descriptor"]["default_chg_spin"] = [0.0, 1.0]
        model = get_model(params).to(_CPU).eval()

        sample_inputs = _make_sample_inputs(model, nframes=5, nloc=7, device=_CPU)
        metadata = _collect_metadata(model, ["energy"])
        dynamic_shapes = _build_dynamic_shapes(sample_inputs)

        self.assertEqual(len(sample_inputs), 7)
        self.assertEqual(sample_inputs[-1].shape, (5, 2))
        self.assertEqual(len(dynamic_shapes), len(sample_inputs))
        self.assertEqual(metadata["dim_chg_spin"], 2)
        self.assertEqual(metadata["default_chg_spin"], [0.0, 1.0])

    def test_is_sezm_checkpoint_rejects_non_sezm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "ener.pt"
            torch.save(
                {"model": {"_extra_state": {"model_params": {"type": "ener"}}}},
                ckpt_path,
            )
            self.assertFalse(is_sezm_checkpoint(str(ckpt_path)))

    def test_is_sezm_checkpoint_accepts_dpa4_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "dpa4.pt"
            torch.save(
                {"model": {"_extra_state": {"model_params": {"type": "dpa4"}}}},
                ckpt_path,
            )
            self.assertTrue(is_sezm_checkpoint(str(ckpt_path)))

    def test_freeze_rejects_head_selection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "fake.pt"
            torch.save(
                {"model": {"_extra_state": {"model_params": {"type": "SeZM"}}}},
                ckpt_path,
            )
            out = Path(tmp) / "out.pt2"
            with self.assertRaises(NotImplementedError):
                freeze_sezm_to_pt2(str(ckpt_path), str(out), head="branch")

    def test_freeze_requires_head_for_multi_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_path = Path(tmp) / "multi.pt"
            torch.save(
                {
                    "model": {
                        "_extra_state": {
                            "model_params": {
                                "type": "SeZM",
                                "model_dict": {"branch": {}},
                            }
                        }
                    }
                },
                ckpt_path,
            )
            out = Path(tmp) / "out.pt2"
            with self.assertRaises(ValueError):
                freeze_sezm_to_pt2(str(ckpt_path), str(out))

    def test_freeze_accepts_multi_task_dpa4_head(self) -> None:
        """Multitask DPA4 checkpoints should export the selected branch."""

        def fake_compile(_exported: torch.export.ExportedProgram, package_path: str):
            with zipfile.ZipFile(package_path, "w") as zf:
                zf.writestr("model/data.pkl", b"")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            branch_params = _tiny_sezm_model_params()
            branch_params["type"] = "dpa4"
            branch_params["descriptor"]["type"] = "dpa4"
            params = {"model_dict": {"Domains_Alloy": branch_params}}
            model = {
                "Domains_Alloy": get_model(copy.deepcopy(branch_params)).to(_CPU).eval()
            }
            wrapper = ModelWrapper(model, model_params=copy.deepcopy(params))
            ckpt_path = tmp_path / "multi_dpa4.pt"
            torch.save({"model": wrapper.state_dict()}, ckpt_path)
            out = tmp_path / "multi_dpa4.pt2"

            self.assertTrue(is_sezm_checkpoint(str(ckpt_path)))
            with mock.patch(
                "torch._inductor.aoti_compile_and_package",
                side_effect=fake_compile,
            ):
                freeze_sezm_to_pt2(
                    str(ckpt_path), str(out), device=_CPU, head="Domains_Alloy"
                )

            with zipfile.ZipFile(str(out), "r") as zf:
                model_def = json.loads(
                    zf.read("model/extra/model_def_script.json").decode("utf-8")
                )

        self.assertEqual(model_def["type"], "dpa4")

    def test_freeze_accepts_spin_checkpoint_metadata(self) -> None:
        """SeZM spin checkpoints should export a spin-compatible pt2 contract."""

        def fake_compile(_exported: torch.export.ExportedProgram, package_path: str):
            with zipfile.ZipFile(package_path, "w") as zf:
                zf.writestr("model/data.pkl", b"")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            params = _tiny_sezm_spin_model_params()
            ckpt_path = _write_tiny_sezm_checkpoint(tmp_path, params)
            out = tmp_path / "spin.pt2"

            with mock.patch(
                "torch._inductor.aoti_compile_and_package",
                side_effect=fake_compile,
            ):
                freeze_sezm_to_pt2(str(ckpt_path), str(out), device=_CPU)

            with zipfile.ZipFile(str(out), "r") as zf:
                metadata = json.loads(
                    zf.read("model/extra/metadata.json").decode("utf-8")
                )

        self.assertTrue(metadata["is_spin"])
        self.assertEqual(metadata["type_map"], params["type_map"])
        self.assertEqual(metadata["ntypes"], len(params["type_map"]))
        self.assertEqual(metadata["dim_chg_spin"], 0)
        self.assertIsNone(metadata["default_chg_spin"])
        self.assertEqual(metadata["use_spin"], params["spin"]["use_spin"])
        self.assertEqual(metadata["ntypes_spin"], 1)
        self.assertIn("energy_derv_r_mag", metadata["output_keys"])
        self.assertIn("energy_derv_c_redu", metadata["output_keys"])


if __name__ == "__main__":
    unittest.main()
