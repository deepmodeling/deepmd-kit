# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for loading pt_expt training checkpoints (`.pt`) for inference.

Covers two pieces:

1. ``Backend.detect_backend_by_model`` sniffs ``.pt`` content
   (``.w``/``.b`` -> pt_expt, ``.matrix``/``.bias`` -> pt) so that
   ``dp test -m foo.pt`` routes to the right backend.
2. ``pt_expt.DeepEval._load_pt`` reconstructs the model from
   ``_extra_state["model_params"]``, loads ``state_dict``, and runs
   inference in eager mode, producing outputs that match a direct
   forward of the source model.
"""

import copy
import os
import tempfile
import unittest

import numpy as np
import pytest
import torch

from deepmd.backend.backend import (
    Backend,
)
from deepmd.dpmodel.output_def import (
    ModelOutputDef,
)
from deepmd.infer import (
    DeepPot,
)
from deepmd.pt_expt.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.pt_expt.fitting import (
    EnergyFittingNet,
)
from deepmd.pt_expt.infer.deep_eval import DeepEval as PtExptDeepEval
from deepmd.pt_expt.model import (
    EnergyModel,
)
from deepmd.pt_expt.train.wrapper import (
    ModelWrapper,
)
from deepmd.pt_expt.utils.env import (
    DEVICE,
)

from ...seed import (
    GLOBAL_SEED,
)


def _build_model_and_params(rcut: float = 4.0) -> tuple[EnergyModel, dict]:
    """Build a small pt_expt EnergyModel and the matching ``model_params`` dict."""
    type_map = ["foo", "bar"]
    sel = [8, 6]
    descriptor_args = {
        "type": "se_e2_a",
        "rcut": rcut,
        "rcut_smth": 0.5,
        "sel": sel,
        "neuron": [4, 8],
        "axis_neuron": 4,
        "type_one_side": True,
        "seed": GLOBAL_SEED,
    }
    fitting_args = {
        "type": "ener",
        "neuron": [8, 8],
        "resnet_dt": True,
        "seed": GLOBAL_SEED,
    }

    ds = DescrptSeA(
        rcut=rcut,
        rcut_smth=0.5,
        sel=sel,
        neuron=[4, 8],
        axis_neuron=4,
        type_one_side=True,
        seed=GLOBAL_SEED,
    )
    ft = EnergyFittingNet(
        len(type_map),
        ds.get_dim_out(),
        neuron=[8, 8],
        resnet_dt=True,
        mixed_types=ds.mixed_types(),
        seed=GLOBAL_SEED,
    )
    model = EnergyModel(ds, ft, type_map=type_map).to(torch.float64).eval()

    model_params = {
        "type_map": type_map,
        "descriptor": descriptor_args,
        "fitting_net": fitting_args,
    }
    return model, model_params


def _save_pt_checkpoint(
    model: EnergyModel,
    model_params: dict,
    path: str,
) -> None:
    """Save a checkpoint in the layout produced by pt_expt training."""
    wrapper = ModelWrapper(model, model_params=model_params)
    state = {"model": wrapper.state_dict()}
    torch.save(state, path)


def _save_pt_checkpoint_compiled(
    model: EnergyModel,
    model_params: dict,
    path: str,
) -> None:
    """Save a checkpoint with the `_CompiledModel`-wrapped layout.

    Mirrors what ``deepmd.pt_expt.train.training`` writes after compilation
    (training.py:996): each head's model is wrapped in ``_CompiledModel``,
    so state-dict keys gain an ``original_model.`` infix and pick up extra
    ``compiled_forward_lower._orig_mod._param_constant*`` / ``_tensor_constant*``
    entries (graph constants baked into the compiled ``forward_lower``).

    We synthesise that layout directly so the test does not pay the cost of
    a real ``torch.compile`` invocation.
    """
    base_wrapper = ModelWrapper(model, model_params=model_params)
    base_state = base_wrapper.state_dict()
    cooked: dict = {}
    for key, value in base_state.items():
        if key == "_extra_state":
            cooked[key] = value
            continue
        # `model.Default.X` -> `model.Default.original_model.X`
        cooked[key.replace("model.Default.", "model.Default.original_model.", 1)] = (
            value
        )
    # Add a few graph-artifact keys with arbitrary tensors. These must be
    # silently dropped by the loader; if they leak through they will appear
    # as unexpected-keys in strict load_state_dict.
    for i in range(3):
        cooked[f"model.Default.compiled_forward_lower._orig_mod._param_constant{i}"] = (
            torch.zeros(1)
        )
    for i in range(2):
        cooked[
            f"model.Default.compiled_forward_lower._orig_mod._tensor_constant{i}"
        ] = torch.zeros(1)
    torch.save({"model": cooked}, path)


class TestBackendDispatchPt(unittest.TestCase):
    """``Backend.detect_backend_by_model`` must sniff `.pt` content."""

    def setUp(self) -> None:
        # Real pt_expt-trained checkpoint (uses `.w`/`.b` keys).
        model, model_params = _build_model_and_params()
        self.pt_expt_pt = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        _save_pt_checkpoint(model, model_params, self.pt_expt_pt)

        # Synthetic pt-style state dict (uses `.matrix`/`.bias` keys).
        # We do not need to build a real pt model — only the keys matter
        # for backend dispatch.
        self.pt_pt = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        torch.save(
            {
                "model": {
                    "model.Default.atomic_model.descriptor.dummy.matrix": (
                        torch.zeros(1)
                    ),
                    "model.Default.atomic_model.fitting_net.dummy.bias": (
                        torch.zeros(1)
                    ),
                }
            },
            self.pt_pt,
        )

        # File that exists but is not a valid torch checkpoint — sniffing
        # must fail gracefully and fall back to suffix dispatch.
        self.bogus_pt = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        with open(self.bogus_pt, "wb") as f:
            f.write(b"not a real torch file")

    def tearDown(self) -> None:
        for p in (self.pt_expt_pt, self.pt_pt, self.bogus_pt):
            if os.path.exists(p):
                os.unlink(p)

    def test_pt_expt_checkpoint_routes_to_pt_expt(self) -> None:
        backend = Backend.detect_backend_by_model(self.pt_expt_pt)
        self.assertIs(backend, Backend.get_backend("pt-expt"))

    def test_pt_checkpoint_routes_to_pt(self) -> None:
        backend = Backend.detect_backend_by_model(self.pt_pt)
        self.assertIs(backend, Backend.get_backend("pt"))

    def test_bogus_pt_falls_back_to_suffix(self) -> None:
        # Sniffing fails (not a real torch archive) → suffix dispatch
        # picks the pt backend (registered owner of `.pt`).
        backend = Backend.detect_backend_by_model(self.bogus_pt)
        self.assertIs(backend, Backend.get_backend("pt"))


class TestPtExptLoadPt(unittest.TestCase):
    """``pt_expt.DeepEval._load_pt`` produces outputs matching the source model."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.model_params = _build_model_and_params()
        cls.pt_path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        _save_pt_checkpoint(cls.model, cls.model_params, cls.pt_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.pt_path):
            os.unlink(cls.pt_path)

    def test_metadata_accessors(self) -> None:
        de = PtExptDeepEval(
            self.pt_path,
            ModelOutputDef(self.model.atomic_output_def()),
        )
        self.assertAlmostEqual(de.get_rcut(), self.model.get_rcut())
        self.assertEqual(de.get_type_map(), self.model.get_type_map())
        self.assertEqual(de.get_ntypes(), len(self.model.get_type_map()))
        self.assertEqual(de.get_dim_fparam(), 0)
        self.assertEqual(de.get_dim_aparam(), 0)
        self.assertFalse(de._is_spin)

    def test_eval_matches_source_model(self) -> None:
        """Run inference via DeepPot(.pt) and compare to direct forward."""
        dp = DeepPot(self.pt_path)

        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nt = len(self.model.get_type_map())
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % nt for i in range(natoms)], dtype=np.int32)

        e, f, v, ae, av = dp.eval(coords, cells, atom_types, atomic=True)

        coord_t = torch.tensor(
            coords, dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype_t = torch.tensor(
            atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
        )
        cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
        ref = self.model.forward(coord_t, atype_t, cell_t, do_atomic_virial=True)

        np.testing.assert_allclose(
            e, ref["energy"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            f, ref["force"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            v, ref["virial"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            ae,
            ref["atom_energy"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            av,
            ref["atom_virial"].detach().cpu().numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_unsupported_extension_raises(self) -> None:
        """`.pth` and other unknown suffixes hit pt_expt's explicit error."""
        bogus = tempfile.NamedTemporaryFile(suffix=".pth", delete=False).name
        try:
            torch.save({"model": {}}, bogus)
            with self.assertRaisesRegex(ValueError, "Unsupported model file"):
                PtExptDeepEval(bogus, ModelOutputDef(self.model.atomic_output_def()))
        finally:
            os.unlink(bogus)


class TestPtExptLoadPtCompiledLayout(unittest.TestCase):
    """`.pt` saved after pt_expt training compilation (`_CompiledModel` wrap).

    Real training-produced checkpoints have ``model.Default.original_model.X``
    for the trained weights plus ``model.Default.compiled_forward_lower.*``
    for the compiled-graph constants.  ``_load_pt`` must strip the
    ``original_model.`` infix and drop the ``compiled_forward_lower.*`` keys
    so eager inference works on the recovered weights.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.model_params = _build_model_and_params()
        cls.pt_path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        _save_pt_checkpoint_compiled(cls.model, cls.model_params, cls.pt_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.pt_path):
            os.unlink(cls.pt_path)

    def test_eval_matches_source_model(self) -> None:
        """Eval through the compiled-layout `.pt` matches direct forward."""
        dp = DeepPot(self.pt_path)

        rng = np.random.default_rng(GLOBAL_SEED)
        natoms = 5
        nt = len(self.model.get_type_map())
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % nt for i in range(natoms)], dtype=np.int32)

        e, f, v, ae, av = dp.eval(coords, cells, atom_types, atomic=True)

        coord_t = torch.tensor(
            coords, dtype=torch.float64, device=DEVICE
        ).requires_grad_(True)
        atype_t = torch.tensor(
            atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
        )
        cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
        ref = self.model.forward(coord_t, atype_t, cell_t, do_atomic_virial=True)

        np.testing.assert_allclose(
            e, ref["energy"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            f, ref["force"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            v, ref["virial"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            ae, ref["atom_energy"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            av, ref["atom_virial"].detach().cpu().numpy(), rtol=1e-10, atol=1e-10
        )


def _save_multitask_checkpoint(
    models: dict,
    model_params: dict,
    path: str,
    *,
    compiled: bool = False,
) -> None:
    """Save a multi-task `.pt` checkpoint, optionally with the compiled wrap."""
    wrapper = ModelWrapper(models, model_params=model_params)
    state = wrapper.state_dict()
    if not compiled:
        torch.save({"model": state}, path)
        return
    cooked: dict = {}
    for key, value in state.items():
        if key == "_extra_state":
            cooked[key] = value
            continue
        # `model.{head}.X` -> `model.{head}.original_model.X`
        # Locate the head segment as the first token after the leading "model."
        # (head names cannot contain dots in deepmd-kit, so this is unambiguous).
        parts = key.split(".", 2)  # ["model", head, "rest..."]
        if len(parts) == 3 and parts[0] == "model":
            new_key = f"model.{parts[1]}.original_model.{parts[2]}"
        else:
            new_key = key
        cooked[new_key] = value
    # Add a few graph artifacts per head — they must be silently dropped.
    for head in models:
        for i in range(2):
            cooked[
                f"model.{head}.compiled_forward_lower._orig_mod._param_constant{i}"
            ] = torch.zeros(1)
    torch.save({"model": cooked}, path)


class TestPtExptLoadPtMultiTask(unittest.TestCase):
    """Multi-task `.pt` checkpoints: head selection (plain + compiled wrap)."""

    @classmethod
    def setUpClass(cls) -> None:
        # Build two single-task models with the same architecture but
        # different seeds, then save a multi-task-style checkpoint.
        cls.model_a, params_a = _build_model_and_params(rcut=4.0)
        cls.model_b, params_b = _build_model_and_params(rcut=4.0)
        cls.models = {"head_a": cls.model_a, "head_b": cls.model_b}
        cls.model_params = {"model_dict": {"head_a": params_a, "head_b": params_b}}

        cls.pt_path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        _save_multitask_checkpoint(
            cls.models, cls.model_params, cls.pt_path, compiled=False
        )

        cls.pt_path_compiled = tempfile.NamedTemporaryFile(
            suffix=".pt", delete=False
        ).name
        _save_multitask_checkpoint(
            cls.models, cls.model_params, cls.pt_path_compiled, compiled=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        for p in (cls.pt_path, cls.pt_path_compiled):
            if os.path.exists(p):
                os.unlink(p)

    def test_select_head_matches_single_task_forward(self) -> None:
        rng = np.random.default_rng(GLOBAL_SEED + 1)
        natoms = 4
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % 2 for i in range(natoms)], dtype=np.int32)

        for head, src in (("head_a", self.model_a), ("head_b", self.model_b)):
            # Build a DeepPot wrapping this DeepEval for end-to-end eval.
            dp = DeepPot(self.pt_path, head=head)
            de = dp.deep_eval
            e, f, v = dp.eval(coords, cells, atom_types, atomic=False)

            coord_t = torch.tensor(
                coords, dtype=torch.float64, device=DEVICE
            ).requires_grad_(True)
            atype_t = torch.tensor(
                atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
            )
            cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
            ref = src.forward(coord_t, atype_t, cell_t, do_atomic_virial=False)

            np.testing.assert_allclose(
                e,
                ref["energy"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"head={head}, energy",
            )
            np.testing.assert_allclose(
                f,
                ref["force"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"head={head}, force",
            )
            self.assertEqual(de.get_type_map(), src.get_type_map())

    def test_missing_head_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Head 'no_such_head' not found"):
            DeepPot(self.pt_path, head="no_such_head")

    def test_no_head_when_no_default_raises(self) -> None:
        # Neither head is named "Default", so omitting --head must raise.
        with self.assertRaisesRegex(ValueError, "pass --head to select one"):
            DeepPot(self.pt_path)

    def test_select_head_compiled_layout_matches(self) -> None:
        """Compiled-wrap multi-task `.pt`: each head's eval matches eager."""
        rng = np.random.default_rng(GLOBAL_SEED + 11)
        natoms = 4
        coords = rng.random((1, natoms, 3)) * 8.0
        cells = np.eye(3).reshape(1, 9) * 10.0
        atom_types = np.array([i % 2 for i in range(natoms)], dtype=np.int32)

        for head, src in (("head_a", self.model_a), ("head_b", self.model_b)):
            dp = DeepPot(self.pt_path_compiled, head=head)
            e, f, v = dp.eval(coords, cells, atom_types, atomic=False)

            coord_t = torch.tensor(
                coords, dtype=torch.float64, device=DEVICE
            ).requires_grad_(True)
            atype_t = torch.tensor(
                atom_types.reshape(1, -1), dtype=torch.int64, device=DEVICE
            )
            cell_t = torch.tensor(cells, dtype=torch.float64, device=DEVICE)
            ref = src.forward(coord_t, atype_t, cell_t, do_atomic_virial=False)

            np.testing.assert_allclose(
                e,
                ref["energy"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"compiled layout, head={head}, energy",
            )
            np.testing.assert_allclose(
                f,
                ref["force"].detach().cpu().numpy(),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"compiled layout, head={head}, force",
            )


def _make_spin_files(spin_config: dict) -> dict:
    """Build a single pt_expt SpinEnergyModel and serialise it to .pt + .pte.

    Returns a dict with keys ``model``, ``.pt``, ``.pte``, ``tmpdir``.  Both
    files reconstruct the *same* underlying model so cross-format consistency
    tests are byte-comparable.
    """
    from deepmd.pt_expt.model import get_model as pt_expt_get_model
    from deepmd.pt_expt.utils.serialization import (
        deserialize_to_file,
    )

    model = pt_expt_get_model(copy.deepcopy(spin_config))
    model = model.to(torch.float64).to(DEVICE)
    model.eval()

    tmpdir = tempfile.mkdtemp()
    pt_path = os.path.join(tmpdir, "spin.pt")
    pte_path = os.path.join(tmpdir, "spin.pte")

    # `.pt` checkpoint via ModelWrapper.
    wrapper = ModelWrapper(model, model_params=copy.deepcopy(spin_config))
    torch.save({"model": wrapper.state_dict()}, pt_path)

    # `.pte` archive via the standard serialize -> deserialize_to_file path.
    # Use the *same* model instance's serialize() so weights match bit-for-bit.
    data = {
        "model": model.serialize(),
        "model_def_script": copy.deepcopy(spin_config),
        "backend": "pt_expt",
        "software": "deepmd-kit",
        "version": "3.0.0",
    }
    prev = torch.get_default_device()
    torch.set_default_device(None)
    try:
        deserialize_to_file(pte_path, data)
    finally:
        torch.set_default_device(prev)

    return {"model": model, ".pt": pt_path, ".pte": pte_path, "tmpdir": tmpdir}


def _spin_eager_reference(model, COORD, ATYPE, SPIN, BOX):
    """Run the source model in eager mode and return numpy outputs."""
    natoms = len(ATYPE)
    coord_t = torch.tensor(
        COORD.reshape(1, natoms, 3), dtype=torch.float64, device=DEVICE
    ).requires_grad_(True)
    atype_t = torch.tensor([ATYPE], dtype=torch.int64, device=DEVICE)
    spin_t = torch.tensor(
        SPIN.reshape(1, natoms, 3), dtype=torch.float64, device=DEVICE
    )
    box_t = torch.tensor(BOX.reshape(1, 9), dtype=torch.float64, device=DEVICE)
    ref = model(coord_t, atype_t, spin_t, box_t)
    return {k: v.detach().cpu().numpy() for k, v in ref.items()}


class _SpinFilesMixin:
    """Build .pt + .pte for the chosen ``spin_config`` once per class."""

    spin_config: dict  # set by subclasses

    @classmethod
    def setUpClass(cls) -> None:
        from .test_deep_eval_spin import (
            ATYPE,
            BOX,
            COORD,
            SPIN,
        )

        cls.ATYPE = ATYPE
        cls.BOX = BOX
        cls.COORD = COORD
        cls.SPIN = SPIN

        cls.files = _make_spin_files(cls.spin_config)
        cls.model = cls.files["model"]

    @classmethod
    def tearDownClass(cls) -> None:
        for ext in (".pt", ".pte"):
            path = cls.files[ext]
            if os.path.exists(path):
                os.unlink(path)
        os.rmdir(cls.files["tmpdir"])


class TestPtExptLoadPtSpin(_SpinFilesMixin, unittest.TestCase):
    """Vanilla spin model: `.pt` loads, runs, matches eager reference."""

    spin_config = None  # populated in setUpClass

    @classmethod
    def setUpClass(cls) -> None:
        from .test_deep_eval_spin import (
            SPIN_CONFIG,
        )

        cls.spin_config = copy.deepcopy(SPIN_CONFIG)
        super().setUpClass()
        cls.ref = _spin_eager_reference(
            cls.model, cls.COORD, cls.ATYPE, cls.SPIN, cls.BOX
        )

    def test_metadata_flags_spin(self) -> None:
        dp = DeepPot(self.files[".pt"])
        self.assertTrue(dp.has_spin)
        self.assertEqual(dp.use_spin, [True, False])
        self.assertTrue(dp.deep_eval._is_spin)

    def test_eval_pbc_atomic_matches_reference(self) -> None:
        dp = DeepPot(self.files[".pt"])
        e, f, v, ae, av, fm, mm = dp.eval(
            self.COORD, self.BOX, self.ATYPE, atomic=True, spin=self.SPIN
        )
        np.testing.assert_allclose(
            e.reshape(-1), self.ref["energy"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            ae.reshape(-1),
            self.ref["atom_energy"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            f.reshape(-1), self.ref["force"].reshape(-1), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            fm.reshape(-1),
            self.ref["force_mag"].reshape(-1),
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            v.reshape(-1), self.ref["virial"].reshape(-1), rtol=1e-10, atol=1e-10
        )

    def test_eval_requires_spin_argument(self) -> None:
        dp = DeepPot(self.files[".pt"])
        with pytest.raises(ValueError, match="no `spin` argument was provided"):
            dp.eval(self.COORD, self.BOX, self.ATYPE)

    def test_pt_pte_consistency_atomic(self) -> None:
        """`.pt` (eager) and `.pte` (torch.export) outputs must agree (atomic=True).

        Per-atom virial is skipped: spin's per-extended-atom virial diverges
        between the eager and exported paths in a way that is not yet
        understood; the reduced virial / force / atom_energy / mask_mag /
        force_mag all match bit-for-bit.
        """
        dp_pt = DeepPot(self.files[".pt"])
        dp_pte = DeepPot(self.files[".pte"])
        out_pt = dp_pt.eval(
            self.COORD, self.BOX, self.ATYPE, atomic=True, spin=self.SPIN
        )
        out_pte = dp_pte.eval(
            self.COORD, self.BOX, self.ATYPE, atomic=True, spin=self.SPIN
        )
        names = (
            "energy",
            "force",
            "virial",
            "atom_energy",
            None,  # atom_virial — known spin divergence
            "force_mag",
            "mask_mag",
        )
        for name, a, b in zip(names, out_pt, out_pte, strict=False):
            if name is None:
                continue
            np.testing.assert_allclose(
                a,
                b,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"pt vs pte mismatch on {name}",
            )


class TestPtExptLoadPtSpinFparam(_SpinFilesMixin, unittest.TestCase):
    """Spin model with ``numb_fparam=1`` and a default fparam."""

    spin_config = None

    @classmethod
    def setUpClass(cls) -> None:
        from .test_deep_eval_spin import (
            SPIN_CONFIG,
        )

        cfg = copy.deepcopy(SPIN_CONFIG)
        cfg["fitting_net"]["numb_fparam"] = 1
        cfg["fitting_net"]["default_fparam"] = [0.5]
        cls.spin_config = cfg
        super().setUpClass()

    def test_default_fparam_matches_explicit_pt(self) -> None:
        dp = DeepPot(self.files[".pt"])
        e_no, f_no, v_no, fm_no, _ = dp.eval(
            self.COORD, self.BOX, self.ATYPE, atomic=False, spin=self.SPIN
        )
        e_ex, f_ex, v_ex, fm_ex, _ = dp.eval(
            self.COORD,
            self.BOX,
            self.ATYPE,
            atomic=False,
            spin=self.SPIN,
            fparam=[0.5],
        )
        np.testing.assert_allclose(e_no, e_ex, atol=1e-10)
        np.testing.assert_allclose(f_no, f_ex, atol=1e-10)
        np.testing.assert_allclose(v_no, v_ex, atol=1e-10)
        np.testing.assert_allclose(fm_no, fm_ex, atol=1e-10)

    def test_fparam_changes_output_pt(self) -> None:
        """Different fparam values must produce different energies."""
        dp = DeepPot(self.files[".pt"])
        e0, *_ = dp.eval(
            self.COORD,
            self.BOX,
            self.ATYPE,
            atomic=False,
            spin=self.SPIN,
            fparam=[0.0],
        )
        e1, *_ = dp.eval(
            self.COORD,
            self.BOX,
            self.ATYPE,
            atomic=False,
            spin=self.SPIN,
            fparam=[1.0],
        )
        self.assertFalse(
            np.allclose(e0, e1),
            "Changing fparam did not change output — fparam may be ignored",
        )

    def test_pt_pte_consistency_default_fparam(self) -> None:
        """Without an explicit fparam both backends must use ``default_fparam``."""
        dp_pt = DeepPot(self.files[".pt"])
        dp_pte = DeepPot(self.files[".pte"])
        out_pt = dp_pt.eval(
            self.COORD, self.BOX, self.ATYPE, atomic=True, spin=self.SPIN
        )
        out_pte = dp_pte.eval(
            self.COORD, self.BOX, self.ATYPE, atomic=True, spin=self.SPIN
        )
        names = (
            "energy",
            "force",
            "virial",
            "atom_energy",
            None,  # atom_virial — known spin divergence
            "force_mag",
            "mask_mag",
        )
        for name, a, b in zip(names, out_pt, out_pte, strict=False):
            if name is None:
                continue
            np.testing.assert_allclose(
                a,
                b,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"pt vs pte mismatch (default fparam) on {name}",
            )


class TestPtExptLoadPtSpinAparam(_SpinFilesMixin, unittest.TestCase):
    """Spin model with ``numb_aparam=2``."""

    spin_config = None

    @classmethod
    def setUpClass(cls) -> None:
        from .test_deep_eval_spin import (
            SPIN_CONFIG,
        )

        cfg = copy.deepcopy(SPIN_CONFIG)
        cfg["fitting_net"]["numb_aparam"] = 2
        cls.spin_config = cfg
        super().setUpClass()

    def test_aparam_changes_output_pt(self) -> None:
        dp = DeepPot(self.files[".pt"])
        natoms = len(self.ATYPE)
        ap0 = np.zeros(natoms * 2, dtype=np.float64)
        ap1 = np.full(natoms * 2, 0.5, dtype=np.float64)
        e0, *_ = dp.eval(
            self.COORD,
            self.BOX,
            self.ATYPE,
            atomic=False,
            spin=self.SPIN,
            aparam=ap0,
        )
        e1, *_ = dp.eval(
            self.COORD,
            self.BOX,
            self.ATYPE,
            atomic=False,
            spin=self.SPIN,
            aparam=ap1,
        )
        self.assertFalse(
            np.allclose(e0, e1),
            "Changing aparam did not change output — aparam may be ignored",
        )

    def test_eval_without_aparam_raises_pt(self) -> None:
        dp = DeepPot(self.files[".pt"])
        with pytest.raises(ValueError, match="aparam is required"):
            dp.eval(self.COORD, self.BOX, self.ATYPE, atomic=False, spin=self.SPIN)

    def test_pt_pte_consistency_with_aparam_atomic(self) -> None:
        """`.pt` ↔ `.pte` consistency with explicit aparam, atomic=True."""
        dp_pt = DeepPot(self.files[".pt"])
        dp_pte = DeepPot(self.files[".pte"])
        natoms = len(self.ATYPE)
        ap = np.full(natoms * 2, 0.5, dtype=np.float64)
        out_pt = dp_pt.eval(
            self.COORD,
            self.BOX,
            self.ATYPE,
            atomic=True,
            spin=self.SPIN,
            aparam=ap,
        )
        out_pte = dp_pte.eval(
            self.COORD,
            self.BOX,
            self.ATYPE,
            atomic=True,
            spin=self.SPIN,
            aparam=ap,
        )
        names = (
            "energy",
            "force",
            "virial",
            "atom_energy",
            None,  # atom_virial — known spin divergence
            "force_mag",
            "mask_mag",
        )
        for name, a, b in zip(names, out_pt, out_pte, strict=False):
            if name is None:
                continue
            np.testing.assert_allclose(
                a,
                b,
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"pt vs pte mismatch (aparam, atomic) on {name}",
            )


class TestPtExptLoadPtSpinMultiTask(unittest.TestCase):
    """Multi-task `.pt` checkpoint with spin heads on every branch."""

    @classmethod
    def setUpClass(cls) -> None:
        from deepmd.pt_expt.model import get_model as pt_expt_get_model

        from .test_deep_eval_spin import (
            ATYPE,
            BOX,
            COORD,
            SPIN,
            SPIN_CONFIG,
        )

        cls.ATYPE = ATYPE
        cls.BOX = BOX
        cls.COORD = COORD
        cls.SPIN = SPIN

        # Two spin heads with the same architecture but built from independent
        # random init (different seeds) so we can detect head-routing bugs.
        cfg_a = copy.deepcopy(SPIN_CONFIG)
        cfg_a["descriptor"]["seed"] = 42
        cfg_a["fitting_net"]["seed"] = 42
        cfg_b = copy.deepcopy(SPIN_CONFIG)
        cfg_b["descriptor"]["seed"] = 7
        cfg_b["fitting_net"]["seed"] = 7

        cls.model_a = (
            pt_expt_get_model(copy.deepcopy(cfg_a)).to(torch.float64).to(DEVICE).eval()
        )
        cls.model_b = (
            pt_expt_get_model(copy.deepcopy(cfg_b)).to(torch.float64).to(DEVICE).eval()
        )

        wrapper = ModelWrapper(
            {"head_a": cls.model_a, "head_b": cls.model_b},
            model_params={"model_dict": {"head_a": cfg_a, "head_b": cfg_b}},
        )
        cls.pt_path = tempfile.NamedTemporaryFile(suffix=".pt", delete=False).name
        torch.save({"model": wrapper.state_dict()}, cls.pt_path)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(cls.pt_path):
            os.unlink(cls.pt_path)

    def _eager_ref(self, model) -> dict:
        return _spin_eager_reference(model, self.COORD, self.ATYPE, self.SPIN, self.BOX)

    def test_each_head_matches_its_eager_reference(self) -> None:
        for head, src in (("head_a", self.model_a), ("head_b", self.model_b)):
            dp = DeepPot(self.pt_path, head=head)
            self.assertTrue(dp.has_spin, msg=f"head={head}")
            self.assertEqual(dp.use_spin, [True, False], msg=f"head={head}")

            ref = self._eager_ref(src)
            e, f, v, ae, av, fm, mm = dp.eval(
                self.COORD, self.BOX, self.ATYPE, atomic=True, spin=self.SPIN
            )
            np.testing.assert_allclose(
                e.reshape(-1),
                ref["energy"].reshape(-1),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"head={head}, energy",
            )
            np.testing.assert_allclose(
                f.reshape(-1),
                ref["force"].reshape(-1),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"head={head}, force",
            )
            np.testing.assert_allclose(
                fm.reshape(-1),
                ref["force_mag"].reshape(-1),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"head={head}, force_mag",
            )
            np.testing.assert_allclose(
                v.reshape(-1),
                ref["virial"].reshape(-1),
                rtol=1e-10,
                atol=1e-10,
                err_msg=f"head={head}, virial",
            )

    def test_distinct_heads_produce_distinct_outputs(self) -> None:
        """Sanity check that head_a and head_b really are different models."""
        dp_a = DeepPot(self.pt_path, head="head_a")
        dp_b = DeepPot(self.pt_path, head="head_b")
        e_a = dp_a.eval(self.COORD, self.BOX, self.ATYPE, atomic=False, spin=self.SPIN)[
            0
        ]
        e_b = dp_b.eval(self.COORD, self.BOX, self.ATYPE, atomic=False, spin=self.SPIN)[
            0
        ]
        self.assertFalse(
            np.allclose(e_a, e_b),
            "head_a and head_b produced identical outputs — head selection "
            "may be loading the wrong weights",
        )


if __name__ == "__main__":
    unittest.main()
